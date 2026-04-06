import copy
import functools
import os

import blobfile as bf
import torch as th
import torch.distributed as dist
import torch.utils.tensorboard
from torch.optim import AdamW
import torch.amp as amp

import itertools

from . import dist_util, logger
from .resample import LossAwareSampler, UniformSampler
from DWT_IDWT.DWT_IDWT_layer import DWT_3D, IDWT_3D

INITIAL_LOG_LOSS_SCALE = 20.0

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        in_channels,
        image_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        resume_step,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        dataset='tooth',
        summary_writer=None,
        wandb_run=None,
        mode='default',
        loss_level='image',
        target=None,
        training_mode=None,
        conditioning_image=None,
        lambda_mask,
        lambda_quality,
        lambda_quality_overall,
        val_data=None,
        validation_interval=5000,
        early_stop_patience=10,
    ):
        self.training_mode=training_mode
        self.target = target
        self.conditioning_image = conditioning_image
        self.lambda_mask = lambda_mask
        self.lambda_quality = lambda_quality
        self.lambda_quality_overall = lambda_quality_overall
        self.val_data = val_data
        self.iter_val_data = iter(val_data) if val_data is not None else None
        self.validation_interval = int(validation_interval)
        self.early_stop_patience = int(early_stop_patience)
        self.best_val_loss = float("inf")
        self.num_bad_validations = 0
        self.summary_writer = summary_writer
        self.wandb_run = wandb_run
        self.mode = mode
        self.model = model
        self.diffusion = diffusion
        self.datal = data
        self.dataset = dataset
        self.iterdatal = iter(data)
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.image_size = image_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        print(
            f"Using learning rate: {self.lr} | "
            f"Using lambda mask: {self.lambda_mask}"
        )
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        if self.use_fp16:
            self.grad_scaler = amp.GradScaler('cuda')
        else:
            self.grad_scaler = amp.GradScaler('cuda', enabled=False)

        print("Nummer of timesteps:", self.diffusion.num_timesteps)
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.dwt = DWT_3D('haar')
        self.idwt = IDWT_3D('haar')

        self.loss_level = loss_level

        self.step = 1
        self.resume_step = resume_step
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()
        
        # Get rank and device
        self.rank = dist.get_rank()
        self.device = self.model.device
        self._load_and_sync_parameters()

        self.opt = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.resume_step:
            print("Resume Step: " + str(self.resume_step))
            self._load_optimizer_state()

        if not th.cuda.is_available():
            logger.warn(
                "Training requires CUDA. "
            )

    def _next_val_batch(self):
        if self.val_data is None:
            return None
        try:
            batch = next(self.iter_val_data)
        except StopIteration:
            if hasattr(self.val_data, 'sampler') and hasattr(self.val_data.sampler, 'set_epoch'):
                self.val_data.sampler.set_epoch(self.step + self.resume_step)
            self.iter_val_data = iter(self.val_data)
            batch = next(self.iter_val_data)
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    def _build_micro_cond(self, batch, start, end):
        micro_target = batch['image'][start:end].to(self.device)
        micro_condition = None
        if self.conditioning_image != "none" and 'cond_image' in batch:
            micro_condition = batch['cond_image'][start:end].to(self.device)

        micro_label = batch['label'][start:end].to(self.device) if 'label' in batch else None
        micro_cond = {'condition': micro_condition}
        if 'diagnosis' in batch:
            micro_cond['diagnosis'] = batch['diagnosis'][start:end].to(self.device)
        if 'age' in batch:
            micro_cond['age'] = batch['age'][start:end].to(self.device)
        if 'sex' in batch:
            micro_cond['sex'] = batch['sex'][start:end].to(self.device)
        if 'quality' in batch:
            micro_cond['quality'] = batch['quality'][start:end].to(self.device)
        if 'metadata_cond' in batch:
            micro_cond['metadata_cond'] = batch['metadata_cond'][start:end].to(self.device)
        return micro_target, micro_label, micro_cond, micro_condition

    def run_validation(self):
        if self.val_data is None:
            return None, False

        self.model.eval()
        batch = self._next_val_batch()
        if batch is None:
            self.model.train()
            return None, False

        val_total = th.zeros(1, device=self.device)
        n_micro = 0
        val_recon = None
        val_input = None
        val_output = None

        with th.no_grad():
            for i in range(0, batch['image'].shape[0], self.microbatch):
                end = i + self.microbatch
                micro_target, micro_label, micro_cond, micro_condition = self._build_micro_cond(batch, i, end)
                t, _ = self.schedule_sampler.sample(micro_target.shape[0], self.device)
                autocast_enabled = self.use_fp16 and th.cuda.is_available()
                with amp.autocast('cuda', enabled=autocast_enabled):
                    losses1 = self.diffusion.training_losses(
                        self.model,
                        x_start=micro_target,
                        t=t,
                        model_kwargs=micro_cond,
                        labels=micro_label,
                        mode=self.mode,
                    )

                losses = losses1[0]
                sample_idwt = losses1[2]
                weights = th.ones(len(losses['mse_wav']), device=self.device)
                val_loss = (losses['mse_wav'] * weights).mean()
                if 'masked_mse' in losses:
                    val_loss = val_loss + self.lambda_mask * losses['masked_mse']
                    val_recon = losses['masked_mse'].detach()

                val_total += val_loss.detach()
                n_micro += 1

                if val_output is None:
                    val_output = sample_idwt.detach()
                    val_input = micro_condition.detach() if micro_condition is not None else micro_target.detach()

        val_total = val_total / max(n_micro, 1)
        if dist.is_initialized():
            dist.all_reduce(val_total, op=dist.ReduceOp.SUM)
            val_total /= dist.get_world_size()
            if val_recon is not None:
                dist.all_reduce(val_recon, op=dist.ReduceOp.SUM)
                val_recon /= dist.get_world_size()

        should_stop = False
        improved = val_total.item() < (self.best_val_loss - 1e-8)
        if improved:
            self.best_val_loss = val_total.item()
            self.num_bad_validations = 0
            if self.rank == 0:
                self.save_best(self.best_val_loss)
        else:
            self.num_bad_validations += 1
            if self.num_bad_validations >= self.early_stop_patience:
                should_stop = True

        if self.rank == 0:
            logger.logkv_mean('val/loss', float(val_total.item()))
            logger.logkv_mean('val/best_loss', float(self.best_val_loss))
            logger.logkv('val/patience_count', int(self.num_bad_validations))
            if val_recon is not None:
                logger.logkv_mean('val/reconstruction_mse', float(val_recon.item()))

            if self.summary_writer is not None:
                self.summary_writer.add_scalar('val/loss', float(val_total.item()), global_step=self.step + self.resume_step)
                self.summary_writer.add_scalar('val/best_loss', float(self.best_val_loss), global_step=self.step + self.resume_step)
                if val_recon is not None:
                    self.summary_writer.add_scalar('val/reconstruction_mse', float(val_recon.item()), global_step=self.step + self.resume_step)

            if self.wandb_run is not None:
                try:
                    import wandb
                    wandb_log = {
                        'val/loss': float(val_total.item()),
                        'val/best_loss': float(self.best_val_loss),
                        'val/patience_count': int(self.num_bad_validations),
                    }
                    if val_recon is not None:
                        wandb_log['val/reconstruction_mse'] = float(val_recon.item())

                    if val_input is not None and val_output is not None:
                        image_size = val_output.size()[2]
                        out_mid = val_output[0, 0, :, :, image_size // 2].detach().cpu().numpy()
                        in_mid = val_input[0, 0, :, :, image_size // 2].detach().cpu().numpy()
                        in_mid = (in_mid - in_mid.min()) / (in_mid.max() - in_mid.min() + 1e-8)
                        out_mid = (out_mid - out_mid.min()) / (out_mid.max() - out_mid.min() + 1e-8)
                        wandb_log['val/input_image'] = wandb.Image((in_mid * 255).astype('uint8'), caption='validation_input')
                        wandb_log['val/output_image'] = wandb.Image((out_mid * 255).astype('uint8'), caption='validation_output')
                    self.wandb_run.log(wandb_log, step=self.step + self.resume_step)
                except Exception as e:
                    logger.log(f'Failed validation wandb logging: {e}')

        self.model.train()
        stop_tensor = th.tensor([1 if should_stop else 0], device=self.device, dtype=th.int32)
        if dist.is_initialized():
            dist.broadcast(stop_tensor, src=0)
        return float(val_total.item()), bool(stop_tensor.item())

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            print('resume model ...')
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if not dist.is_initialized() or dist.get_rank() == 0:
                logger.log(f"Loading model from checkpoint: {resume_checkpoint}...")
                state_dict = th.load(self.resume_checkpoint, map_location="cpu")
                if any(k.startswith("module.") for k in state_dict.keys()):
                    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                    state_dict = new_state_dict
                self.model.module.load_state_dict(state_dict)
        if dist.is_initialized():       
            dist.barrier()
            for p in self.model.parameters():
                dist.broadcast(p.data, src=0)
            dist.barrier()
        

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        
        if main_checkpoint:
            checkpoint_name = os.path.basename(main_checkpoint)
            # Because opt is just opt_{checkpoint} hardcoded
            opt_file = f"opt_{checkpoint_name}"
            opt_checkpoint = bf.join(
                bf.dirname(main_checkpoint), opt_file
            )
            if bf.exists(opt_checkpoint):
                if dist.get_rank() == 0 or not dist.is_initialized():
                    logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
                    state_dict = th.load(opt_checkpoint, map_location='cpu')
                else: 
                    state_dict = None
                if dist.is_initialized():
                    obj_list = [state_dict]
                    dist.broadcast_object_list(obj_list, src=0)
                    state_dict = obj_list[0]
                if state_dict is not None:
                    self.opt.load_state_dict(state_dict)
            else:
                if not dist.is_initialized() or dist.get_rank() == 0:
                    print('no optimizer checkpoint exists')

    def run_loop(self):
        import time
        t = time.time()
        t_start = time.time()
        while not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps:
            self.datal.sampler.set_epoch(self.step + self.resume_step)
            t_total = time.time() - t
            t = time.time()
            step_start_time = time.time()
            try:
                batch = next(self.iterdatal)
                cond = {}
            except StopIteration:
                if hasattr(self.datal, 'sampler') and hasattr(self.datal.sampler, 'set_epoch'):
                    shuffle_seed = self.step + self.resume_step
                    self.datal.sampler.set_epoch(shuffle_seed)
                self.iterdatal = iter(self.datal)
                batch = next(self.iterdatal)
                cond = {}
                
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

            t_fwd = time.time()
            t_load = t_fwd-t

            lossmse, sample, sample_idwt = self.run_step(batch, cond)

            t_fwd = time.time()-t_fwd

            names = ["LLL", "LLH", "LHL", "LHH", "HLL", "HLH", "HHL", "HHH"]
            avg_loss = lossmse.detach()
            if dist.is_initialized():
                # Sum losses from all ranks, then divide by world size
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss /= dist.get_world_size()
                
            if self.summary_writer is not None:
                self.summary_writer.add_scalar('time/load', t_load, global_step=self.step + self.resume_step)
                self.summary_writer.add_scalar('time/forward', t_fwd, global_step=self.step + self.resume_step)
                self.summary_writer.add_scalar('time/total', t_total, global_step=self.step + self.resume_step)
                self.summary_writer.add_scalar('loss/MSE', avg_loss.item(), global_step=self.step + self.resume_step)            

                if self.step % 200 == 0:
                    image_size = sample_idwt.size()[2]
                    midplane = sample_idwt[0, 0, :, :, image_size // 2]
                    self.summary_writer.add_image('sample/x_0', midplane.unsqueeze(0),
                                                global_step=self.step + self.resume_step)

                    image_size = sample.size()[2]
                    for ch in range(8):
                        midplane = sample[0, ch, :, :, image_size // 2]
                        self.summary_writer.add_image('sample/{}'.format(names[ch]), midplane.unsqueeze(0),
                                                    global_step=self.step + self.resume_step)
                    logger.log(f"Logged images at step {self.step + self.resume_step}")
                    # Also log images to wandb if enabled
                    if self.wandb_run is not None:
                        try:
                            import wandb
                            imgs = []
                            # x_0 image (spatial reconstruction)
                            image_size = sample_idwt.size()[2]
                            midplane_x0 = sample_idwt[0, 0, :, :, image_size // 2]
                            arr = midplane_x0.detach().cpu().numpy()
                            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
                            arr = (arr * 255).astype('uint8')
                            imgs.append(wandb.Image(arr, caption='sample/x_0'))
                            # wavelet channels
                            image_size = sample.size()[2]
                            for ch in range(8):
                                mid = sample[0, ch, :, :, image_size // 2]
                                mid_arr = mid.detach().cpu().numpy()
                                mid_arr = (mid_arr - mid_arr.min()) / (mid_arr.max() - mid_arr.min() + 1e-8)
                                mid_arr = (mid_arr * 255).astype('uint8')
                                imgs.append(wandb.Image(mid_arr, caption=f'sample/{names[ch]}'))
                            self.wandb_run.log({"samples": imgs}, step=self.step + self.resume_step)
                        except Exception as e:
                            logger.log(f"Failed to log images to wandb: {e}")
            # Log scalars to wandb if enabled (even if no tensorboard)
            if self.wandb_run is not None:
                try:
                    import wandb
                    self.wandb_run.log({
                        'time/load': float(t_load),
                        'time/forward': float(t_fwd),
                        'time/total': float(t_total),
                        'loss/MSE': float(avg_loss.item()),
                    }, step=self.step + self.resume_step)
                except Exception:
                    pass

            if self.step % self.log_interval == 0 and (not dist.is_initialized() or self.rank == 0):
                step_elapsed = time.time() - step_start_time
                total_elapsed = time.time() - t_start
                avg_time_per_step = total_elapsed / self.step
                print(
                    f"[Step {self.step}] Step time: {step_elapsed:.2f}s | "
                    f"Avg per step: {avg_time_per_step:.2f}s | "
                    f"Total elapsed: {total_elapsed/60:.1f} min"
                )
                logger.dumpkvs()
            
            if self.step % 100 == 0:
                print(f"[Rank {self.rank}] Step {self.step} — processed {(self.step + self.resume_step) * self.batch_size} samples.")

            if self.validation_interval > 0 and self.val_data is not None and self.step % self.validation_interval == 0:
                val_loss, should_stop = self.run_validation()
                if self.rank == 0 and val_loss is not None:
                    print(
                        f"[Validation {self.step}] val_loss={val_loss:.6f} | "
                        f"best={self.best_val_loss:.6f} | "
                        f"patience={self.num_bad_validations}/{self.early_stop_patience}"
                    )
                if should_stop:
                    if self.rank == 0:
                        logger.log("Early stopping triggered by validation patience.")
                        self.save()
                    if dist.is_initialized():
                        dist.barrier()
                    return

            if self.step % self.save_interval == 0 and self.rank==0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1

        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            if self.rank == 0:
                self.save()
            if dist.is_initialized():
                dist.barrier()

    def run_step(self, batch, cond, label=None, info=dict()):
        lossmse, sample, sample_idwt = self.forward_backward(batch, cond, label)

        if self.use_fp16:
            self.grad_scaler.unscale_(self.opt)  # check self.grad_scaler._per_optimizer_states

        # compute norms
        with th.no_grad():
            params = [p for p in self.model.parameters()]
            param_max_norm = max([p.abs().max().item() for p in params], default=0.0)
            grads = [p.grad for p in params if p.grad is not None]
            grad_max_norm = max([g.abs().max().item() for g in grads], default=0.0)
            info['norm/param_max'] = param_max_norm
            info['norm/grad_max'] = grad_max_norm

        if not th.isfinite(lossmse): #infinite
            if not th.isfinite(th.tensor(param_max_norm)):
                logger.log(f"Rank {self.rank}: Model parameters non-finite {param_max_norm}", level=logger.ERROR)
                breakpoint()
            else:
                logger.log(f"Rank {self.rank}: Model parameters are finite, but loss is not: {lossmse}"
                           "\n -> update will be skipped in grad_scaler.step()", level=logger.WARN)

        if self.use_fp16:
            self.grad_scaler.step(self.opt)
            self.grad_scaler.update()
            info['scale'] = self.grad_scaler.get_scale()
        else:
            self.opt.step()
        self._anneal_lr()
        self.log_step()
        
        # Log norm metrics to wandb
        if self.wandb_run is not None and self.rank == 0:
            try:
                import wandb
                norms_dict = {
                    'norms/param_max': info.get('norm/param_max', 0.0),
                    'norms/grad_max': info.get('norm/grad_max', 0.0),
                }
                if 'scale' in info:
                    norms_dict['fp16/grad_scale'] = info['scale']
                self.wandb_run.log(norms_dict, step=self.step + self.resume_step)
            except Exception:
                pass
        
        return lossmse, sample, sample_idwt

    def forward_backward(self, batch, cond, label=None):
        for p in self.model.parameters():  # Zero out gradient
            p.grad = None
        
        target_img = 'image'  

        if self.conditioning_image != "none":
            condition_image = 'cond_image'
        else:
            condition_image = None
        for i in range(0, batch[target_img].shape[0], self.microbatch):
            micro_target = batch[target_img][i: i + self.microbatch].to(self.device)
            if condition_image is not None: 
                micro_condition = batch[condition_image][i: i + self.microbatch].to(self.device)
            else:
                micro_condition = None
            
            if 'label' in batch:
                micro_label = batch['label'][i: i + self.microbatch].to(self.device) # The mask label used for masked loss
            else:
                micro_label = None
            
            # Extract clinical conditioning (MRI metadata)
            if 'diagnosis' in batch:
                micro_diagnosis = batch['diagnosis'][i: i + self.microbatch].to(self.device)
            else:
                micro_diagnosis = None
            
            if 'age' in batch:
                micro_age = batch['age'][i: i + self.microbatch].to(self.device)
            else:
                micro_age = None
            
            if 'sex' in batch:
                micro_sex = batch['sex'][i: i + self.microbatch].to(self.device)
            else:
                micro_sex = None

            if 'quality' in batch:
                micro_quality = batch['quality'][i: i + self.microbatch].to(self.device)
            else:
                micro_quality = None

            if 'metadata_cond' in batch:
                micro_metadata_cond = batch['metadata_cond'][i: i + self.microbatch].to(self.device)
            else:
                micro_metadata_cond = None
            
            if cond is not None:
                micro_cond = {k: v[i: i + self.microbatch].to(self.device) for k, v in cond.items()}
            else:
                micro_cond = {}
            
            micro_cond['condition'] = micro_condition # Conditioning image
            if micro_diagnosis is not None:
                micro_cond['diagnosis'] = micro_diagnosis
            if micro_age is not None:
                micro_cond['age'] = micro_age
            if micro_sex is not None:
                micro_cond['sex'] = micro_sex
            if micro_quality is not None:
                micro_cond['quality'] = micro_quality
            if micro_metadata_cond is not None:
                micro_cond['metadata_cond'] = micro_metadata_cond

            last_batch = (i + self.microbatch) >= batch[target_img].shape[0]
            t, weights = self.schedule_sampler.sample(micro_target.shape[0], self.device)
            
            compute_losses = functools.partial(self.diffusion.training_losses,
                                               self.model,
                                               x_start=micro_target,
                                               t=t,
                                               model_kwargs=micro_cond,
                                               labels=micro_label,
                                               mode=self.mode,
                                               )
            autocast_enabled = self.use_fp16 and th.cuda.is_available()
            with amp.autocast('cuda', enabled=autocast_enabled):
                losses1 = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses1["loss"].detach()
                )

            losses = losses1[0]         # Loss value
            sample = losses1[1]         # Denoised subbands at t=0
            sample_idwt = losses1[2]    # Inverse wavelet transformed denoised subbands at t=0


            # We have to aggregates losses across devices
            mse_wav = losses["mse_wav"].clone().detach()
            dist.all_reduce(mse_wav)
            mse_wav.div_(dist.get_world_size())
            if "masked_mse" in losses:
                masked_mse = losses["masked_mse"].clone().detach()
                dist.all_reduce(masked_mse)
                masked_mse.div_(dist.get_world_size())     
            else:
                masked_mse = None       

            weights = th.ones(len(losses["mse_wav"]), device=self.device)# Equally weight all wavelet channel losses
            
            if self.rank == 0:
                # Log wavelet level loss
                if self.summary_writer is not None:
                    self.summary_writer.add_scalar('loss/mse_wav_lll', mse_wav[0].item(),
                                                global_step=self.step + self.resume_step)
                    self.summary_writer.add_scalar('loss/mse_wav_llh', mse_wav[1].item(),
                                                global_step=self.step + self.resume_step)
                    self.summary_writer.add_scalar('loss/mse_wav_lhl', mse_wav[2].item(),
                                                global_step=self.step + self.resume_step)
                    self.summary_writer.add_scalar('loss/mse_wav_lhh', mse_wav[3].item(),
                                                global_step=self.step + self.resume_step)
                    self.summary_writer.add_scalar('loss/mse_wav_hll', mse_wav[4].item(),
                                                global_step=self.step + self.resume_step)
                    self.summary_writer.add_scalar('loss/mse_wav_hlh', mse_wav[5].item(),
                                                global_step=self.step + self.resume_step)
                    self.summary_writer.add_scalar('loss/mse_wav_hhl', mse_wav[6].item(),
                                                global_step=self.step + self.resume_step)
                    self.summary_writer.add_scalar('loss/mse_wav_hhh', mse_wav[7].item(),
                                                global_step=self.step + self.resume_step)
                if masked_mse is not None:
                    self.summary_writer.add_scalar('loss/reconstruction_mse', masked_mse.item(), global_step=self.step + self.resume_step)
                    logger.logkv_mean("reconstruction_mse", masked_mse.item())
                log_loss_dict(self.diffusion, t, {"mse_wav": mse_wav * weights.to(self.device)})
                
                # Log all wavelet losses and quartile losses to wandb
                if self.wandb_run is not None:
                    try:
                        import wandb
                        wavelet_names = ["LLL", "LLH", "LHL", "LHH", "HLL", "HLH", "HHL", "HHH"]
                        wandb_log_dict = {}
                        # Log per-subband wavelet losses
                        for ch_idx, ch_name in enumerate(wavelet_names):
                            wandb_log_dict[f'loss/mse_wav_{ch_name}'] = mse_wav[ch_idx].item()
                        # Log masked loss if present
                        if masked_mse is not None:
                            wandb_log_dict['loss/reconstruction_mse'] = masked_mse.item()
                        # Log quartile losses
                        for sub_t, sub_loss in zip(t.cpu().numpy(), (mse_wav * weights.to(self.device)).detach().cpu().numpy()):
                            quartile = int(4 * sub_t / self.diffusion.num_timesteps)
                            key = f'loss/mse_wav_q{quartile}'
                            if key not in wandb_log_dict:
                                wandb_log_dict[key] = []
                            wandb_log_dict[key].append(float(sub_loss))
                        # Average quartile losses
                        for key in list(wandb_log_dict.keys()):
                            if key.startswith('loss/mse_wav_q') and isinstance(wandb_log_dict[key], list):
                                wandb_log_dict[key] = sum(wandb_log_dict[key]) / len(wandb_log_dict[key])
                        self.wandb_run.log(wandb_log_dict, step=self.step + self.resume_step)
                    except Exception:
                        pass
                
            loss = (losses["mse_wav"] * weights).mean()
        
            # If we have a mask loss, add it to the total loss and lambda_mask specified in run.sh file or default 10.0
            if "masked_mse" in losses:
                loss = loss + self.lambda_mask * losses["masked_mse"]
            
            lossmse = loss.detach()
            
            # perform some finiteness checks
            if not th.isfinite(loss):
                logger.log(f"Rank {self.rank}: Encountered non-finite loss {loss}")
            if self.use_fp16:
                self.grad_scaler.scale(loss).backward()
            else:
                loss.backward()
            return lossmse.detach(), sample, sample_idwt

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            logger.logkv("step", self.step + self.resume_step)
            logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, state_dict):
            if not dist.is_initialized() or dist.get_rank() == 0:
                logger.log("Saving model...")
                # Build a generic filename that encodes dataset, target and conditioning.
                cond_str = "none"
                if self.conditioning_image is not None and self.conditioning_image != "none":
                    cond_str = str(self.conditioning_image)
                dataset_str = str(self.dataset)
                target_str = str(self.target) if self.target is not None else "target"
                filename = f"{dataset_str}_target_{target_str}_cond_{cond_str}_{(self.step+self.resume_step):06d}.pt"

                with bf.BlobFile(bf.join(get_blob_logdir(), 'checkpoints', filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.model.state_dict())


        #The opt is hardcoded to tooth now, can be adjusted if name of dataset changes, or expanded if necesarry.
        if not dist.is_initialized() or dist.get_rank() == 0:
            checkpoint_dir = os.path.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            # Generic optimizer filename matching the model filename convention
            cond_str = "none"
            if self.conditioning_image is not None and self.conditioning_image != "none":
                cond_str = str(self.conditioning_image)
            dataset_str = str(self.dataset)
            target_str = str(self.target) if self.target is not None else "target"
            optfilename = f"opt_{dataset_str}_target_{target_str}_cond_{cond_str}_{(self.step + self.resume_step):06d}.pt"
            with bf.BlobFile(
                bf.join(checkpoint_dir, optfilename),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

    def save_best(self, val_loss):
        if dist.is_initialized() and dist.get_rank() != 0:
            return

        logger.log(f"Saving new best model (val_loss={val_loss:.6f})...")
        checkpoint_dir = os.path.join(logger.get_dir(), 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)

        cond_str = "none"
        if self.conditioning_image is not None and self.conditioning_image != "none":
            cond_str = str(self.conditioning_image)
        dataset_str = str(self.dataset)
        target_str = str(self.target) if self.target is not None else "target"

        model_name = f"{dataset_str}_target_{target_str}_cond_{cond_str}_best.pt"
        opt_name = f"opt_{dataset_str}_target_{target_str}_cond_{cond_str}_best.pt"

        with bf.BlobFile(bf.join(checkpoint_dir, model_name), "wb") as f:
            th.save(self.model.state_dict(), f)
        with bf.BlobFile(bf.join(checkpoint_dir, opt_name), "wb") as f:
            th.save(self.opt.state_dict(), f)

def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """

    split = os.path.basename(filename)
    split = split.split(".")[-2]  # remove extension
    split = split.split("_")[-1]  # remove possible underscores, keep only last word
    # extract trailing number
    reversed_split = []
    for c in reversed(split):
        if not c.isdigit():
            break
        reversed_split.append(c)
    split = ''.join(reversed(reversed_split))
    split = ''.join(c for c in split if c.isdigit())  # remove non-digits
    try:
        return int(split)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
