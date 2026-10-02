from utils.wan_wrapper import WanDiffusionWrapper
from utils.scheduler import SchedulerInterface
from typing import List, Optional
import torch
import torch.distributed as dist


class SelfForcingTrainingPipeline:
    def __init__(self,
                 denoising_step_list: List[int],
                 scheduler: SchedulerInterface,
                 generator: WanDiffusionWrapper,
                 num_frame_per_block=3,
                 independent_first_frame: bool = False,
                 same_step_across_blocks: bool = False,
                 last_step_only: bool = False,
                 num_max_frames: int = 21,
                 context_noise: int = 0,
                 denoising_step_list_first_chunk: Optional[List[int]] = None,
                 image_or_video_shape: Optional[List[int]] = None,
                 **kwargs):
        super().__init__()
        self.scheduler = scheduler
        self.generator = generator
        self.denoising_step_list = denoising_step_list
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]  # remove the zero timestep for inference

        # Optional: dedicated schedule for the first chunk (block 0).
        # None means all blocks share `denoising_step_list` (backwards compatible).
        self.denoising_step_list_first_chunk = denoising_step_list_first_chunk
        if self.denoising_step_list_first_chunk is not None and self.denoising_step_list_first_chunk[-1] == 0:
            self.denoising_step_list_first_chunk = self.denoising_step_list_first_chunk[:-1]

        # Wan specific hyperparameters
        self.num_transformer_blocks = 30
        latent_height, latent_width = (
            image_or_video_shape[-2:] if image_or_video_shape is not None else (60, 104)
        )
        _, patch_h, patch_w = getattr(getattr(self.generator, "model", None), "patch_size", (1, 2, 2))
        self.frame_seq_length = latent_height * latent_width // (patch_h * patch_w)
        self.num_frame_per_block = num_frame_per_block
        self.context_noise = context_noise
        self.i2v = False

        self.kv_cache1 = None
        self.kv_cache2 = None
        self.independent_first_frame = independent_first_frame
        self.same_step_across_blocks = same_step_across_blocks
        self.last_step_only = last_step_only
        self.kv_cache_size = num_max_frames * self.frame_seq_length

    def generate_and_sync_list(self, num_blocks, num_denoising_steps, device):
        rank = dist.get_rank() if dist.is_initialized() else 0

        if rank == 0:
            # Generate random indices
            indices = torch.randint(
                low=0,
                high=num_denoising_steps,
                size=(num_blocks,),
                device=device
            )
            # In our training, self.last_step_only is False
            if self.last_step_only:
                indices = torch.ones_like(indices) * (num_denoising_steps - 1)
        else:
            indices = torch.empty(num_blocks, dtype=torch.long, device=device)

        dist.broadcast(indices, src=0)  # Broadcast the random indices to all ranks
        return indices.tolist()

    def inference_with_trajectory(
            self,
            noise: torch.Tensor,
            clean_image_or_video: torch.Tensor = None, # same shape as noise
            initial_latent: Optional[torch.Tensor] = None,
            return_sim_step: bool = False,
            **conditional_dict
    ) -> torch.Tensor:
        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Step 1: Initialize KV cache to all zeros
        self._initialize_kv_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )


        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None: # Never met
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
            output[:, :1] = initial_latent
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=initial_latent,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length
                )
            current_start_frame += 1

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        # In out training, self.independent_first_frame is False
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        num_denoising_steps = len(self.denoising_step_list)

        # When a dedicated first-chunk schedule is configured, sample an exit
        # flag for the first chunk over its own schedule length, and the rest
        # over the default schedule. Otherwise fall back to the original single
        # sample covering all blocks.
        if self.denoising_step_list_first_chunk is not None:
            num_denoising_steps_first = len(self.denoising_step_list_first_chunk)
            exit_flag_first = self.generate_and_sync_list(1, num_denoising_steps_first, device=noise.device)[0]
            exit_flags_other = self.generate_and_sync_list(
                len(all_num_frames) - 1, num_denoising_steps, device=noise.device)
            exit_flags = None
        else:
            exit_flag_first = None
            exit_flags_other = None
            exit_flags = self.generate_and_sync_list(len(all_num_frames), num_denoising_steps, device=noise.device)
        start_gradient_frame_index = num_output_frames - 21

        # for block_index in range(num_blocks):
        for block_index, current_num_frames in enumerate(all_num_frames):

            if True:
                noisy_input = noise[
                    :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

                # Select denoising schedule for this block. Block 0 may use a
                # dedicated schedule when configured; otherwise all blocks share
                # `denoising_step_list`.
                current_denoising_list = (
                    self.denoising_step_list_first_chunk
                    if block_index == 0 and self.denoising_step_list_first_chunk is not None
                    else self.denoising_step_list
                )

                # Select the exit-step index for this block.
                if exit_flags_other is not None:
                    # First-chunk schedule is active.
                    if block_index == 0:
                        current_exit_flag_index = exit_flag_first
                    elif self.same_step_across_blocks:
                        current_exit_flag_index = exit_flags_other[0]
                    else:
                        current_exit_flag_index = exit_flags_other[block_index - 1]
                else:
                    # Original path: one shared sample across all blocks.
                    if self.same_step_across_blocks:
                        current_exit_flag_index = exit_flags[0]
                    else:
                        current_exit_flag_index = exit_flags[block_index]

                # Step 3.1: Spatial denoising loop
                # Such a loop corresponds to the truncated denoising algorithm:
                #    T -> \tau_1 -> \tau_2 ->...-> \tau —— enable grad ——> 0
                # For many-step model, we certainly cannot use this method, but for 4-step DMD,
                # we can inherit it for a fair comaprison. Note that as long as the conditions
                # are clean GT rather than self-generated frames, we can perform TF. So this
                # method does not conflict with TF in the frame- dimension.
                for index, current_timestep in enumerate(current_denoising_list):
                    exit_flag = (index == current_exit_flag_index)
                    timestep = torch.ones(
                        [batch_size, current_num_frames],
                        device=noise.device,
                        dtype=torch.int64) * current_timestep

                    if not exit_flag:
                        with torch.no_grad():
                            _, denoised_pred = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=conditional_dict,
                                timestep=timestep,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame * self.frame_seq_length
                            )
                            next_timestep = current_denoising_list[index + 1]
                            noisy_input = self.scheduler.add_noise(
                                denoised_pred.flatten(0, 1),
                                torch.randn_like(denoised_pred.flatten(0, 1)),
                                next_timestep * torch.ones(
                                    [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                            ).unflatten(0, denoised_pred.shape[:2])
                    else:
                        # for getting real output
                        # with torch.set_grad_enabled(current_start_frame >= start_gradient_frame_index):
                        if current_start_frame < start_gradient_frame_index: # Always True as long as we train 21 latent frames
                            with torch.no_grad():
                                _, denoised_pred = self.generator(
                                    noisy_image_or_video=noisy_input,
                                    conditional_dict=conditional_dict,
                                    timestep=timestep,
                                    kv_cache=self.kv_cache1,
                                    crossattn_cache=self.crossattn_cache,
                                    current_start=current_start_frame * self.frame_seq_length
                                )
                        else: # enable grad
                            _, denoised_pred = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=conditional_dict,
                                timestep=timestep,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame * self.frame_seq_length
                            )
                        break

            # Step 3.2: record the model's output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # Step 3.3: rerun with timestep zero to update the cache
            context_timestep = torch.ones_like(timestep) * self.context_noise
            # add context noise
            denoised_pred = self.scheduler.add_noise(
                denoised_pred.flatten(0, 1),
                torch.randn_like(denoised_pred.flatten(0, 1)),
                context_timestep * torch.ones(
                    [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
            ).unflatten(0, denoised_pred.shape[:2])
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=denoised_pred,
                    conditional_dict=conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length
                )

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        # Step 3.5: Return the denoised timestep
        # DMD's timestep sampling must align with the schedule that actually
        # carries gradient for the non-first blocks (which produce most of the
        # output). When a first-chunk schedule is active, use the "other" exit
        # flag over `denoising_step_list`; otherwise use the original shared one.
        if exit_flags_other is not None:
            final_exit_flag = exit_flags_other[0] if self.same_step_across_blocks else None
        else:
            final_exit_flag = exit_flags[0] if self.same_step_across_blocks else None

        if not self.same_step_across_blocks:  # Useless, never met
            denoised_timestep_from, denoised_timestep_to = None, None
        # T -> \tau_1 -> \tau_2 ->...-> \tau —— enable grad ——> 0
        # denoised_timestep_from = \tau
        # denoised_timestep_to = next timestep smaller than \tau
        # These are just engineering tricks
        # to align DMD timestep sampling with the actual denoising range used by the generator
        elif final_exit_flag == len(self.denoising_step_list) - 1:
            # corner case when \tau is the smallest non-zero timestep
            denoised_timestep_to = 0
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[final_exit_flag].cuda()).abs(), dim=0).item()
        else:
            denoised_timestep_to = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[final_exit_flag + 1].cuda()).abs(), dim=0).item()
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[final_exit_flag].cuda()).abs(), dim=0).item()

        if return_sim_step:  # False
            return output, denoised_timestep_from, denoised_timestep_to, final_exit_flag + 1

        return output, denoised_timestep_from, denoised_timestep_to

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache
