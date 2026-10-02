import gc
import logging
from contextlib import ExitStack
from utils.dataset import cycle
from utils.dataset import TextDataset
from utils.distributed import EMA_FSDP, fsdp_wrap, fsdp_state_dict, launch_distributed_job
from utils.misc import set_seed
import torch.distributed as dist
from omegaconf import OmegaConf
from model import DMD
import torch
import wandb
import time
import os

from trainer.inference_hook import LogStepInferenceHook


class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.causal = config.causal
        self.disable_wandb = config.disable_wandb

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        if self.is_main_process and not self.disable_wandb:
            wandb.login(host=config.wandb_host, key=config.wandb_key)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                mode="online",
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir
            )

        self.output_path = config.logdir

        # Step 2: Initialize the model and optimizer
        if config.distribution_loss == "dmd":
            self.model = DMD(config, device=self.device)
        else:
            raise ValueError("Invalid distribution matching loss")

        # Save pretrained model state_dicts to CPU
        self.fake_score_state_dict_cpu = self.model.fake_score.state_dict()

        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy,
            cpu_offload=False
        )

        self.model.real_score = fsdp_wrap(
            self.model.real_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.real_score_fsdp_wrap_strategy,
            cpu_offload=False
        )

        self.model.fake_score = fsdp_wrap(
            self.model.fake_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.fake_score_fsdp_wrap_strategy,
            cpu_offload=False
        )

        if getattr(self.model, "qwen_image_fake_critic_enabled", False):
            qwen_cfg = getattr(config, "qwen_image_teacher")
            self.model.qwen_image_fake_score = fsdp_wrap(
                self.model.qwen_image_fake_score,
                sharding_strategy=config.sharding_strategy,
                mixed_precision=config.mixed_precision,
                wrap_strategy=getattr(
                    qwen_cfg,
                    "fake_critic_fsdp_wrap_strategy",
                    config.fake_score_fsdp_wrap_strategy,
                ),
                cpu_offload=getattr(qwen_cfg, "fake_critic_cpu_offload", False),
            )

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", False)
        )

        if getattr(self.model, "qwen_image_teacher_enabled", False):
            qwen_cfg = getattr(config, "qwen_image_teacher")
            self.model.qwen_image_score = fsdp_wrap(
                self.model.qwen_image_score,
                sharding_strategy=config.sharding_strategy,
                mixed_precision=config.mixed_precision,
                wrap_strategy=getattr(qwen_cfg, "fsdp_wrap_strategy", "size"),
                cpu_offload=getattr(qwen_cfg, "cpu_offload", False)
            )

        if (not config.no_visualize) or config.load_raw_video or getattr(config, "log_inference", False):
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        critic_lr = config.lr_critic if hasattr(config, "lr_critic") else config.lr
        critic_param_groups = [{
            "params": [
                param for param in self.model.fake_score.parameters()
                if param.requires_grad
            ],
            "lr": critic_lr,
        }]
        if getattr(self.model, "qwen_image_fake_critic_enabled", False):
            qwen_cfg = getattr(config, "qwen_image_teacher")
            critic_param_groups.append({
                "params": [
                    param for param in self.model.qwen_image_fake_score.parameters()
                    if param.requires_grad
                ],
                "lr": float(getattr(qwen_cfg, "fake_critic_lr", critic_lr)),
            })
        self.critic_optimizer = torch.optim.AdamW(
            critic_param_groups,
            lr=critic_lr,
            betas=(config.beta1_critic, config.beta2_critic),
            weight_decay=config.weight_decay
        )

        # Step 3: Initialize the dataloader
        dataset = TextDataset(config.data_path)
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=8)

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)

        ##############################################################################################################
        # 6. Set up EMA parameter containers
        rename_param = (
            lambda name: name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        self.name_to_trainable_params = {}
        for n, p in self.model.generator.named_parameters():
            if not p.requires_grad:
                continue

            renamed_n = rename_param(n)
            self.name_to_trainable_params[renamed_n] = p
        ema_weight = config.ema_weight
        self.generator_ema = None
        if (ema_weight is not None) and (ema_weight > 0.0):
            print(f"Setting up EMA with weight {ema_weight}")
            self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)

        ##############################################################################################################
        # 7. (If resuming) Load the model and optimizer, lr_scheduler, ema's statedicts
        if getattr(config, "generator_ckpt", False):
            print(f"Loading pretrained generator from {config.generator_ckpt}")
            state_dict = torch.load(config.generator_ckpt, map_location="cpu")
            if "generator" in state_dict:
                state_dict = state_dict["generator"]
                fixed = {}
                for k, v in state_dict.items():
                    if k.startswith("model._fsdp_wrapped_module."):
                        k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
                    fixed[k] = v
                state_dict = fixed
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            elif "generator_ema" in state_dict:
                gen_sd = state_dict["generator_ema"]
                fixed = {}
                for k, v in gen_sd.items():
                    if k.startswith("model._fsdp_wrapped_module."):
                        k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
                    fixed[k] = v
                state_dict = fixed
            self.model.generator.load_state_dict(
                state_dict, strict=True
            )

        ##############################################################################################################

        # Let's delete EMA params for early steps to save some computes at training and inference
        if self.step < config.ema_start_step:
            self.generator_ema = None

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.previous_time = None
        self.gradient_accumulation_steps = int(
            getattr(config, "gradient_accumulation_steps", getattr(config, "ga", 1))
        )
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps/ga must be >= 1")
        self.log_inference_hook = (
            LogStepInferenceHook(self)
            if getattr(config, "log_inference", False)
            else None
        )

    def save(self):
        print("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(
            self.model.generator)
        critic_state_dict = fsdp_state_dict(
            self.model.fake_score)
        qwen_fake_critic_state_dict = None
        if getattr(self.model, "qwen_image_fake_critic_enabled", False):
            qwen_fake_critic_state_dict = fsdp_state_dict(
                self.model.qwen_image_fake_score
            )

        if self.config.ema_start_step < self.step:
            state_dict = {
                "generator_ema": self.generator_ema.full_state_dict(self.model.generator),
            }
        else:
            state_dict = {
                "generator": generator_state_dict,
            }

        if self.is_main_process:
            checkpoint_dir = os.path.join(
                self.output_path,
                f"checkpoint_model_{self.step:06d}",
            )
            model_path = os.path.join(checkpoint_dir, "model.pt")
            critic_path = os.path.join(checkpoint_dir, "critic.pt")
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(state_dict, model_path)
            critic_checkpoint = {"fake_score": critic_state_dict}
            if qwen_fake_critic_state_dict is not None:
                critic_checkpoint["qwen_image_fake_score"] = (
                    qwen_fake_critic_state_dict
                )
            torch.save(critic_checkpoint, critic_path)
            print("Model saved to", model_path)
            print("Fake score saved to", critic_path)

    def save_critic(self):
        print("Start gathering distributed model states...")

        critic_state_dict = fsdp_state_dict(
            self.model.fake_score)
        qwen_fake_critic_state_dict = None
        if getattr(self.model, "qwen_image_fake_critic_enabled", False):
            qwen_fake_critic_state_dict = fsdp_state_dict(
                self.model.qwen_image_fake_score
            )


        state_dict = critic_state_dict
        if qwen_fake_critic_state_dict is not None:
            state_dict = {
                "fake_score": critic_state_dict,
                "qwen_image_fake_score": qwen_fake_critic_state_dict,
            }

        if self.is_main_process:
            os.makedirs(os.path.join(self.output_path,
                        f"checkpoint_model_{self.step:06d}"), exist_ok=True)
            torch.save(state_dict, os.path.join(self.output_path,
                       f"checkpoint_model_{self.step:06d}", "model.pt"))
            print("Model saved to", os.path.join(self.output_path,
                  f"checkpoint_model_{self.step:06d}", "model.pt"))

    def fwdbwd_one_step(self, batch, train_generator, clean_latent=None, loss_scale=1.0, clip_grad=True):
        self.model.eval()  # prevent any randomness (e.g. dropout)

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]
        if self.config.i2v:
            # clean_latent = None #original code here
            image_latent = batch["ode_latent"][:, -1][:, 0:1, ].to(
                device=self.device, dtype=self.dtype)
        else:
            # clean_latent = None #original code here
            image_latent = None

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(
                text_prompts=text_prompts)

            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach()
                                      for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

        # Step 3: Store gradients for the generator (if training the generator)
        if train_generator:
            generator_loss, generator_log_dict = self.model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                initial_latent=image_latent if self.config.i2v else None,
                text_prompts=text_prompts,
                current_step=self.step,
            )

            (generator_loss * loss_scale).backward()

            generator_log_dict.update({"generator_loss": generator_loss})
            if clip_grad:
                generator_grad_norm = self.model.generator.clip_grad_norm_(
                    self.max_grad_norm_generator)
                generator_log_dict.update({"generator_grad_norm": generator_grad_norm})

            return generator_log_dict
        else:
            generator_log_dict = {}

        # Step 4: Store gradients for the critic (if training the critic)
        critic_loss, critic_log_dict = self.model.critic_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=image_latent if self.config.i2v else None
        )

        (critic_loss * loss_scale).backward()

        critic_log_dict.update({"critic_loss": critic_loss})
        if clip_grad:
            critic_grad_norm = self.model.fake_score.clip_grad_norm_(
                self.max_grad_norm_critic)
            critic_log_dict.update({"critic_grad_norm": critic_grad_norm})
            if getattr(self.model, "qwen_image_fake_critic_enabled", False):
                qwen_fake_critic_grad_norm = (
                    self.model.qwen_image_fake_score.clip_grad_norm_(
                        self.max_grad_norm_critic
                    )
                )
                critic_log_dict.update({
                    "qwen_image_fake_critic_grad_norm": (
                        qwen_fake_critic_grad_norm
                    )
                })

        return critic_log_dict

    def _mean_log_dict(self, log_dicts):
        merged = {}
        if not log_dicts:
            return merged
        for key in log_dicts[0].keys():
            values = [log_dict[key] for log_dict in log_dicts if key in log_dict]
            if not values:
                continue
            if torch.is_tensor(values[0]):
                stacked = torch.stack([value.detach() for value in values])
                if key.startswith("qwen_image_") and key.endswith("_min"):
                    merged[key] = stacked.min()
                elif key.startswith("qwen_image_") and key.endswith("_max"):
                    merged[key] = stacked.max()
                else:
                    merged[key] = stacked.mean()
            else:
                merged[key] = sum(values) / len(values)
        return merged

    def _global_mean_probe_logs(self, log_dict):
        """Aggregate scalar diagnostic logs across data-parallel ranks.

        Means/fractions use a packed average; extrema use packed MIN/MAX
        collectives. This changes logging only; none of these detached values
        participate in optimization.
        """
        if self.world_size <= 1:
            return
        keys = sorted(
            key for key, value in log_dict.items()
            if key.startswith((
                "qwen_image_transition_",
                "qwen_image_probe_",
                "qwen_image_frame_idx_",
                "qwen_image_frame_offset_",
                "qwen_image_qwen_wan_",
                "qwen_image_qwen_to_wan_",
                "qwen_image_residual_to_qwen_",
                "qwen_image_qwen_on_wan_",
                "qwen_image_qwen_gradient_l2_",
                "qwen_image_wan_frame_gradient_l2_",
                "qwen_image_qwen_video_wan_",
                "qwen_image_qwen_to_video_wan_",
                "qwen_image_fake_critic_",
            ))
            and torch.is_tensor(value)
            and value.numel() == 1
        )
        if not keys:
            return
        groups = [
            ([key for key in keys if key.endswith("_min")], dist.ReduceOp.MIN),
            ([key for key in keys if key.endswith("_max")], dist.ReduceOp.MAX),
            ([
                key for key in keys
                if not key.endswith(("_min", "_max"))
            ], dist.ReduceOp.SUM),
        ]
        for group_keys, operation in groups:
            if not group_keys:
                continue
            packed = torch.stack([
                log_dict[key].detach().to(
                    device=self.device, dtype=torch.float32
                )
                for key in group_keys
            ])
            dist.all_reduce(packed, op=operation)
            if operation == dist.ReduceOp.SUM:
                packed.div_(self.world_size)
            for key, value in zip(group_keys, packed):
                log_dict[key] = value

    def _accumulate_fwdbwd(self, train_generator):
        log_dicts = []
        loss_scale = 1.0 / self.gradient_accumulation_steps

        for accumulation_idx in range(self.gradient_accumulation_steps):
            batch = next(self.dataloader)
            should_sync = accumulation_idx == self.gradient_accumulation_steps - 1
            with ExitStack() as stack:
                if not should_sync:
                    if train_generator:
                        stack.enter_context(self.model.generator.no_sync())
                    else:
                        stack.enter_context(self.model.fake_score.no_sync())
                        if getattr(
                            self.model, "qwen_image_fake_critic_enabled", False
                        ):
                            stack.enter_context(
                                self.model.qwen_image_fake_score.no_sync()
                            )
                log_dict = self.fwdbwd_one_step(
                    batch,
                    train_generator,
                    loss_scale=loss_scale,
                    clip_grad=False,
                )
            log_dicts.append(log_dict)

        merged_log_dict = self._mean_log_dict(log_dicts)
        if train_generator:
            merged_log_dict["generator_grad_norm"] = self.model.generator.clip_grad_norm_(
                self.max_grad_norm_generator)
        else:
            merged_log_dict["critic_grad_norm"] = self.model.fake_score.clip_grad_norm_(
                self.max_grad_norm_critic)
            if getattr(self.model, "qwen_image_fake_critic_enabled", False):
                merged_log_dict["qwen_image_fake_critic_grad_norm"] = (
                    self.model.qwen_image_fake_score.clip_grad_norm_(
                        self.max_grad_norm_critic
                    )
                )
        return merged_log_dict

    def _run_log_inference(self):
        if self.log_inference_hook is None:
            return
        torch.cuda.empty_cache()
        self.log_inference_hook(self.step)
        torch.cuda.empty_cache()

    def train(self):
        start_step = self.step

        while self.step < self.config.max_train_steps:
            TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0

            # Train the generator
            if TRAIN_GENERATOR:
                self.generator_optimizer.zero_grad(set_to_none=True)
                generator_log_dict = self._accumulate_fwdbwd(True)
                self._global_mean_probe_logs(generator_log_dict)
                self.generator_optimizer.step()
                if self.generator_ema is not None:
                    self.generator_ema.update(self.model.generator)
                # The critic update below does not consume generator gradients.
                # Release them immediately instead of retaining them across the
                # critic-only iterations until the next generator update.
                self.generator_optimizer.zero_grad(set_to_none=True)
            else:
                generator_log_dict = {}

            # Train the critic
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_log_dict = self._accumulate_fwdbwd(False)
            self.critic_optimizer.step()
            # Generator updates run before the next critic-side zero_grad().
            # Clear both video-critic and Qwen fake-critic gradients here so a
            # later Qwen teacher FSDP unshard has the full memory headroom.
            self.critic_optimizer.zero_grad(set_to_none=True)

            # Increment the step since we finished gradient update
            self.step += 1

            # Create EMA params (if not already created)
            if (self.step >= self.config.ema_start_step) and \
                    (self.generator_ema is None) and (self.config.ema_weight > 0):
                self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)

            # Save and optionally run log-step inference.
            is_log_step = (self.step - start_step) > 0 and self.step % self.config.log_iters == 0
            if is_log_step:
                if not self.config.no_save:
                    torch.cuda.empty_cache()
                    self.save()
                    torch.cuda.empty_cache()
                self._run_log_inference()

            # Logging
            if self.is_main_process:
                wandb_loss_dict = {}
                if TRAIN_GENERATOR:
                    wandb_loss_dict.update(
                        {
                            "generator_loss": generator_log_dict["generator_loss"].mean().item(),
                            "generator_grad_norm": generator_log_dict["generator_grad_norm"].mean().item(),
                            "dmdtrain_gradient_norm": generator_log_dict["dmdtrain_gradient_norm"].mean().item()
                        }
                    )
                    for key in [
                        "video_dmd_loss",
                        "legacy__qwen-dmd_loss",
                        "qwen_image_qwen_dmd_loss",
                        "qwen_image_md_dmd_loss",
                        "qwen_image_ca_dmd_loss",
                        "qwen_image_wan_frame_dmd_loss",
                        "legacy__qwen-dmd_gradient_norm",
                        "qwen_image_qwen_dmd_gradient_norm",
                        "qwen_image_md_gradient_norm",
                        "qwen_image_ca_gradient_norm",
                        "qwen_image_subtracted_wan_frame_dmd_gradient_norm",
                        "qwen_image_subtract_wan_frame_dmd_weight",
                        "qwen_image_qwen_wan_gradient_cosine",
                        "qwen_image_qwen_to_wan_gradient_norm_ratio",
                        "qwen_image_residual_to_qwen_gradient_norm_ratio",
                        "qwen_image_qwen_on_wan_projection_coefficient",
                        "qwen_image_qwen_wan_negative_cosine_fraction",
                        "qwen_image_qwen_video_wan_support_cosine",
                        "qwen_image_qwen_video_wan_global_cosine",
                        "qwen_image_qwen_video_wan_negative_fraction",
                        "qwen_image_qwen_to_video_wan_weighted_norm_ratio",
                        "qwen_image_probe_ca_timestep_mean",
                        "qwen_image_probe_ca_timestep_std",
                        "qwen_image_probe_ca_timestep_min",
                        "qwen_image_probe_ca_timestep_max",
                        "qwen_image_transition_candidate_raw_score_mean",
                        "qwen_image_transition_candidate_raw_score_std",
                        "qwen_image_transition_candidate_smoothed_score_mean",
                        "qwen_image_transition_candidate_smoothed_score_std",
                        "qwen_image_transition_peak_raw_score_mean",
                        "qwen_image_transition_peak_raw_score_std",
                        "qwen_image_transition_peak_smoothed_score_mean",
                        "qwen_image_transition_peak_smoothed_score_std",
                        "qwen_image_transition_peak_prominence_mean",
                        "qwen_image_transition_peak_prominence_std",
                        "qwen_image_transition_peak_positive_prominence_fraction",
                        "qwen_image_transition_selected_to_candidate_raw_score_ratio",
                        "qwen_image_transition_selected_to_candidate_smoothed_score_ratio",
                        "qwen_image_transition_peak_fallback_fraction",
                        "qwen_image_transition_peak_relaxed_nms_fraction",
                        "qwen_image_transition_peak_unsafe_edge_fraction",
                        "qwen_image_transition_peak_first_boundary_fraction",
                        "qwen_image_transition_peak_last_boundary_fraction",
                        "qwen_image_transition_local_max_count",
                        "qwen_image_transition_peak_frame_mean",
                        "qwen_image_transition_peak_frame_std",
                        "qwen_image_transition_peak_separation_mean",
                        "qwen_image_transition_peak_separation_std",
                        "qwen_image_transition_segment_length_mean",
                        "qwen_image_transition_segment_length_std",
                        "qwen_image_transition_segment_length_min",
                        "qwen_image_transition_guarded_segment_length_mean",
                        "qwen_image_transition_guarded_segment_length_std",
                        "qwen_image_transition_guarded_segment_length_min",
                        "qwen_image_transition_guard_fallback_fraction",
                        "qwen_image_transition_sample_frame_mean",
                        "qwen_image_transition_sample_frame_std",
                        "qwen_image_transition_sample_first_candidate_fraction",
                        "qwen_image_transition_sample_last_candidate_fraction",
                        "qwen_image_transition_sample_boundary_distance_mean",
                        "qwen_image_transition_sample_boundary_distance_std",
                        "qwen_image_loss_weight",
                    ]:
                        if key in generator_log_dict:
                            wandb_loss_dict[key] = generator_log_dict[key].mean().item()
                    for key, value in generator_log_dict.items():
                        if key.startswith("qwen_image_frame_idx_"):
                            wandb_loss_dict[key] = value.mean().item()

                wandb_loss_dict.update(
                    {
                        "critic_loss": critic_log_dict["critic_loss"].mean().item(),
                        "critic_grad_norm": critic_log_dict["critic_grad_norm"].mean().item()
                    }
                )
                for key in [
                    "video_critic_loss",
                    "qwen_image_fake_critic_loss",
                    "qwen_image_fake_critic_grad_norm",
                    "qwen_image_fake_critic_first_frame_loss",
                    "qwen_image_fake_critic_continuation_loss",
                ]:
                    if key in critic_log_dict:
                        wandb_loss_dict[key] = critic_log_dict[key].mean().item()
                for key, value in critic_log_dict.items():
                    if key.startswith("qwen_image_fake_critic_offset_"):
                        wandb_loss_dict[key] = value.mean().item()

                if not self.disable_wandb:
                    wandb.log(wandb_loss_dict, step=self.step)

            if self.step % self.config.gc_interval == 0:
                if dist.get_rank() == 0:
                    logging.info("DistGarbageCollector: Running GC.")
                gc.collect()
                torch.cuda.empty_cache()

            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time
