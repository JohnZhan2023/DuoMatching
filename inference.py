import argparse
import torch
import os
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision import transforms
from torchvision.io import write_video
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
import json

from utils.checkpoint import load_generator_checkpoint

from pipeline import CausalInferencePipeline
from utils.dataset import TextDataset, TextImagePairDataset
from utils.misc import set_seed

from demo_utils.memory import get_cuda_free_memory_gb, DynamicSwapInstaller

parser = argparse.ArgumentParser(description="DuoMatching video generation")
parser.add_argument("--config_path", type=str, required=True, help="Path to the config file")
parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to model.pt (use --use_ema for released weights)")
parser.add_argument("--data_path", type=str, help="Path to the dataset")
parser.add_argument(
    "--vbench_info_path",
    type=str,
    default=None,
    help="Path to VBench_full_info.json. Prompts are deduplicated in file order.",
)
parser.add_argument("--output_folder", type=str, required=True, help="Output folder")
parser.add_argument("--num_output_frames", type=int, default=21, help="Number of latent frames; 21 produces 81 video frames")
parser.add_argument("--height", type=int, default=None, help="Output video height in pixels. Defaults to config.height.")
parser.add_argument("--width", type=int, default=None, help="Output video width in pixels. Defaults to config.width.")
parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA parameters")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--start_index", type=int, default=0, help="First prompt index (inclusive)")
parser.add_argument("--end_index", type=int, default=None, help="Last prompt index (exclusive)")
parser.add_argument(
    "--shared_seed",
    action="store_true",
    help="Reset to --seed for every prompt (useful for comparable VBench generation).",
)
parser.add_argument(
    "--index_output_filename",
    action="store_true",
    help="Prefix non-VBench output filenames with the prompt index to avoid truncated-name collisions.",
)
parser.add_argument(
    "--output_name_prefix",
    type=str,
    default="",
    help="Optional prefix for indexed non-VBench output filenames.",
)
parser.add_argument("--i2v", action="store_true", help="Whether to perform I2V (or T2V by default)")
parser.add_argument("--report_timing", action="store_true",
                    help="Report generation timing after warmup.")
args = parser.parse_args()
if args.num_output_frames < 1:
    parser.error("--num_output_frames must be positive")
if not args.data_path and not args.vbench_info_path:
    parser.error("one of --data_path or --vbench_info_path is required")
if not torch.cuda.is_available():
    parser.error("DuoMatching video generation requires a CUDA GPU")

# Initialize distributed inference
if "LOCAL_RANK" in os.environ:
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()

else:
    device = torch.device("cuda")
    local_rank = 0
    world_size = 1
    global_rank = 0

set_seed(args.seed)

print(f'Rank {global_rank} using {device}; free VRAM {get_cuda_free_memory_gb(device)} GB')
low_memory = get_cuda_free_memory_gb(device) < 40

torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)
default_config = OmegaConf.load("configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)

height = int(args.height if args.height is not None else config.height)
width = int(args.width if args.width is not None else config.width)
if height % 16 != 0 or width % 16 != 0:
    raise ValueError(
        f"height and width must be divisible by 16 for Wan latent patches, got {height}x{width}"
    )
latent_height = height // 8
latent_width = width // 8
print(f"Inference resolution: {height}x{width} pixels; latent size: {latent_height}x{latent_width}")

# Initialize pipeline
pipeline = CausalInferencePipeline(config, device=device)

if args.checkpoint_path:
    key = 'generator_ema' if args.use_ema else 'generator'
    load_generator_checkpoint(pipeline.generator, args.checkpoint_path, key=key)

pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
else:
    pipeline.text_encoder.to(device=device)
pipeline.generator.to(device=device)
pipeline.vae.to(device=device)


# Create dataset
if args.i2v:
    assert not dist.is_initialized(), "I2V does not support distributed inference yet"
    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = TextImagePairDataset(args.data_path, transform=transform)
elif args.vbench_info_path:
    with open(args.vbench_info_path, encoding="utf-8") as f:
        vbench_info = json.load(f)
    vbench_prompts = list(dict.fromkeys(item["prompt_en"] for item in vbench_info))

    class VBenchTextDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(vbench_prompts)

        def __getitem__(self, idx):
            return {"prompts": vbench_prompts[idx], "idx": idx}

    dataset = VBenchTextDataset()
else:
    if not args.data_path:
        parser.error("one of --data_path or --vbench_info_path is required")
    dataset = TextDataset(prompt_path=args.data_path)
dataset = torch.utils.data.Subset(dataset, range(args.start_index, args.end_index or len(dataset)))
num_prompts = len(dataset)
print(f"Number of prompts: {num_prompts}")

if args.report_timing and num_prompts < 2:
    print(f"[WARN] --report_timing requires at least 2 prompts "
          f"(got {num_prompts}); timing disabled.")
    args.report_timing = False

if dist.is_initialized():
    # Shard without padding or dropping samples. DistributedSampler either
    # duplicates indices or drops the tail when the dataset size is not evenly
    # divisible by world_size, which is undesirable for inference datasets.
    sampler = range(global_rank, num_prompts, world_size)
    print(f"Rank {global_rank}: {len(sampler)} prompts")
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

# Create output directory (only on main process to avoid race conditions)
if local_rank == 0:
    os.makedirs(args.output_folder, exist_ok=True)

if dist.is_initialized():
    dist.barrier()

def encode(self, videos: torch.Tensor) -> torch.Tensor:
    device, dtype = videos[0].device, videos[0].dtype
    scale = [self.mean.to(device=device, dtype=dtype),
             1.0 / self.std.to(device=device, dtype=dtype)]
    output = [
        self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
        for u in videos
    ]

    output = torch.stack(output, dim=0)
    return output


def make_output_name(prompt, idx):
    if args.vbench_info_path:
        return f'{prompt}-0.mp4'
    if args.index_output_filename:
        return f'{args.output_name_prefix}{idx:04d}_{prompt[:70]}.mp4'
    return f'{prompt[:80]}.mp4'


for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    idx = batch_data['idx'].item()

    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]  # First (and only) item in the batch

    all_video = []
    num_generated_frames = 0  # Number of generated (latent) frames


    if args.i2v:
        assert config.num_frame_per_block == 1, "Current I2V only supports the frame-wise model."
        # For image-to-video, batch contains image and caption
        prompt = batch['prompts'][0]  # Get caption from batch
        output_path = os.path.join(args.output_folder, make_output_name(prompt, idx))
        if os.path.exists(output_path):
            print('Video has been generated. Pass!')
            continue
        # Process the image
        image = batch['image'].squeeze(0).unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)

        # Encode the input image as the first latent
        initial_latent = pipeline.vae.encode_to_latent(image).to(device=device, dtype=torch.bfloat16)
        prompts = [prompt]
        sampled_noise = torch.randn(
            [1, args.num_output_frames - 1, 16, latent_height, latent_width],
            device=device,
            dtype=torch.bfloat16,
        )
    else:
        # For text-to-video, batch is just the text prompt
        prompt = batch['prompts'][0]
        output_name = make_output_name(prompt, idx)
        output_path = os.path.join(args.output_folder, output_name)
        if os.path.exists(output_path):
            print('Video has been generated. Pass!')
            continue
        extended_prompt = batch['extended_prompts'][0] if 'extended_prompts' in batch else None
        if extended_prompt is not None:
            prompts = [extended_prompt]
        else:
            prompts = [prompt]

        initial_latent = None
        if args.shared_seed:
            set_seed(args.seed)
        sampled_noise = torch.randn(
            [1, args.num_output_frames, 16, latent_height, latent_width],
            device=device,
            dtype=torch.bfloat16,
        )

    sample_report_timing = args.report_timing and i >= 1
    inference_kwargs = dict(
        noise=sampled_noise,
        text_prompts=prompts,
        return_latents=True,
        initial_latent=initial_latent,
    )
    inference_kwargs["report_timing"] = sample_report_timing
    video, latents = pipeline.inference(**inference_kwargs)
    if sample_report_timing:
        latency = pipeline.first_chunk_time
        elapsed = pipeline.last_generation_time
        num_pixel_frames = video.shape[1]
        fps = num_pixel_frames / elapsed if elapsed > 0 else float('inf')
        print(f"[Sample {i}] {num_pixel_frames} frames, "
              f"latency ↓ {latency:.2f}s, FPS ↑ {fps:.2f}")
        # Only tested on A800, for the Causal Forcing++ paper latency & throughput.
        # Not make claims for other hardware like H100.
        # For the result on H100, refer to the reported results in the Self Forcing paper.
        # We do not guarantee that our FPS/latency measurement protocol is identical to that used in the Self Forcing paper.
    current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
    all_video.append(current_video)
    num_generated_frames += latents.shape[1]

    # Final output video
    clean_latent = latents[0].cpu()
    video = 255.0 * torch.cat(all_video, dim=1)

    # Clear VAE cache
    pipeline.vae.model.clear_cache()

    output_name = make_output_name(prompt, idx)
    output_path = os.path.join(args.output_folder, output_name)
    write_video(output_path, video[0], fps=16)
