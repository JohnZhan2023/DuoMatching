import argparse
import os
from pathlib import Path
from omegaconf import OmegaConf
import wandb

from trainer import ScoreDistillationTrainer


OUTPUT_ROOT = Path(os.environ.get("DUOMATCHING_OUTPUT_ROOT", "outputs"))
LOGS_ROOT = OUTPUT_ROOT / "logs"


def resolve_logdir(value, config_name):
    if not value:
        return str(LOGS_ROOT / config_name)
    path = Path(value)
    if path.is_absolute():
        return str(path)
    if path.parts and path.parts[0] == "logs":
        path = Path(*path.parts[1:])
    return str(LOGS_ROOT / path)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--no_visualize", action="store_true")
    parser.add_argument("--logdir", type=str, default="", help="Path to the directory to save logs")
    parser.add_argument("--wandb-save-dir", type=str, default="", help="Path to the directory to save wandb logs")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--tf", action="store_true")

    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)
    config.no_save = args.no_save
    config.no_visualize = args.no_visualize
    config.tf = args.tf
    # get the filename of config_path
    config_name = os.path.basename(args.config_path).split(".")[0]
    config.config_name = config_name
    config.logdir = resolve_logdir(args.logdir, config_name)
    config.wandb_save_dir = args.wandb_save_dir or str(OUTPUT_ROOT)
    config.disable_wandb = args.disable_wandb

    if config.trainer == "score_distillation":
        trainer = ScoreDistillationTrainer(config)
    else:
        raise ValueError(f"Unknown trainer: {config.trainer}")
    trainer.train()

    wandb.finish()


if __name__ == "__main__":
    main()
