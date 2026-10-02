import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, prompt_path, extended_prompt_path=None):
        with open(prompt_path, encoding="utf-8") as f:
            self.prompt_list = [line.rstrip() for line in f]

        if extended_prompt_path is not None:
            with open(extended_prompt_path, encoding="utf-8") as f:
                self.extended_prompt_list = [line.rstrip() for line in f]
            assert len(self.extended_prompt_list) == len(self.prompt_list)
        else:
            self.extended_prompt_list = None

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, idx):
        batch = {"prompts": self.prompt_list[idx], "idx": idx}
        if self.extended_prompt_list is not None:
            batch["extended_prompts"] = self.extended_prompt_list[idx]
        return batch


class TextImagePairDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.transform = transform
        data_dir = Path(data_dir)
        metadata_files = list(data_dir.glob("target_crop_info_*.json"))
        if len(metadata_files) != 1:
            raise ValueError(
                f"Expected one target_crop_info_*.json in {data_dir}, "
                f"found {len(metadata_files)}"
            )

        metadata_path = metadata_files[0]
        self.image_dir = data_dir / metadata_path.stem.split("_")[-1]
        with open(metadata_path, encoding="utf-8") as f:
            self.metadata = json.load(f)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        image = Image.open(self.image_dir / item["file_name"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return {"image": image, "prompts": item["caption"], "idx": idx}


def cycle(dataloader):
    while True:
        yield from dataloader
