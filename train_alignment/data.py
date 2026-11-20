"""Single-sequence MRI JSONL dataset and OriGen alignment collator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from torch.utils.data import Dataset

from custom_model.medical_image_preprocessing import (
    preprocess_mri_image,
    sample_slices_from_volume_to_pil_fast,
)


class SingleSequenceMRIDataset(Dataset):
    """Read one MRI NIfTI and one target description per JSONL record."""

    def __init__(self, jsonl_path: str):
        self.path = Path(jsonl_path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Training JSONL not found: {self.path}")
        self.records = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {self.path}:{line_number}") from exc
                self.records.append(self._normalize(record, line_number))
        if not self.records:
            raise ValueError(f"No records found in {self.path}")

    def _normalize(self, record: Dict[str, object], line_number: int) -> Dict[str, str]:
        messages = record.get("messages", {})
        if not isinstance(messages, dict):
            raise ValueError(f"{self.path}:{line_number} has a non-object 'messages' field.")
        paths = messages.get("nifti")
        if not isinstance(paths, list) or len(paths) != 1 or not isinstance(paths[0], str) or not paths[0]:
            raise ValueError(f"{self.path}:{line_number} requires exactly one NIfTI path in messages.nifti.")
        nifti = paths[0]
        description = record.get("GT")
        prompt = messages.get("prompt")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"{self.path}:{line_number} requires a non-empty GT.")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"{self.path}:{line_number} requires a non-empty messages.prompt.")

        image_path = Path(nifti).expanduser()
        if not image_path.is_absolute():
            image_path = self.path.parent / image_path
        if not image_path.is_file():
            raise FileNotFoundError(f"MRI file at {self.path}:{line_number} not found: {image_path}")
        return {
            "nifti": str(image_path.resolve()),
            "description": description.strip(),
            "prompt": prompt,
        }

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, str]:
        return self.records[index]

class MRITrainingCollator:
    def __init__(
        self,
        processor,
        volume_depth: int,
        image_size: int,
        sample_images: int,
        lower_percentile: float = 0.5,
        upper_percentile: float = 99.5,
    ):
        self.processor = processor
        self.volume_depth = int(volume_depth)
        self.image_size = int(image_size)
        self.sample_images = int(sample_images)
        self.percentiles = (float(lower_percentile), float(upper_percentile))
        self.processor.tokenizer.padding_side = "right"
        self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        self.assistant_marker = self.processor.tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )
        if not self.assistant_marker:
            raise ValueError("Could not tokenize the assistant response marker.")

    def _load_volume(self, nifti_path: str) -> torch.Tensor:
        return preprocess_mri_image(
            nifti_path,
            target_size=(self.volume_depth, self.image_size, self.image_size),
            lower_percentile=self.percentiles[0],
            upper_percentile=self.percentiles[1],
        ).float()

    @staticmethod
    def _last_subsequence(sequence: Sequence[int], pattern: Sequence[int]) -> int:
        for start in range(len(sequence) - len(pattern), -1, -1):
            if list(sequence[start : start + len(pattern)]) == list(pattern):
                return start
        return -1

    def __call__(self, records: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        texts = []
        image_batches = []
        volumes = []
        for record in records:
            volume = self._load_volume(record["nifti"])
            images = sample_slices_from_volume_to_pil_fast(
                volume,
                num_slices=self.sample_images,
                random=False,
            )
            user_content = [{"type": "image"} for _ in images]
            user_content.append({"type": "text", "text": record["prompt"]})
            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": [{"type": "text", "text": record["description"]}]},
            ]
            texts.append(
                self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
            image_batches.append(images)
            volumes.append(volume)

        batch = self.processor(
            text=texts,
            images=image_batches,
            padding=True,
            return_tensors="pt",
            images_kwargs={"volume": volumes},
        )
        labels = batch["input_ids"].clone()
        for row_index in range(labels.shape[0]):
            valid_length = int(batch["attention_mask"][row_index].sum().item())
            token_ids = labels[row_index, :valid_length].tolist()
            marker_start = self._last_subsequence(token_ids, self.assistant_marker)
            if marker_start < 0:
                raise ValueError("Assistant marker not found after chat-template tokenization.")
            response_start = marker_start + len(self.assistant_marker)
            labels[row_index, :response_start] = -100
            labels[row_index, valid_length:] = -100
        batch["labels"] = labels
        return dict(batch)
