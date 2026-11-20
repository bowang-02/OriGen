"""Train OriGen from a fusion checkpoint with dynamic single-plane sampling."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from train_OriGen.data import OrientationSamplingDataset, MultiSequenceMRITrainingCollator
from train_OriGen.lora import ATTENTION_TARGETS, FFN_TARGETS, LoRALinear, lora_parameters, merge_text_lora
from train_OriGen.model_loader import load_model_and_processor
from train_utils import close_logger, create_accelerator, make_logger, save_full_model, train_epochs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--fusion-model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--effective-global-batch-size", type=int, default=16)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=float, default=64.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--axial-only-prob", type=float, default=0.1)
    parser.add_argument("--sagittal-only-prob", type=float, default=0.1)
    parser.add_argument("--lora-learning-rate", type=float, default=3e-5)
    parser.add_argument("--fusion-learning-rate", type=float, default=1e-5)
    parser.add_argument("--fusion-weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--mri-lower-percentile", type=float, default=0.5)
    parser.add_argument("--mri-upper-percentile", type=float, default=99.5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def set_training_mode(model) -> None:
    model.eval()
    model.model.visual.multi_sequence_fusion.train()
    for module in model.model.language_model.modules():
        if isinstance(module, LoRALinear):
            module.train()
        if module.__class__.__name__ == "Qwen3VLTextDecoderLayer" and getattr(module, "gradient_checkpointing", False):
            module.training = True


def optimizer_groups(model, args: argparse.Namespace) -> list[dict]:
    fusion = model.model.visual.multi_sequence_fusion
    decay, no_decay = [], []
    for name, parameter in fusion.named_parameters():
        if name.endswith("bias") or parameter.ndim == 1 or "queries" in name or "sequence_embedding" in name:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": list(lora_parameters(model)), "lr": args.lora_learning_rate, "weight_decay": 0.0},
        {"params": decay, "lr": args.fusion_learning_rate, "weight_decay": args.fusion_weight_decay},
        {"params": no_decay, "lr": args.fusion_learning_rate, "weight_decay": 0.0},
    ]


def main() -> None:
    args = parse_args()
    if not 0 <= args.axial_only_prob <= 1 or not 0 <= args.sagittal_only_prob <= 1 - args.axial_only_prob:
        raise ValueError("Single-plane probabilities must sum to at most one.")
    if (Path(args.output_dir) / "final").resolve() == Path(args.fusion_model_path).resolve():
        raise ValueError("Output checkpoint must differ from the fusion checkpoint.")

    accelerator, accumulation = create_accelerator(args)
    random.seed(args.seed + accelerator.process_index)
    output_dir = Path(args.output_dir)
    logger, handler = make_logger(accelerator, output_dir, args, "train_OriGen")

    # 1. Restore the fusion checkpoint and add trainable decoder LoRA.
    model, processor, preprocessing, lora_count = load_model_and_processor(
        args.fusion_model_path, args.lora_rank, args.lora_alpha, args.lora_dropout
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    set_training_mode(model)

    # 2. Sample complete MRI orientations as the training data is read.
    collator = MultiSequenceMRITrainingCollator(
        processor,
        volume_depth=preprocessing.merlin_volume_depth,
        image_size=preprocessing.merlin_image_size,
        sample_images=preprocessing.merlin_sample_images,
        query_tokens=256,
        lower_percentile=args.mri_lower_percentile,
        upper_percentile=args.mri_upper_percentile,
    )
    dataset = OrientationSamplingDataset(
        args.train_data, args.axial_only_prob, args.sagittal_only_prob
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collator, pin_memory=True,
    )
    optimizer = AdamW(optimizer_groups(model, args), foreach=False)
    accelerator.print(
        f"records={len(dataset)} lora_modules={lora_count} accumulation={accumulation}"
    )
    # 3. Train fusion and LoRA with separate learning rates.
    result = train_epochs(
        accelerator=accelerator, model=model, optimizer=optimizer, loader=loader,
        args=args, accumulation=accumulation, set_mode=set_training_mode, logger=logger,
    )

    lora_config = {
        "rank": args.lora_rank, "alpha": args.lora_alpha,
        "dropout": args.lora_dropout,
        "targets": [*ATTENTION_TARGETS, *FFN_TARGETS], "merged": True,
    }
    metadata = {
        "source_fusion": str(Path(args.fusion_model_path).resolve()),
        "plane_sampling": {
            "axial_only": args.axial_only_prob,
            "sagittal_only": args.sagittal_only_prob,
            "both_planes": 1 - args.axial_only_prob - args.sagittal_only_prob,
        },
        "lora": lora_config,
        **{key: value for key, value in result.items() if key != "model"},
    }
    accelerator.wait_for_everyone()
    # 4. Merge LoRA and save one full checkpoint for inference.
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(result["model"])
        unwrapped.eval()
        merge_text_lora(unwrapped)
        unwrapped.config.origen_training_config = lora_config
    save_full_model(accelerator, result["model"], processor, output_dir / "final", metadata)
    accelerator.print(f"Saved OriGen checkpoint to {output_dir / 'final'}")
    close_logger(logger, handler, accelerator)


if __name__ == "__main__":
    main()
