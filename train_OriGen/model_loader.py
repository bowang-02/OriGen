"""Load the fusion checkpoint and prepare the trainable OriGen modules."""

from __future__ import annotations

import torch

from custom_model.image_processing_multi_enc_qwen2_vl import ClinFusionProcessor
from custom_model.modeling_multi_enc_qwen3_vl import MultiEnc_Qwen3VLForConditionalGeneration
from train_OriGen.lora import install_text_lora


def load_model_and_processor(checkpoint: str, rank: int, alpha: float, dropout: float):
    model = MultiEnc_Qwen3VLForConditionalGeneration.from_pretrained(
        checkpoint,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
        local_files_only=True,
    )
    # PE-3D's RoPE cache is not stored in the checkpoint.
    model.model.visual.pe_3d.visual.rope.init_tensors()
    processor = ClinFusionProcessor.from_pretrained(checkpoint, local_files_only=True)

    model.config.use_cache = False
    model.model.config.use_cache = False
    model.model.language_model.config.use_cache = False
    model.requires_grad_(False)
    model.model.visual.multi_sequence_fusion.float().requires_grad_(True)
    lora_count = install_text_lora(model, rank=rank, alpha=alpha, dropout=dropout)
    return model, processor, processor.image_processor, lora_count
