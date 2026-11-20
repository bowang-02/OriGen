"""Generate cervical MRI reports with the released OriGen checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from custom_model.image_processing_multi_enc_qwen2_vl import ClinFusionProcessor
from custom_model.modeling_multi_enc_qwen3_vl import MultiEnc_Qwen3VLForConditionalGeneration
from train_fusion_only.data import MultiSequenceMRIDataset, MultiSequenceMRITrainingCollator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/OriGen"))
    parser.add_argument("--input", type=Path, default=Path("dataset/infer.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("result/output.jsonl"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    args = parser.parse_args()

    with args.input.open(encoding="utf-8") as handle:
        original_records = [json.loads(line) for line in handle if line.strip()]
    dataset = MultiSequenceMRIDataset(str(args.input), require_report=False)

    torch.cuda.set_device(0)
    model = MultiEnc_Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
        local_files_only=True,
    )
    model.model.visual.pe_3d.visual.rope.init_tensors()
    processor = ClinFusionProcessor.from_pretrained(args.model, local_files_only=True)
    model.to("cuda:0").eval()
    model.config.use_cache = True
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    image_processor = processor.image_processor
    collator = MultiSequenceMRITrainingCollator(
        processor,
        volume_depth=image_processor.merlin_volume_depth,
        image_size=image_processor.merlin_image_size,
        sample_images=image_processor.merlin_sample_images,
        query_tokens=256,
        lower_percentile=0.5,
        upper_percentile=99.5,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for start in range(0, len(dataset), args.batch_size):
            indices = range(start, min(start + args.batch_size, len(dataset)))
            inputs = {
                key: value.to("cuda:0", non_blocking=True)
                for key, value in collator.build_generation_batch([dataset[i] for i in indices]).items()
            }
            prompt_width = inputs["input_ids"].shape[1]
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                generated = model.generate(
                    **inputs,
                    do_sample=False,
                    num_beams=1,
                    num_return_sequences=1,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=processor.tokenizer.eos_token_id,
                    eos_token_id=model.generation_config.eos_token_id,
                )
            reports = processor.batch_decode(
                generated[:, prompt_width:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for index, report in zip(indices, reports):
                record = dict(original_records[index])
                record["model_generation"] = report
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            print(f"Generated {min(start + args.batch_size, len(dataset))}/{len(dataset)} reports", flush=True)
    print(f"Saved reports to {args.output}")


if __name__ == "__main__":
    main()
