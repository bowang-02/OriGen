# OriGen: Orientation-Aware Fusion for Cervical Cancer MRI Report Generation

Run the commands below from the repository root.

## 1. Environment

```bash
conda create -n origen python=3.11 -y
conda activate origen
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install ninja packaging psutil
python -m pip install flash-attn==2.8.3 --no-build-isolation
```

## 2. Run the released model

Download the [OriGen weights](https://huggingface.co/BoWang-02/OriGen) to `models/OriGen/`:

```bash
hf download BoWang-02/OriGen --local-dir models/OriGen
```

Run inference on one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python infer.py
```

The result is saved to `result/output.jsonl`. It keeps the input fields and adds `model_generation`. For another dataset:

```bash
CUDA_VISIBLE_DEVICES=0 python infer.py --input dataset/my_cases.jsonl --output result/my_cases.jsonl
```

Each JSONL line needs `messages.prompt` and `messages.nifti`; `id` is optional:

```json
{"messages":{"prompt":"We will provide you multi-sequence cervical MRI volumes from a patient. We hope you can make a radiology report based on these volumes.","nifti":["../Data/case01/case01-T2A.nii.gz","../Data/case01/case01-T2S.nii.gz"]}}
```

NIfTI paths can be absolute or relative to the JSONL file's directory. Filenames must end in `-T1CA`, `-T2A`, `-ADC`, `-T1CS`, or `-T2S`, followed by `.nii` or `.nii.gz`. Use at most one file of each type.

## 3. Prepare training data

Create `dataset/alignment.jsonl` with one sequence per record and `dataset/train.jsonl` with one to five sequences per record. Add a `GT` to each record.

`dataset/alignment.jsonl`:

```json
{"messages":{"prompt":"Describe the imaging findings in this single cervical MRI sequence.","nifti":["../Data/case01/case01-T2A.nii.gz"]},"GT":"Single-sequence description."}
```

`dataset/train.jsonl`:

```json
{"messages":{"prompt":"We will provide you multi-sequence cervical MRI volumes from a patient. We hope you can make a radiology report based on these volumes.","nifti":["../Data/case01/case01-T2A.nii.gz","../Data/case01/case01-T2S.nii.gz"]},"GT":"Full report."}
```

## 4. Download the starting checkpoints

```bash
hf download Alibaba-DAMO-Academy/ClinFusion-8B --local-dir models/ClinFusion-8B
hf download Qwen/Qwen3-VL-8B-Instruct --local-dir models/Qwen3-VL-8B-Instruct
hf download facebook/dinov2-large --local-dir models/dinov2-large
hf download laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup --local-dir models/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup
```

## 5. Train the three stages

Training uses two GPUs. Run A1 and A2 alignment, fusion training, then OriGen training:

```bash
bash train_alignment/run.sh
bash train_fusion_only/run.sh
bash train_OriGen/run.sh
```

| Stage | Epochs | Effective global batch | Learning rate |
| --- | ---: | ---: | --- |
| A1 alignment | 1 | 16 | `1e-5` |
| A2 alignment | 3 | 16 | `5e-6`; PE-3D `1e-6` |
| Fusion only | 5 | 4 | Fusion `1e-4` |
| OriGen | 6 | 16 | LoRA `3e-5`, fusion `1e-5` |

The checkpoints are saved at `output/alignment_a2/final`, `output/train_fusion_only/final`, and `output/OriGen/final`. The scripts use GPUs 0 and 1 by default; set `GPU_IDS` to choose another pair.

Run your trained checkpoint with:

```bash
CUDA_VISIBLE_DEVICES=0 python infer.py --model output/OriGen/final --output result/trained.jsonl
```
