# GenText-Forensics: Document Tampering Detection

ACM MM 2026 GenText-Forensics Challenge — **10th place, s_fin = 0.4877**

## Task

Given a document image, detect whether it has been tampered with (classification), localize the tampered regions (segmentation), and generate a detailed forensic analysis report explaining the anomalies found.

Evaluation metrics: SDet (classification), SLoc (localization IoU), SRep (report quality), SExp (explanation quality), combined into s_fin.

## Architecture

```
Document Image
      │
      ├──► SegFormer-b5 (semantic segmentation)
      │         │
      │         ├── No mask → AUTHENTIC
      │         │
      │         └── Has mask → Highlight overlay on image
      │                              │
      └──────────────────────────────┤
                                     ▼
                          Qwen2.5-VL-7B (MLLM)
                          LoRA fine-tuned
                                     │
                                     ▼
                          Forensic Report Generation
                                     │
                              Post-processing:
                          ├── SegFormer bbox replacement
                          ├── Force FORGED override
                          └── Format cleanup
                                     │
                                     ▼
                            Final Prediction
```

**SegFormer-b5**: Binary segmentation (authentic vs. tampered pixels). Acts as a gate — if no tampered region detected, image is classified as AUTHENTIC without MLLM inference.

**Qwen2.5-VL-7B + LoRA**: Generates structured forensic reports with conclusion, risk score, anomaly descriptions, grounding coordinates, and reasoning.

**Key pipeline innovations**:
- **BBox replacement**: MLLM hallucinate coordinates (89% cluster at x=2530). We replace all MLLM grounding coordinates with SegFormer connected-component bounding boxes. IoU: 0.067 → 0.755.
- **Force FORGED override**: When SegFormer detects a mask but MLLM says AUTHENTIC, force the classification to FORGED with re-inference. SDet: 80% → 93%.

## Training

### SegFormer-b5
- Dataset: RealText-V2 train split (binary masks)
- Standard semantic segmentation fine-tuning

### Qwen2.5-VL-7B LoRA (best config: v2c)
- Framework: ms-swift 3.5.0
- Base model: `Qwen/Qwen2.5-VL-7B-Instruct`
- LoRA rank: 64, alpha: 128
- Target modules: all-linear
- freeze_vit: true, freeze_aligner: true
- max_length: 4096 (critical — 2048 truncated long reports, causing major quality loss)
- max_pixels: 350000
- Batch size: 1, gradient accumulation: 8
- Epochs: 4 (best by eval_loss)
- Learning rate: 5e-5, cosine schedule, warmup 0.03
- Training data: 11,538 samples (FORGED with SegFormer-highlighted images + AUTHENTIC)

```bash
swift sft \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --dataset train_v2_with_authentic.jsonl \
    --lora_rank 64 --lora_alpha 128 \
    --target_modules all-linear \
    --freeze_vit true --freeze_aligner true \
    --max_length 4096 --max_pixels 350000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 4 \
    --learning_rate 5e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --gradient_checkpointing true \
    --bf16 true
```

## Files

| File | Description |
|------|-------------|
| `infer_final_v6.py` | Main inference pipeline (SegFormer + MLLM + post-processing) |
| `postprocess_enhance.py` | Crop refinement post-processing (not used in final submission — hurt official score) |
| `eval_and_optimize.py` | Holdout evaluation (SegFormer threshold search + end-to-end metrics) |
| `llm_judge.py` | LLM-based report quality evaluation using DeepSeek API |
| `run_resume.py` | Batch inference wrapper with checkpoint resume (saves every 200 images) |
| `prediction.zip` | Final submission file (6478 test images) |

## Model Weights

LoRA checkpoint (v2e, epoch 4): to be merged with `Qwen/Qwen2.5-VL-7B-Instruct`

```bash
# Merge LoRA into base model
swift export \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --adapters path/to/checkpoint-5712 \
    --merge_lora true \
    --output_dir output/merged_model
```

SegFormer-b5 weights: fine-tuned from `nvidia/segformer-b5-finetuned-ade-640-640`

## Key Experiments & Findings

### What worked
| Change | Impact |
|--------|--------|
| SegFormer bbox replacement | IoU 0.067 → 0.755 |
| Force FORGED override | SDet 80% → 93% |
| max_length 2048 → 4096 | OVERALL 2.21 → 2.85 (reports were being truncated) |
| LoRA rank 16 → 64 | OVERALL 2.85 → 2.92 |

### What didn't work
| Attempt | Result | Why |
|---------|--------|-----|
| RSFT (6 experiments) | All failed or tied | Reward function negatively correlated with quality (Pearson = -0.357) |
| Crop refinement | Official score dropped 0.4877 → 0.4279 | Added details penalized as hallucination by official eval |
| Unfreeze ViT | OVERALL 2.92 → 2.48 | 11K samples insufficient to fine-tune vision encoder |
| SegFormer fallback (MLLM re-judge) | Rescued 0 images | Model learned "no highlight = AUTHENTIC" from training data |

### Lessons learned
1. **Submit early** — don't optimize blindly without knowing your real score
2. **max_length truncation is a silent killer** — always check if training samples are being truncated
3. **Local eval ≠ official eval** — crop refinement improved local LLM Judge but hurt official s_fin
4. **Engineering robustness matters** — PIL crash lost 15 hours of inference; always implement checkpoint resume
5. **Diagnose before experimenting** — one good analysis (Pearson = -0.357) saved more time than five blind runs

## Competition Context

- Competition: ACM MM 2026 GenText-Forensics Challenge
- Platform: CodaBench
- Dataset: [RealText-V2](https://huggingface.co/datasets/vankey/RealText-V2)
- Result: 13th place, s_fin = 0.4877 (1st place: 0.85)
- Team: first-time competition participant
