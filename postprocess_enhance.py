#!/usr/bin/env python3
"""
后处理增强: 去样板 + 区域裁剪精炼 REASON
==========================================

两个独立步骤, 可以单独或组合使用:

  # Step 1: 只去样板 (秒级, 不需要模型)
  python postprocess_enhance.py strip \
      --pred prediction_v4.jsonl \
      --output prediction_v4_stripped.jsonl

  # Step 2: 去样板 + 区域裁剪精炼 (需要 MLLM, ~8s/区域)
  python postprocess_enhance.py refine \
      --pred prediction_v4.jsonl \
      --image_dir /root/autodl-tmp/gentext/holdout_images \
      --mllm_path /root/autodl-tmp/gentext/output/sft_v3_rsft_merged \
      --output prediction_v5.jsonl
"""
import argparse, json, re, os
from pathlib import Path
from tqdm import tqdm

# ============================================================
# Step 1: 去样板文字
# ============================================================

# MLLM 在 SFT 中学到的样板, GT 里完全不存在
_BOILERPLATE_PATTERNS = [
    # "Report ID: FAR-xxxx-xx-xx" 及其变体
    r'Report ID:\s*FAR[-\w]*\s*',
    # "Date of Examination: xxxx-xx-xx"
    r'Date of Examination:\s*[\d\-x/]+\s*',
    # "Case Type: ..."
    r'Case Type:\s*[^\n]*(?:Authentication|Fraud|Analysis|Forensic)[^\n]*\s*',
    # "Examiner: ..." 
    r'Examiner:\s*[^\n]*\s*',
    # "Document Type: ..."
    r'Document Type:\s*[^\n]*\s*',
    # "Image Resolution: ..."
    r'Image Resolution:\s*[^\n]*\s*',
    # "File Format: ..."  
    r'File Format:\s*[^\n]*\s*',
]


def strip_boilerplate(report: str) -> str:
    """删除 MLLM 学到的样板文字, 不影响官方格式标签"""
    for pattern in _BOILERPLATE_PATTERNS:
        report = re.sub(pattern, '', report, flags=re.IGNORECASE)
    
    # 清理多余空行
    report = re.sub(r'\n{3,}', '\n\n', report)
    return report.strip()


def run_strip(pred_path, output_path):
    """对已有预测文件去样板"""
    records = []
    with open(pred_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    n_changed = 0
    total_words_saved = 0

    for rec in records:
        report = rec.get("report", "")
        stripped = strip_boilerplate(report)
        if stripped != report:
            n_changed += 1
            total_words_saved += len(report.split()) - len(stripped.split())
        rec["report"] = stripped

    with open(output_path, 'w') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    print(f"✅ 去样板完成: {output_path}")
    print(f"   修改 {n_changed}/{len(records)} 条")
    print(f"   共删除 ~{total_words_saved} 词样板文字")


# ============================================================
# Step 2: 区域裁剪精炼 REASON
# ============================================================

CROP_PROMPT = (
    "This is a close-up of a suspicious region from a document image. "
    "Describe in detail any visual anomalies you observe: "
    "font inconsistencies (size, weight, style), edge artifacts, "
    "background texture differences, noise patterns, color mismatches, "
    "alignment issues, or any other signs of digital manipulation. "
    "Be specific about what you see. If you also notice semantic issues "
    "(spelling errors, logical contradictions), mention those too. "
    "Keep your response concise (2-3 sentences)."
)


def crop_region(image_pil, box, margin_ratio=0.3):
    """
    从原图裁剪区域, 加边距保留上下文
    margin_ratio=0.3 表示每边扩展 30% 的区域尺寸
    """
    w, h = image_pil.size
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    mx = int(bw * margin_ratio)
    my = int(bh * margin_ratio)

    cx1 = max(0, x1 - mx)
    cy1 = max(0, y1 - my)
    cx2 = min(w, x2 + mx)
    cy2 = min(h, y2 + my)

    # 最小尺寸保证
    if cx2 - cx1 < 50:
        cx1 = max(0, (x1 + x2) // 2 - 50)
        cx2 = min(w, cx1 + 100)
    if cy2 - cy1 < 50:
        cy1 = max(0, (y1 + y2) // 2 - 50)
        cy2 = min(h, cy1 + 100)

    return image_pil.crop((cx1, cy1, cx2, cy2))


def run_crop_mllm(model, proc, crop_pil):
    """对裁剪区域做聚焦分析"""
    content = [
        {"type": "image", "image": crop_pil},
        {"type": "text", "text": CROP_PROMPT},
    ]
    messages = [{"role": "user", "content": content}]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[crop_pil],
                  padding=True, return_tensors="pt").to(model.device)

    import torch
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=300, do_sample=False)
    return proc.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


def enhance_reason_with_crop(report, crop_details):
    """
    用裁剪分析结果增强原有 REASON (追加, 不替换)
    """
    # 找到所有 REASON 段落
    anomaly_pattern = re.compile(
        r'(\[REASON\]\s*[:：]\s*)(.+?)(?=\[GROUNDING\]|###|---|\*\*END|## SUMMARY|\Z)',
        re.DOTALL
    )

    matches = list(anomaly_pattern.finditer(report))

    # 从后往前替换, 避免偏移
    for i in range(min(len(matches), len(crop_details)) - 1, -1, -1):
        m = matches[i]
        original_reason = m.group(2).strip()
        crop_detail = crop_details[i].strip()

        if not crop_detail or len(crop_detail) < 20:
            continue

        # 避免重复: 如果裁剪分析和原 REASON 高度重叠, 跳过
        original_words = set(original_reason.lower().split())
        crop_words = set(crop_detail.lower().split())
        if len(original_words) > 0:
            overlap = len(original_words & crop_words) / len(original_words)
            if overlap > 0.6:
                continue

        # 追加裁剪细节
        enhanced = (
            f"{original_reason}\n"
            f"Close-up analysis: {crop_detail}"
        )
        report = report[:m.start(2)] + enhanced + report[m.end(2):]

    return report


def run_refine(pred_path, image_dir, mllm_path, output_path):
    """去样板 + 区域裁剪精炼"""
    import torch
    from PIL import Image

    records = []
    with open(pred_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # 先去样板
    for rec in records:
        rec["report"] = strip_boilerplate(rec.get("report", ""))

    # 找出需要精炼的 FORGED 样本
    forged_recs = [r for r in records if '**[Conclusion]:** FORGED' in r.get("report", "")]
    print(f"FORGED 样本: {len(forged_recs)} 条需要区域裁剪精炼")

    if not forged_recs:
        with open(output_path, 'w') as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        print("无 FORGED 样本, 只做了去样板")
        return

    # 加载 MLLM
    print("Loading MLLM...")
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        mllm_path, torch_dtype=torch.bfloat16, device_map="auto")
    proc = AutoProcessor.from_pretrained(mllm_path)

    n_enhanced = 0
    n_crops_total = 0

    for rec in tqdm(forged_recs, desc="Crop refining"):
        report = rec["report"]

        # 提取 GROUNDING 坐标
        boxes = []
        for m in re.finditer(
            r'\[GROUNDING\]\s*[:：]\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', report
        ):
            boxes.append([int(m.group(i)) for i in range(1, 5)])

        if not boxes:
            continue

        # 找图片
        img_name = rec["image_name"]
        img_path = os.path.join(image_dir, img_name)
        if not os.path.isfile(img_path):
            for p in Path(image_dir).rglob(img_name):
                img_path = str(p)
                break
        if not os.path.isfile(img_path):
            continue

        try:
            image_pil = Image.open(img_path).convert("RGB")
        except:
            continue

        # 对每个区域裁剪并分析
        crop_details = []
        for box in boxes:
            try:
                crop_pil = crop_region(image_pil, box)
                detail = run_crop_mllm(model, proc, crop_pil)
                # 清理: 只取前 2-3 句
                sentences = re.split(r'(?<=[.!?])\s+', detail)
                detail = ' '.join(sentences[:3])
                crop_details.append(detail)
                n_crops_total += 1
            except Exception as e:
                crop_details.append("")

        # 增强 REASON
        if any(d for d in crop_details):
            rec["report"] = enhance_reason_with_crop(report, crop_details)
            n_enhanced += 1

    # 保存
    with open(output_path, 'w') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    print(f"\n✅ 精炼完成: {output_path}")
    print(f"   去样板: 全部 {len(records)} 条")
    print(f"   裁剪精炼: {n_enhanced}/{len(forged_recs)} 条 FORGED")
    print(f"   裁剪分析: {n_crops_total} 个区域")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    p1 = sub.add_parser("strip", help="只去样板 (不需要模型)")
    p1.add_argument("--pred", required=True)
    p1.add_argument("--output", required=True)

    p2 = sub.add_parser("refine", help="去样板 + 区域裁剪精炼")
    p2.add_argument("--pred", required=True)
    p2.add_argument("--image_dir", required=True)
    p2.add_argument("--mllm_path", required=True)
    p2.add_argument("--output", required=True)

    args = parser.parse_args()

    if args.cmd == "strip":
        run_strip(args.pred, args.output)
    elif args.cmd == "refine":
        run_refine(args.pred, args.image_dir, args.mllm_path, args.output)
    else:
        parser.print_help()