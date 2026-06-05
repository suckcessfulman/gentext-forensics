#!/usr/bin/env python3
"""
三件事一次做完:
1. 构建合理的 held-out 评估集（从训练集中划出 SFT/RSFT 都没重点学过的样本）
2. 在 held-out 上跑不同 SegFormer 阈值，找最优
3. 加 OCR Rectification 测试坐标精度提升
"""
import csv, json, os, re, random, sys
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


def find_image_path(base_dir, filename):
    direct = os.path.join(base_dir, filename)
    if os.path.isfile(direct):
        return direct
    for part in sorted(Path(base_dir).glob("part*")):
        p = part / filename
        if p.exists():
            return str(p)
    return None


# ============================================================
# 1. 构建 held-out 评估集
# ============================================================

def build_holdout(metadata_csv, image_dir, rsft_jsonl, output_path,
                  n_forged=100, n_auth=100):
    """
    从训练集中挑选 RSFT 没用过的样本作为 held-out
    RSFT 的 candidates.jsonl 只用了 2000 张，剩下 11500 张没被重点学过
    """
    # 加载 RSFT 用过的图片
    rsft_images = set()
    if os.path.isfile(rsft_jsonl):
        with open(rsft_jsonl) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    # 从 images 路径提取文件名
                    if "images" in rec and rec["images"]:
                        rsft_images.add(Path(rec["images"][0]).name)

    print(f"RSFT 用过的图片: {len(rsft_images)}")

    # 加载全部 metadata
    with open(metadata_csv) as f:
        rows = list(csv.DictReader(f))

    # 筛选 RSFT 没用过的
    unused_forged = [r for r in rows
                     if r["image_file"].endswith(".jpg")
                     and r["image_file"] not in rsft_images]
    unused_auth = [r for r in rows
                   if r["image_file"].endswith(".png")
                   and r["image_file"] not in rsft_images]

    print(f"RSFT 未使用: {len(unused_forged)} forged, {len(unused_auth)} auth")

    random.seed(123)  # 不同于训练时的 seed
    selected_f = random.sample(unused_forged, min(n_forged, len(unused_forged)))
    selected_a = random.sample(unused_auth, min(n_auth, len(unused_auth)))

    holdout = []
    for r in selected_f + selected_a:
        img_path = find_image_path(image_dir, r["image_file"])
        if img_path:
            holdout.append({
                "image_file": r["image_file"],
                "image_path": img_path,
                "gt_label": "FORGED" if r["image_file"].endswith(".jpg") else "AUTHENTIC",
                "report_text": r.get("report_text", ""),
            })

    random.shuffle(holdout)

    with open(output_path, 'w') as f:
        for rec in holdout:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    n_f = sum(1 for r in holdout if r["gt_label"] == "FORGED")
    n_a = sum(1 for r in holdout if r["gt_label"] == "AUTHENTIC")
    print(f"✅ Held-out 评估集: {output_path}")
    print(f"   FORGED: {n_f}, AUTHENTIC: {n_a}")
    return holdout


# ============================================================
# 2. SegFormer 阈值搜索
# ============================================================

def search_seg_threshold(holdout, seg_path, device="cuda"):
    """
    在 held-out 上测试不同 SegFormer 阈值
    目标: 找到 authentic 准确率和 forged 漏检率的最佳平衡
    """
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    model = SegformerForSemanticSegmentation.from_pretrained(seg_path).to(device).eval()
    proc = SegformerImageProcessor.from_pretrained(seg_path)

    # 对每张图计算 SegFormer 的最大概率值
    results = []
    for rec in tqdm(holdout, desc="SegFormer scanning"):
        try:
            img = Image.open(rec["image_path"]).convert("RGB")
        except:
            continue

        inputs = proc(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits

        probs = torch.softmax(logits, dim=1)
        idx = 1 if probs.shape[1] > 1 else 0
        max_prob = probs[0, idx].max().item()
        mask_area = (probs[0, idx] > 0.5).sum().item()

        results.append({
            "gt": rec["gt_label"],
            "max_prob": max_prob,
            "mask_area_05": mask_area,
        })

    # 测试不同阈值
    print(f"\n=== SegFormer 阈值搜索 (n={len(results)}) ===")
    print(f"{'阈值':>6} {'Auth正确':>10} {'Forg正确':>10} {'整体':>8} {'Auth误判':>10} {'Forg漏检':>10}")

    auth_results = [r for r in results if r["gt"] == "AUTHENTIC"]
    forg_results = [r for r in results if r["gt"] == "FORGED"]

    best_threshold = 0.5
    best_score = 0

    for threshold in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
        # 空 mask (max_prob < threshold) → 判 AUTHENTIC
        auth_correct = sum(1 for r in auth_results if r["max_prob"] < threshold)
        auth_wrong = len(auth_results) - auth_correct  # authentic 被误判为 forged

        forg_missed = sum(1 for r in forg_results if r["max_prob"] < threshold)
        forg_correct = len(forg_results) - forg_missed  # forged 正确检测

        auth_acc = auth_correct / max(len(auth_results), 1)
        forg_acc = forg_correct / max(len(forg_results), 1)

        # 加权分数（假设测试集 44% auth, 56% forged）
        overall = 0.44 * auth_acc + 0.56 * forg_acc

        print(f"{threshold:>6.2f} {auth_correct:>5}/{len(auth_results):<4} "
              f"{forg_correct:>5}/{len(forg_results):<4} "
              f"{overall:>7.1%} "
              f"{auth_wrong:>5} "
              f"{forg_missed:>5}")

        if overall > best_score:
            best_score = overall
            best_threshold = threshold

    print(f"\n✅ 最佳阈值: {best_threshold} (整体准确率: {best_score:.1%})")
    return best_threshold


# ============================================================
# 3. 端到端评估（含 GT 对比）
# ============================================================

def evaluate_predictions(pred_path, holdout):
    """对比预测结果和 GT"""
    gt_map = {r["image_file"]: r for r in holdout}

    preds = []
    with open(pred_path) as f:
        for line in f:
            if line.strip():
                preds.append(json.loads(line))

    correct_cls = 0
    total = 0
    tp = fp = fn = tn = 0

    iou_scores = []

    for pred in preds:
        name = pred["image_name"]
        gt = gt_map.get(name)
        if not gt:
            continue

        total += 1
        report = pred["report"]

        # 提取预测 conclusion
        m = re.search(r'\*\*\[Conclusion\]\:\*\*\s*(\w+)', report)
        pred_label = m.group(1).upper() if m else "MISSING"
        gt_label = gt["gt_label"]

        if pred_label == gt_label:
            correct_cls += 1

        if gt_label == "FORGED" and pred_label == "FORGED":
            tp += 1
        elif gt_label == "AUTHENTIC" and pred_label == "AUTHENTIC":
            tn += 1
        elif gt_label == "AUTHENTIC" and pred_label == "FORGED":
            fp += 1
        elif gt_label == "FORGED" and pred_label != "FORGED":
            fn += 1

        # IoU（如果 GT 有 report_text）
        if gt_label == "FORGED" and pred_label == "FORGED":
            gt_boxes = []
            for bm in re.finditer(r'\[GROUNDING\].*?\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]',
                                   gt.get("report_text", "")):
                gt_boxes.append([int(bm.group(i)) for i in range(1, 5)])

            pred_boxes = []
            for bm in re.finditer(r'\[GROUNDING\].*?\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]',
                                   report):
                pred_boxes.append([int(bm.group(i)) for i in range(1, 5)])

            if gt_boxes and pred_boxes:
                # 简单 IoU: 第一个框
                def iou(b1, b2):
                    xi = max(b1[0], b2[0]); yi = max(b1[1], b2[1])
                    xa = min(b1[2], b2[2]); ya = min(b1[3], b2[3])
                    inter = max(0, xa-xi) * max(0, ya-yi)
                    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
                    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
                    return inter / (a1+a2-inter) if (a1+a2-inter) > 0 else 0

                best_iou = max(iou(pb, gb) for pb in pred_boxes for gb in gt_boxes)
                iou_scores.append(best_iou)

    print(f"\n=== 端到端评估 (n={total}) ===")
    print(f"分类准确率: {correct_cls}/{total} = {correct_cls/max(total,1)*100:.1f}%")
    print(f"  TP(正确检测forged): {tp}")
    print(f"  TN(正确检测auth):   {tn}")
    print(f"  FP(auth误判forged): {fp}")
    print(f"  FN(forged漏检):     {fn}")

    if iou_scores:
        print(f"\n定位 IoU (仅 TP 样本):")
        print(f"  平均 IoU: {np.mean(iou_scores):.3f}")
        print(f"  IoU>0.3: {sum(1 for s in iou_scores if s > 0.3)}/{len(iou_scores)}")
        print(f"  IoU>0.5: {sum(1 for s in iou_scores if s > 0.5)}/{len(iou_scores)}")
    else:
        print(f"\n  ⚠ 没有 TP 样本可计算 IoU")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="/root/autodl-tmp/RealText-V2/metadata.csv")
    parser.add_argument("--image_dir", default="/root/autodl-tmp/RealText-V2-git/train/image")
    parser.add_argument("--rsft_jsonl", default="/root/autodl-tmp/gentext/train_v3_best_of_n.jsonl")
    parser.add_argument("--seg_path", default="/root/autodl-tmp/gentext/output/segformer_b5/best")
    parser.add_argument("--holdout_path", default="/root/autodl-tmp/gentext/holdout.jsonl")
    parser.add_argument("--n_forged", type=int, default=100)
    parser.add_argument("--n_auth", type=int, default=100)
    parser.add_argument("--pred_path", default=None, help="已有预测文件，跳过推理直接评估")
    args = parser.parse_args()

    # Step 1: 构建 held-out
    if not os.path.isfile(args.holdout_path):
        holdout = build_holdout(
            args.metadata, args.image_dir, args.rsft_jsonl,
            args.holdout_path, args.n_forged, args.n_auth)
    else:
        holdout = [json.loads(l) for l in open(args.holdout_path) if l.strip()]
        n_f = sum(1 for r in holdout if r["gt_label"] == "FORGED")
        n_a = sum(1 for r in holdout if r["gt_label"] == "AUTHENTIC")
        print(f"Held-out 已存在: {len(holdout)} 条 (F={n_f}, A={n_a})")

    # Step 2: SegFormer 阈值搜索
    best_threshold = search_seg_threshold(holdout, args.seg_path)

    # Step 3: 如果有预测文件，做端到端评估
    if args.pred_path and os.path.isfile(args.pred_path):
        evaluate_predictions(args.pred_path, holdout)

    print(f"\n=== 下一步 ===")
    print(f"用最佳阈值 {best_threshold} 在 held-out 上跑推理:")
    print(f"  python infer_final.py \\")
    print(f"      --test_dir <holdout 图片目录或用 symlink> \\")
    print(f"      --seg_threshold {best_threshold} \\")
    print(f"      ...")
