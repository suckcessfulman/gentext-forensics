#!/usr/bin/env python3
"""
LLM Judge: 用 DeepSeek API 评估 REASON 质量
=============================================
比 ROUGE-L 更接近比赛真实评分的方式。

用法:
  export DEEPSEEK_API_KEY="sk-..."
  python llm_judge.py \
      --pred /root/autodl-tmp/gentext/prediction_v4_stripped.jsonl \
      --holdout /root/autodl-tmp/gentext/holdout.jsonl \
      --n_samples 30 \
      --output /root/autodl-tmp/gentext/diagnosis/judge_results.json

费用估算: 30 样本 × ~1000 token/样本 ≈ 30K token ≈ ¥0.03 (DeepSeek 极便宜)
"""
import argparse, json, os, re, random, time, sys
from collections import defaultdict

JUDGE_PROMPT = """你是一个文档篡改检测报告的评估专家。请对以下两份报告进行对比评分。

【标准答案 (GT)】
{gt_report}

【模型输出 (Pred)】
{pred_report}

请从以下 4 个维度打分 (1-5 分):

1. **Factuality (事实准确性)**: 模型描述的异常是否真实存在？坐标是否指向正确区域？有没有编造不存在的异常？
   - 5: 所有描述的异常都与 GT 一致
   - 3: 部分正确，部分偏差
   - 1: 大部分是编造的

2. **Reasoning (推理质量)**: REASON 的分析是否有逻辑？是否说明了"为什么这是篡改"而不是泛泛而谈？
   - 5: 给出了具体、可验证的视觉/语义线索
   - 3: 有一些具体描述但也有套话
   - 1: 全是模板化的泛泛之词

3. **Completeness (完整性)**: GT 中提到的所有篡改区域，模型是否都检测到了？
   - 5: 检测到了所有区域
   - 3: 检测到一半以上
   - 1: 遗漏大部分

4. **Specificity (具体性)**: REASON 是否包含具体的视觉细节（如"字体粗细不一致"、"边缘有锯齿"、"背景噪声模式不同"）？
   - 5: 每个异常都有具体的视觉描述
   - 3: 混合了具体和模糊描述
   - 1: 只有"该区域有异常"之类的空话

请严格按以下 JSON 格式输出，不要有其他内容:
{{"factuality": <1-5>, "reasoning": <1-5>, "completeness": <1-5>, "specificity": <1-5>, "comment": "<一句话总评>"}}"""


def call_deepseek(prompt, api_key, model="deepseek-chat", max_retries=3):
    """调用 DeepSeek API"""
    import urllib.request
    
    for attempt in range(max_retries):
        try:
            data = json.dumps({
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 300,
            }).encode('utf-8')
            
            req = urllib.request.Request(
                "https://api.deepseek.com/chat/completions",
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
            )
            
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode('utf-8'))
                content = result["choices"][0]["message"]["content"].strip()
                
                # 提取 JSON
                # 尝试直接解析
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    # 尝试提取 JSON 块
                    m = re.search(r'\{[^}]+\}', content, re.DOTALL)
                    if m:
                        return json.loads(m.group())
                    print(f"    ⚠ 无法解析响应: {content[:100]}")
                    
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"    ❌ API 错误: {e}")
    
    return None


def extract_report_for_judge(report, max_chars=1500):
    """提取报告的关键内容用于 judge (去掉格式标签减少 token)"""
    # 保留 ANOMALY + REASON + GROUNDING + SUMMARY
    text = report
    # 去掉 ** 标记
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    # 截断
    if len(text) > max_chars:
        text = text[:max_chars] + "...[截断]"
    return text


def main():
    parser = argparse.ArgumentParser(description="LLM Judge 评估")
    parser.add_argument("--pred", required=True)
    parser.add_argument("--holdout", required=True)
    parser.add_argument("--n_samples", type=int, default=30)
    parser.add_argument("--output", default=None)
    parser.add_argument("--api_key", default=None,
                        help="DeepSeek API key (或设 DEEPSEEK_API_KEY 环境变量)")
    args = parser.parse_args()
    
    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("❌ 需要设置 DeepSeek API key:")
        print("   export DEEPSEEK_API_KEY='sk-...'")
        print("   或 --api_key sk-...")
        sys.exit(1)
    
    # 加载数据
    gt_map = {}
    with open(args.holdout) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                gt_map[rec["image_file"]] = rec
    
    preds = []
    with open(args.pred) as f:
        for line in f:
            if line.strip():
                preds.append(json.loads(line))
    
    # 找 TP 样本 (GT=FORGED, Pred=FORGED, 有 GT report_text)
    tp_samples = []
    for pred in preds:
        gt = gt_map.get(pred["image_name"])
        if not gt:
            continue
        if gt["gt_label"] != "FORGED":
            continue
        if "**[Conclusion]:** FORGED" not in pred.get("report", ""):
            continue
        if not gt.get("report_text"):
            continue
        tp_samples.append((pred, gt))
    
    print(f"TP 样本: {len(tp_samples)} 个")
    
    # 随机抽样
    random.seed(42)
    n = min(args.n_samples, len(tp_samples))
    selected = random.sample(tp_samples, n)
    print(f"抽样评估: {n} 个")
    
    # 逐个评估
    results = []
    scores_all = defaultdict(list)
    
    for i, (pred, gt) in enumerate(selected):
        name = pred["image_name"]
        gt_text = extract_report_for_judge(gt["report_text"])
        pred_text = extract_report_for_judge(pred["report"])
        
        prompt = JUDGE_PROMPT.format(gt_report=gt_text, pred_report=pred_text)
        
        print(f"  [{i+1}/{n}] {name}...", end=" ", flush=True)
        scores = call_deepseek(prompt, api_key)
        
        if scores and all(k in scores for k in ["factuality", "reasoning", "completeness", "specificity"]):
            print(f"F={scores['factuality']} R={scores['reasoning']} "
                  f"C={scores['completeness']} S={scores['specificity']}")
            scores["image"] = name
            scores["language"] = gt.get("language_code", "?")
            results.append(scores)
            for k in ["factuality", "reasoning", "completeness", "specificity"]:
                scores_all[k].append(scores[k])
        else:
            print("跳过 (解析失败)")
        
        time.sleep(0.3)  # 限速
    
    # 汇总
    print(f"\n{'='*60}")
    print(f"  LLM Judge 评估结果 ({len(results)}/{n} 成功)")
    print(f"{'='*60}")
    
    if not results:
        print("  ❌ 无有效结果")
        return
    
    for dim in ["factuality", "reasoning", "completeness", "specificity"]:
        vals = scores_all[dim]
        avg = sum(vals) / len(vals)
        bar = "█" * int(avg) + "░" * (5 - int(avg))
        print(f"  {dim:<15} {avg:.2f}/5  {bar}")
    
    overall = sum(sum(scores_all[k]) for k in scores_all) / (len(results) * 4)
    print(f"  {'OVERALL':<15} {overall:.2f}/5")
    
    # 按语种拆分
    lang_scores = defaultdict(lambda: defaultdict(list))
    for r in results:
        lang = r.get("language", "?")
        for k in ["factuality", "reasoning", "completeness", "specificity"]:
            lang_scores[lang][k].append(r[k])
    
    if len(lang_scores) > 1:
        print(f"\n  按语种拆分:")
        for lang in sorted(lang_scores.keys()):
            ls = lang_scores[lang]
            n_lang = len(ls["factuality"])
            avg = sum(sum(ls[k]) for k in ls) / (n_lang * 4)
            print(f"    {lang}: {avg:.2f}/5 (n={n_lang})")
    
    # 最差样本
    print(f"\n  最差 5 个:")
    ranked = sorted(results, 
                    key=lambda r: sum(r[k] for k in ["factuality", "reasoning", "completeness", "specificity"]))
    for r in ranked[:5]:
        total = sum(r[k] for k in ["factuality", "reasoning", "completeness", "specificity"])
        print(f"    {r['image']}  {total}/20  {r.get('comment', '')[:60]}")
    
    # 保存
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump({
                "summary": {k: sum(v)/len(v) for k, v in scores_all.items()},
                "overall": overall,
                "n_evaluated": len(results),
                "details": results,
            }, f, indent=2, ensure_ascii=False)
        print(f"\n  💾 {args.output}")


if __name__ == "__main__":
    main()