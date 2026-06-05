import json, os, sys, shutil
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

OUTPUT = "prediction_submit_raw.jsonl"
TEST_DIR = "test_images"
BATCH = 200

# 已完成的
done = set()
if os.path.isfile(OUTPUT):
    with open(OUTPUT) as f:
        for l in f:
            if l.strip():
                done.add(json.loads(l)["image_name"])
print(f"已完成: {len(done)}")

# 待处理
all_imgs = sorted(f for f in os.listdir(TEST_DIR) if f.endswith(('.jpg','.png')))
remaining = [f for f in all_imgs if f not in done]
print(f"总计: {len(all_imgs)}, 待处理: {len(remaining)}")

for start in range(0, len(remaining), BATCH):
    batch = remaining[start:start+BATCH]
    
    # 临时目录
    os.makedirs("_btmp", exist_ok=True)
    for f in os.listdir("_btmp"):
        os.unlink(f"_btmp/{f}")
    for f in batch:
        src = os.path.abspath(f"{TEST_DIR}/{f}")
        if not os.path.exists(f"_btmp/{f}"):
            os.symlink(src, f"_btmp/{f}")
    
    print(f"\n>>> 批次 {start//BATCH+1}: 第 {len(done)+start+1}-{len(done)+start+len(batch)} 张")
    
    ret = os.system(
        "python infer_final_v6.py "
        "--test_dir _btmp "
        "--mllm_path output/sft_v2e_merged "
        "--seg_path output/segformer_b5/best "
        "--output _btmp_out.jsonl"
    )
    
    # 追加结果
    if os.path.isfile("_btmp_out.jsonl"):
        with open("_btmp_out.jsonl") as fin, open(OUTPUT, "a") as fout:
            for l in fin:
                fout.write(l)
        n = sum(1 for _ in open("_btmp_out.jsonl"))
        os.remove("_btmp_out.jsonl")
        print(f"  ✅ 写入 {n} 条")
    else:
        print(f"  ⚠ 批次无输出, 继续")

shutil.rmtree("_btmp", ignore_errors=True)
total = sum(1 for _ in open(OUTPUT)) if os.path.isfile(OUTPUT) else 0
print(f"\n✅ 完成: {total}/{len(all_imgs)}")
