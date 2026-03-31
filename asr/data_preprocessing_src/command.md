# Step 1: 已完成 (preprocess.py 已处理全部 303 个 CHA)
uv run python preprocess.py
# Step 2: 对齐全部音频 (预计 ~303 × 3min ≈ 15 小时)
uv run python align.py
# Step 3: 构建最终数据集
uv run python build_dataset.py