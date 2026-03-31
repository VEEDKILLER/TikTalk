# Step 1: Completed (preprocess.py has processed all 303 CHA files)
uv run python preprocess.py
# Step 2: Align all audio files (estimated ~303 × 3 min ≈ 15 hours)
uv run python align.py
# Step 3: Construct final dataset
uv run python build_dataset.py