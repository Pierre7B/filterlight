# 1) Create venv (recommended)
python -m venv .venv && source .venv/bin/activate

# 2) Install PyTorch (pick the right command from pytorch.org for your OS/CUDA)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3) Install the rest
pip install -r requirements.txt

# 4a) Run with LoRA adapter (slightly slower first time; merges by default)
python carbon_filter.py /path/to/input.pdf \
  --adapter ./Carbon_xlmr_lora \
  --base-model xlm-roberta-base \
  --threshold 0.25 \
  --out ./cut_Carbon_input.pdf

# 4b) (Optional) If you already merged LoRA -> HF once:
python carbon_filter.py /path/to/input.pdf \
  --hf-merged /path/to/merged_models/Carbon_xlmr_merged \
  --threshold 0.25


If some PDFs fail during vector copy, add --raster-fallback (slower, but robust).

Want at least one page even if none pass threshold? --ensure-at-least 1.