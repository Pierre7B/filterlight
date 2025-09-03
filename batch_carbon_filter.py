#!/usr/bin/env python3
"""
Batch Carbon PDF filter

Scans an input folder (optionally recursively), scores each PDF with the Carbon
classifier (LoRA adapter or pre-merged HF), and writes cut PDFs under:

    cuts/Carbon/<YYYYMMDD>/cut_Carbon_<original>.pdf

Also writes a manifest CSV with per-file results.

Examples
--------
python batch_carbon_filter.py /path/to/pdfs \
  --adapter models/Carbon_xlmr_lora --base-model xlm-roberta-base \
  --threshold 0.25 --recursive

# If you already merged the adapter:
python batch_carbon_filter.py /path/to/pdfs --hf-merged merged_models/Carbon_xlmr_merged
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
from tqdm import tqdm

# Import helpers from the single-file CLI
try:
    from carbon_filter import (
        load_lora_model,
        load_hf_model,
        load_pdf_pages_text,
        safe_build_cut_pdf,
        score_pages,
    )
except Exception as e:
    print("[error] Could not import carbon_filter.py. Place this file next to carbon_filter.py.", file=sys.stderr)
    raise

def find_pdfs(root: Path, recursive: bool) -> List[Path]:
    if root.is_file() and root.suffix.lower() == ".pdf":
        return [root]
    pattern = "**/*.pdf" if recursive else "*.pdf"
    return sorted([p for p in root.glob(pattern) if p.is_file()])

def ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def pick_device(selection: str) -> str:
    if selection == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return selection

def process_one(pdf: Path,
                out_dir: Path,
                tok,
                model,
                device: str,
                threshold: float,
                ensure_at_least: int,
                batch_size: int,
                max_length: int,
                raster_fallback: bool) -> Tuple[int, int, List[int], Path | None]:
    # Read
    pages = load_pdf_pages_text(str(pdf))
    page_nums = [p for p, _ in pages]
    texts = [t for _, t in pages]

    # Score
    probs = score_pages(
        texts, tok=tok, model=model, device=device,
        batch_size=batch_size, max_length=max_length
    )

    # Keep
    kept = [p for p, s in zip(page_nums, probs) if float(s) >= float(threshold)]
    if not kept and ensure_at_least > 0:
        order = sorted(zip(page_nums, probs), key=lambda x: x[1], reverse=True)
        kept = [p for p, _ in order[:ensure_at_least]]

    # Output
    out_pdf = None
    if kept:
        out_name = f"cut_Carbon_{pdf.name}"
        out_pdf = out_dir / out_name
        ensure_parent(out_pdf)
        safe_build_cut_pdf(
            str(pdf), kept, str(out_pdf),
            try_vector_first=not raster_fallback, raster_dpi=200
        )

    return len(page_nums), len(kept), kept, out_pdf

def main():
    ap = argparse.ArgumentParser(description="Batch Carbon PDF filter → cuts/Carbon/<YYYYMMDD>/")
    ap.add_argument("input", help="Folder (or single PDF).")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--adapter", help="LoRA adapter folder (e.g., models/Carbon_xlmr_lora).")
    group.add_argument("--hf-merged", help="Pre-merged HF folder (e.g., merged_models/Carbon_xlmr_merged).")
    ap.add_argument("--base-model", default="xlm-roberta-base", help="Base model (with --adapter).")
    ap.add_argument("--no-merge", action="store_true", help="Do not merge adapter (slower).")
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--ensure-at-least", type=int, default=0, help="If none pass threshold, keep top-K pages.")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--recursive", action="store_true", help="Scan input folder recursively.")
    ap.add_argument("--out-root", default=None, help="Override output root. Default: cuts/Carbon/<YYYYMMDD>")
    ap.add_argument("--raster-fallback", action="store_true", help="Force raster mode (robust for tricky PDFs).")
    args = ap.parse_args()

    root = Path(args.input).expanduser().resolve()
    if not root.exists():
        sys.exit(f"Input path not found: {root}")

    # Output root
    date_tag = datetime.now().strftime("%Y%m%d")
    out_root = Path(args.out_root) if args.out_root else Path("cuts") / "Carbon" / date_tag
    out_root.mkdir(parents=True, exist_ok=True)

    # Load model once
    device = pick_device(args.device)
    if args.hf_merged:
        print(f"[info] Loading merged HF model: {args.hf_merged}")
        tok, model = load_hf_model(args.hf_merged, device)
    else:
        print(f"[info] Loading base + LoRA adapter: base={args.base_model}, adapter={args.adapter}")
        tok, model = load_lora_model(
            adapter_dir=args.adapter,
            base_model=args.base_model,
            device=device,
            merge=not args.no_merge
        )

    # Collect PDFs
    pdfs = find_pdfs(root, args.recursive)
    if not pdfs:
        print("[warn] No PDFs found.")
        return

    print(f"[info] Found {len(pdfs)} PDFs under {root}")
    manifest_path = out_root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as mf:
        writer = csv.writer(mf)
        writer.writerow([
            "input_path", "pages_total", "kept_count", "kept_pages",
            "threshold", "out_pdf"
        ])

        for pdf in tqdm(pdfs, desc="Batch", unit="pdf"):
            try:
                total, kept_n, kept_pages, out_pdf = process_one(
                    pdf=pdf,
                    out_dir=out_root,
                    tok=tok,
                    model=model,
                    device=device,
                    threshold=args.threshold,
                    ensure_at_least=args.ensure_at_least,
                    batch_size=args.batch_size,
                    max_length=args.max_length,
                    raster_fallback=args.raster_fallback,
                )
                writer.writerow([
                    str(pdf),
                    total,
                    kept_n,
                    " ".join(map(str, kept_pages)) if kept_pages else "",
                    args.threshold,
                    str(out_pdf) if out_pdf else "",
                ])
            except Exception as e:
                # Log failure row
                writer.writerow([str(pdf), "", "", "", args.threshold, f"ERROR: {e}"])
                print(f"[error] {pdf.name}: {e}", file=sys.stderr)

    print(f"\n[ok] Done. Cuts in: {out_root}\n[ok] Manifest: {manifest_path}")

if __name__ == "__main__":
    main()

