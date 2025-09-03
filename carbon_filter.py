#!/usr/bin/env python3
"""
Carbon PDF filter (standalone)

- Input: a PDF file path
- Output: a cut PDF that contains only pages whose score >= threshold
- Model: LoRA adapter over a base HF sequence classification model (default: xlm-roberta-base)
- Optional: use a pre-merged plain HF model folder instead of an adapter

Examples
--------
# Using a LoRA adapter folder
python carbon_filter.py input.pdf \
  --adapter models/Carbon_xlmr_lora --base-model xlm-roberta-base \
  --threshold 0.25 --out cuts/cut_Carbon_input.pdf

# Using a pre-merged HF folder (faster startup)
python carbon_filter.py input.pdf --hf-merged merged_models/Carbon_xlmr_merged

Notes
-----
- Assumes a binary classifier with logits [..., 2] where class-1 is "positive".
- If no page passes threshold, you can keep top-K via --ensure-at-least K.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Sequence

import fitz  # PyMuPDF
import torch
from tqdm import tqdm

# transformers / peft
from transformers import AutoTokenizer, AutoModelForSequenceClassification
try:
    from peft import PeftModel
except Exception:
    PeftModel = None  # Will error if user tries to use --adapter without peft installed


# --------------------------- PDF helpers ---------------------------

def load_pdf_pages_text(pdf_path: str) -> List[tuple[int, str]]:
    """Return [(1, text), (2, text), ...]."""
    pages: List[tuple[int, str]] = []
    doc = fitz.open(pdf_path)
    try:
        for i in range(doc.page_count):
            txt = doc.load_page(i).get_text("text") or ""
            pages.append((i + 1, txt))
    finally:
        doc.close()
    return pages


def safe_build_cut_pdf(src_pdf: str,
                       selected_pages: Sequence[int],
                       out_pdf: str,
                       *,
                       try_vector_first: bool = True,
                       raster_dpi: int = 200) -> None:
    """
    Build a cut PDF with the given pages (1-based). Robust against tricky PDFs.

    Strategy:
      1) Try a normal vector copy with insert_pdf(...). If MuPDF chokes
         (e.g., FzErrorLimit with widget grafts), fall back to rasterization.
      2) Raster fallback draws each source page to a pixmap and embeds it as a new page.
    """
    selected = sorted(set(int(p) for p in selected_pages if p is not None and p >= 1))
    if not selected:
        raise ValueError("No pages to keep.")

    src = fitz.open(src_pdf)
    try:
        if try_vector_first:
            out = fitz.open()
            try:
                for p in selected:
                    # One page at a time: minimizes widget/link graft complexity
                    out.insert_pdf(src, from_page=p - 1, to_page=p - 1)
                out.save(out_pdf)
                out.close()
                return
            except Exception as e:
                # Fall back to raster mode
                out.close()
                print(f"[warn] vector copy failed ({e}); falling back to rasterized pages...")

        # Rasterized fallback
        out = fitz.open()
        try:
            for p in selected:
                page = src.load_page(p - 1)
                # Render to pixmap
                mat = fitz.Matrix(raster_dpi / 72, raster_dpi / 72)  # scale
                pix = page.get_pixmap(matrix=mat, alpha=False)
                # Create a blank page with same size as rendered image
                rect = fitz.Rect(0, 0, pix.width, pix.height)
                new_page = out.new_page(width=rect.width, height=rect.height)
                # Insert the image
                img = out.insert_image(new_page.rect, stream=pix.tobytes("png"))
            out.save(out_pdf)
        finally:
            out.close()
    finally:
        src.close()


# --------------------------- Model loading ---------------------------

def _tokenizer_from(adapter_dir: str | None, base_model: str):
    """
    Prefer tokenizer artifacts in adapter_dir if present; else fall back to base.
    """
    if adapter_dir:
        has_tok = any(
            (Path(adapter_dir) / name).exists()
            for name in ("tokenizer.json", "sentencepiece.bpe.model", "vocab.json", "special_tokens_map.json")
        )
        if has_tok:
            return AutoTokenizer.from_pretrained(adapter_dir, use_fast=True)
    return AutoTokenizer.from_pretrained(base_model, use_fast=True)


def load_lora_model(adapter_dir: str, base_model: str, device: str, merge: bool = True):
    """
    Load base + LoRA adapter. Optionally merge-and-unload for faster inference.
    """
    if PeftModel is None:
        raise RuntimeError("peft is not installed. Install with: pip install peft")

    tok = _tokenizer_from(adapter_dir, base_model)
    base = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=2)
    model = PeftModel.from_pretrained(base, adapter_dir)

    if merge:
        model = model.merge_and_unload()  # returns a plain HF model

    model.eval()
    model.to(device)
    return tok, model


def load_hf_model(model_dir: str, device: str):
    """
    Load a plain HF model directory (e.g., a pre-merged folder).
    """
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    model.to(device)
    return tok, model


# --------------------------- Scoring ---------------------------

@torch.inference_mode()
def score_pages(texts: List[str],
                tok: AutoTokenizer,
                model: AutoModelForSequenceClassification,
                device: str,
                batch_size: int = 8,
                max_length: int = 512) -> List[float]:
    """
    Return probability for positive class (index 1).
    """
    out: List[float] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Scoring", unit="batch"):
        batch = texts[i:i + batch_size]
        inputs = tok(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        logits = model(**inputs).logits  # [B, 2]
        probs = torch.softmax(logits, dim=-1)[:, 1]  # positive class
        out.extend(probs.detach().cpu().tolist())
    return out


# --------------------------- CLI ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Carbon PDF page filter (LoRA / HF).")
    ap.add_argument("pdf", help="Input PDF path.")
    ap.add_argument("--out", help="Output PDF path. Default: cut_Carbon_<inputname>.pdf next to input.")
    # Model sources
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--adapter", help="Path to LoRA adapter folder (e.g., models/Carbon_xlmr_lora).")
    group.add_argument("--hf-merged", help="Path to a plain HF folder (pre-merged).")
    ap.add_argument("--base-model", default="xlm-roberta-base",
                    help="Base model name or path (used with --adapter). Default: xlm-roberta-base.")
    ap.add_argument("--no-merge", action="store_true",
                    help="If set with --adapter, do NOT merge LoRA into base (slower).")
    # Inference params
    ap.add_argument("--threshold", type=float, default=0.25, help="Keep pages with score >= threshold. Default: 0.25")
    ap.add_argument("--ensure-at-least", type=int, default=0,
                    help="If nothing meets threshold, keep top-K pages (0 = keep none).")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"],
                    help="Device selection. 'auto' picks CUDA / MPS when available.")
    ap.add_argument("--raster-fallback", action="store_true",
                    help="Force raster fallback (skip vector copy). Useful for broken PDFs.")
    args = ap.parse_args()

    in_pdf = Path(args.pdf)
    if not in_pdf.exists():
        raise SystemExit(f"Input PDF not found: {in_pdf}")

    # Output path
    if args.out:
        out_pdf = Path(args.out)
    else:
        out_pdf = in_pdf.with_name(f"cut_Carbon_{in_pdf.name}")

    # Device pick
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"  # Apple Silicon
        else:
            device = "cpu"
    else:
        device = args.device

    # Load model
    if args.hf_merged:
        print(f"[info] Loading merged HF model from: {args.hf_merged}")
        tok, model = load_hf_model(args.hf_merged, device)
    else:
        if PeftModel is None:
            raise SystemExit("peft is not installed. Install with: pip install peft")
        if not args.adapter:
            raise SystemExit("Please provide --adapter path or --hf-merged path.")
        print(f"[info] Loading base + LoRA adapter: base={args.base_model}, adapter={args.adapter}")
        tok, model = load_lora_model(
            adapter_dir=args.adapter,
            base_model=args.base_model,
            device=device,
            merge=not args.no_merge
        )

    # Read PDF
    pages = load_pdf_pages_text(str(in_pdf))
    page_nums = [p for p, _ in pages]
    texts = [t for _, t in pages]

    # Score
    probs = score_pages(
        texts,
        tok=tok,
        model=model,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length
    )

    # Pick kept pages
    kept = [p for p, s in zip(page_nums, probs) if float(s) >= float(args.threshold)]
    if not kept and args.ensure_at_least > 0:
        # Keep top-K
        order = sorted(zip(page_nums, probs), key=lambda x: x[1], reverse=True)
        kept = [p for p, _ in order[:args.ensure_at_least]]

    # Summary
    print("\n=== Summary ===")
    print(f"PDF: {in_pdf.name}")
    print(f"Pages: {len(page_nums)}")
    print(f"Threshold: {args.threshold}")
    print(f"Kept pages: {kept if kept else 'none'}")
    if kept:
        # Build cut PDF
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        safe_build_cut_pdf(
            str(in_pdf),
            kept,
            str(out_pdf),
            try_vector_first=not args.raster_fallback,
            raster_dpi=200
        )
        print(f"[ok] Wrote: {out_pdf}")
    else:
        print("[note] No page met the threshold and ensure-at-least=0 ; no file written.")


if __name__ == "__main__":
    main()
