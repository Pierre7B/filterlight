# Carbon PDF Filter (Dockerized)

Batch-filter ESG/Carbon-relevant pages from PDFs using an XLM-Roberta base + LoRA adapter.

- **Entry points**  
  - `batch_carbon_filter.py`: batch mode (default in Docker image)  
  - `carbon_filter.py`: single-PDF mode (optional)
- **Model**: uses a local LoRA adapter directory (e.g. `Carbon_xlmr_lora/`) applied to `xlm-roberta-base`.

---

## Contents

```
.
├── batch_carbon_filter.py    # batch CLI (default ENTRYPOINT)
├── carbon_filter.py          # single-PDF CLI
├── Carbon_xlmr_lora/         # LoRA adapter directory
├── testpdf/                  # sample PDFs
├── cuts/                     # output folder (manifest + cut PDFs)
├── Dockerfile                # lean CPU-only image
├── docker-compose.yml        # optional compose
├── run_batch.sh              # helper for docker run
├── requirements.txt
└── README.md
```

---

## Prerequisites

- Docker (Desktop or Engine)
- (Optional) A Docker Hub / GHCR account if you’ll pull the prebuilt image.

---

## Option A — Pull & Run (recommended for teammates)

> Replace `pielebo/carbon-filter:cpu-slim` with your actual pushed image if different.

```bash
docker pull pielebo/carbon-filter:cpu-slim

# basic run (Linux/macOS)
docker run --rm --user 0:0   -v "$PWD/testpdf:/in:ro"   -v "$PWD/cuts:/out"   -v "$PWD/Carbon_xlmr_lora:/models/Carbon_xlmr_lora:ro"   pielebo/carbon-filter:cpu-slim   /in   --adapter /models/Carbon_xlmr_lora   --base-model xlm-roberta-base   --threshold 0.25   --out-root /out
```

- `/in` → folder with PDFs
- `--out-root /out` → output folder
- `--adapter …` → LoRA adapter directory (must contain tokenizer + adapter files)
- `--base-model xlm-roberta-base` → HF base model

> Why `--user 0:0`? Some host filesystems (exFAT/NTFS/external drives) can cause permission issues for non-root users. Running as root avoids read-permission surprises. If you prefer non-root, see **Permissions** below.

---

## Option B — Build Locally (if you want your own image)

```bash
# build the lean CPU-only image
docker build --no-cache -t carbon-filter:cpu-slim .

# run it
docker run --rm --user 0:0   -v "$PWD/testpdf:/in:ro"   -v "$PWD/cuts:/out"   -v "$PWD/Carbon_xlmr_lora:/models/Carbon_xlmr_lora:ro"   carbon-filter:cpu-slim   /in   --adapter /models/Carbon_xlmr_lora   --base-model xlm-roberta-base   --threshold 0.25   --out-root /out
```

**Push to registry (for teammates):**
```bash
# Docker Hub
docker tag carbon-filter:cpu-slim <your-dockerhub-username>/carbon-filter:cpu-slim
docker login
docker push <your-dockerhub-username>/carbon-filter:cpu-slim
```

---

## (Nice to have) Hugging Face cache for faster runs

Avoid re-downloading `xlm-roberta-base` on each run by using a **named volume**:

```bash
docker volume create hf_cache
docker run --rm --user 0:0   -e HF_HOME=/cache   -v hf_cache:/cache   -v "$PWD/testpdf:/in:ro"   -v "$PWD/cuts:/out"   -v "$PWD/Carbon_xlmr_lora:/models/Carbon_xlmr_lora:ro"   pielebo/carbon-filter:cpu-slim   /in --adapter /models/Carbon_xlmr_lora   --base-model xlm-roberta-base   --threshold 0.25 --out-root /out
```

First run will download ~1 GB into `hf_cache`. Next runs start immediately.

---

## docker-compose (optional)

`docker-compose.yml` (example):

```yaml
services:
  carbon_batch:
    image: pielebo/carbon-filter:cpu-slim
    user: "0:0"
    command: ["/in", "--adapter", "/models/Carbon_xlmr_lora", "--base-model", "xlm-roberta-base", "--threshold", "0.25", "--out-root", "/out"]
    environment:
      - HF_HOME=/cache
    volumes:
      - ./testpdf:/in:ro
      - ./cuts:/out
      - ./Carbon_xlmr_lora:/models/Carbon_xlmr_lora:ro
      - hf_cache:/cache
volumes:
  hf_cache:
```

Run:
```bash
docker compose up --pull always
```

---

## Single-PDF mode (optional)

```bash
# if you prefer running the single-file script
docker run --rm --user 0:0   -v "$PWD/testpdf:/in:ro"   -v "$PWD/cuts:/out"   -v "$PWD/Carbon_xlmr_lora:/models/Carbon_xlmr_lora:ro"   pielebo/carbon-filter:cpu-slim   python /app/carbon_filter.py /in/Climate-Report-2024.pdf   --adapter /models/Carbon_xlmr_lora   --base-model xlm-roberta-base   --threshold 0.25   --out /out/cut_Carbon_climate.pdf
```

---

## Helper script

`run_batch.sh` (included) wraps the long command:

```bash
bash ./run_batch.sh   /absolute/path/to/input_pdfs   /absolute/path/to/output_cuts   /absolute/path/to/Carbon_xlmr_lora   0.25
```

Defaults (if you just run `bash run_batch.sh`):  
- IN=`$PWD/testpdf`  
- OUT=`$PWD/cuts`  
- MODEL=`$PWD/Carbon_xlmr_lora`  
- THRESH=`0.25`

---

## Permissions & Filesystems (read this if you see errors)

- **PermissionError on adapter/tokenizer** → run with `--user 0:0` (as shown), or:
  ```bash
  # make the model world-readable on host
  chmod -R a+r Carbon_xlmr_lora
  find Carbon_xlmr_lora -type d -exec chmod a+rx {} \;
  ```
- **Stale file handle on /in** (often with external drives or network mounts) → use a **named volume** instead of a bind mount:
  ```bash
  docker volume create pdf_in pdf_out
  docker run --rm --user 0:0 -v pdf_in:/in -v "$PWD/testpdf:/src:ro" python:3.10-slim     bash -lc 'rm -rf /in/* && cp -a /src/. /in/'
  docker run --rm --user 0:0     -v pdf_in:/in:ro -v pdf_out:/out -v hf_cache:/cache     -v "$PWD/Carbon_xlmr_lora:/models/Carbon_xlmr_lora:ro"     pielebo/carbon-filter:cpu-slim /in --adapter /models/Carbon_xlmr_lora     --base-model xlm-roberta-base --threshold 0.25 --out-root /out
  docker run --rm --user 0:0 -v pdf_out:/out -v "$PWD/cuts:/host_out"     python:3.10-slim bash -lc 'cp -a /out/. /host_out/'
  ```
- **Root-owned outputs** (when using `--user 0:0`) → reclaim ownership:
  ```bash
  chown -R $(id -u):$(id -g) cuts
  ```

---

## Development notes

- The image is **CPU-only** (small & portable). If you need GPU, you’ll want a different Dockerfile + CUDA runtime.
- `requirements.txt` purposely excludes `torch`; Dockerfile installs a **CPU-only** PyTorch wheel:
  ```dockerfile
  RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.4.1
  ```
- Add `-e TOKENIZERS_PARALLELISM=false` to silence tokenizer warnings if needed.

---

## Troubleshooting

- _“Some weights … newly initialized”_ on startup: you may briefly see this while the base model loads; the LoRA adapter attaches right after. If outputs look random, double-check `--adapter` points to the correct folder and that you’re **not** passing `--no-merge`.
- _Slow first run_: use the **HF cache** volume to avoid re-downloading models each time.
- _Image too big_: this repo’s Dockerfile builds a ~**1.4 GB** image (CPU-only torch + minimal libs). If you see multi-GB images, you may have accidentally pulled CUDA wheels earlier—rebuild with `--no-cache` using this Dockerfile.

---

## License / Security

- Run locally or inside a trusted environment. The default instructions run as root for convenience with external filesystems; switch to non-root (`USER appuser` + remove `--user 0:0`) if your host paths honor POSIX permissions.

---

If you hit anything odd, open an issue with the exact `docker run` line and the traceback. Happy filtering! 🎯
