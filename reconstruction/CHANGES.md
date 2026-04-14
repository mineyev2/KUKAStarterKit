# Reconstruction Environment — Change Log

## Summary of Changes

### 1. `simple_trainer.py` — Import fix (`datasets.colmap` → `datasets.colmap2`)

**File:** `reconstruction/simple_trainer.py`, line 20  
**Change:**
```python
# Before
from datasets.colmap import Dataset, Parser

# After
from datasets.colmap2 import Dataset, Parser
```
**Why:** `datasets/colmap.py` uses `from pycolmap import SceneManager` which is deprecated and removed
in pycolmap ≥ 4.0. `datasets/colmap2.py` rewrites the same parser using the modern
`pycolmap.Reconstruction` API (no `SceneManager`). Switching the import here means
`simple_trainer.py` works with pycolmap 4.0.3 (built from the colmap submodule).

---

### 2. `colmap/src/pycolmap/estimators/ceres_bindings.cc` — Ceres version guard

**File:** `colmap/src/pycolmap/estimators/ceres_bindings.cc`, ~line 109  
**Change:** Added a `#if CERES_VERSION_MAJOR / MINOR` guard around the CUDA enum value:
```cpp
// Before — unconditionally references DenseLinearAlgebraLibraryType::CUDA
auto dlalt = ...
    .value("CUDA", ceres::DenseLinearAlgebraLibraryType::CUDA);   // compile error

// After — guarded so it compiles against Ceres 2.0.0 or 2.1+
auto dlalt = ...
    .value("EIGEN", ...).value("LAPACK", ...);
#if CERES_VERSION_MAJOR > 2 || \
    (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 1)
dlalt.value("CUDA", ceres::DenseLinearAlgebraLibraryType::CUDA);
#endif
```
**Why:** `ceres::DenseLinearAlgebraLibraryType::CUDA` was added in Ceres 2.1.0.
Ubuntu 22.04 and 24.04 both ship Ceres `2.0.0+dfsg1-5` via `apt`. pycolmap 4.0.3
(the version in the colmap submodule) unconditionally references this enum value,
causing a compile error. The guard makes pycolmap build against any Ceres ≥ 2.0.0;
on systems with Ceres ≥ 2.1 the CUDA binding is also exposed.

---

### 3. `reconstruction/Dockerfile` — Major rewrite

**File:** `reconstruction/Dockerfile`

**What changed and why:**

| Change | Reason |
|--------|--------|
| Base: `nvidia/cuda:13.0.0-devel-ubuntu24.04` | Ubuntu 24 required; CUDA 13.0 to match PyTorch 2.11.0+cu130 |
| Python 3.10 via `deadsnakes/ppa` | Ubuntu 24 ships Python 3.12; project requires 3.10 (`.python-version`) |
| `COLMAP_BUILD_WITH_GUI=OFF` + Qt packages removed | Headless server doesn't need Qt; dramatically reduces build time and image size |
| `COLMAP_BUILD_TESTS=OFF` | No need to build test binaries in a production image |
| `ENV PATH="/usr/local/cuda/bin:${PATH}"` | Ensures `nvcc` is on PATH inside the container before `uv sync` runs; the devel base image sets this but the explicit ENV is more robust across layer boundaries |
| `ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:..."` | Same rationale — ensures CUDA libs are findable during extension builds |
| `ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0;10.0;12.0"` | Required for `gsplat`, `fused-ssim`, `fused-bilagrid` to compile CUDA kernels; covers Turing through Blackwell |
| `ARG MAX_JOBS=4` / `ENV MAX_JOBS` | Prevents OOM on low-RAM builders during parallel ninja compilation |
| MKL replaced with `libblas-dev + liblapack-dev` | `libmkl-full-dev` requires Intel's apt repository (not standard); OpenBLAS from Ubuntu repos is sufficient for COLMAP |
| Build context: run from **repo root** | `colmap/` and `reconstruction/` are both needed; build with `-f reconstruction/Dockerfile .` |

**Build & run commands:**
```bash
# Build (always run from repo root — not from reconstruction/)
docker build -f reconstruction/Dockerfile -t reconstruction:latest .

# Override CUDA arch for your GPU (sm_89 = RTX 4090, sm_90 = H100, sm_120 = Blackwell)
docker build -f reconstruction/Dockerfile --build-arg CUDA_ARCHITECTURES=89 -t reconstruction:latest .

# Run with GPU + viser port 8080
docker run --gpus all -p 8080:8080 -it reconstruction:latest
```

---

### 4. `.dockerignore` — New file at repo root

**File:** `.dockerignore` (new)  
**Why:** `reconstruction/.venv/` contains thousands of installed package files. Without
this exclusion, every `docker build` would send several GB of cached venv files to the
Docker daemon as build context — unnecessarily slow and wasteful. The container runs
its own `uv sync` to build a fresh `.venv/` from scratch.

---

### 5. `pyproject.toml` — No changes required

**File:** `reconstruction/pyproject.toml`  
The file already had:
```toml
[tool.uv.sources]
pycolmap = { path = "../colmap" }
```
This was already correct — `pycolmap` builds from the local colmap submodule (pycolmap 4.0.3),
not from PyPI. No PyPI version of pycolmap currently ships CUDA 13 wheels, which is why
building from source is required. **This line was not changed.**

---

## What Was Tested

Environment tested on Ubuntu 22.04 with CUDA 13.0 (nvidia driver 580.126):

```
✔ uv sync — all 118 packages resolved and installed
    ✔ pycolmap 4.0.3 built from colmap/ submodule source (Ceres guard patch required)
    ✔ fused-bilagrid built from git source (CUDA extension compiled)
    ✔ fused-ssim built from git source (CUDA extension compiled)
    ✔ torch 2.11.0+cu130 installed
✔ import pycolmap  →  4.0.3
✔ import torch  →  2.11.0+cu130, cuda.is_available() = True
✔ from datasets.colmap2 import Dataset, Parser  →  OK
✔ simple_trainer.py default --data_dir data/360_v2/garden/ --data_factor 4 --result_dir ./results/garden
    ✔ Parser loaded 185 images from COLMAP sparse reconstruction
    ✔ Model initialized: 138,766 Gaussians
    ✔ Viser server: http://localhost:8080
    ✔ Training running: loss≈0.13, ~120 it/s at step 400/30000
```

## Running the Sample Trainer

From inside the `reconstruction/` directory:

```bash
# Download the mipnerf360 dataset (~12GB, only needed once)
python datasets/download_dataset.py

# Train on the garden scene (downsampled 4×)
CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default \
    --data_dir data/360_v2/garden/ --data_factor 4 \
    --result_dir ./results/garden

# Viser visualizer is available at http://localhost:8080 during training
```
