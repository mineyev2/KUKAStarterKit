# Build from repo root:
#   docker build -f reconstruction/Dockerfile -t reconstruction:latest .
#
# Override CUDA architecture for your GPU (e.g. 89 for RTX 4090, 90 for H100):
#   docker build -f reconstruction/Dockerfile --build-arg CUDA_ARCHITECTURES=89 -t reconstruction:latest .
#
# Run with GPU + viser on port 8080:
#   docker run --gpus all -p 8080:8080 -it reconstruction:latest

FROM nvidia/cuda:13.0.0-devel-ubuntu24.04

ARG CUDA_ARCHITECTURES=120
ARG DEBIAN_FRONTEND=noninteractive

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    curl wget git \
    cmake ninja-build build-essential \
    libboost-program-options-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libopenimageio-dev \
    openimageio-tools \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libgmock-dev \
    libsqlite3-dev \
    libglew-dev \
    libcgal-dev \
    libceres-dev \
    libsuitesparse-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libblas-dev \
    liblapack-dev \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    && rm -rf /var/lib/apt/lists/*

# Fix Ubuntu's broken openimageio CMake config (missing OpenCV include dir)
RUN mkdir -p /usr/include/opencv4

# ── uv ────────────────────────────────────────────────────────────────────────
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# ── Build & install COLMAP (headless — no Qt/GUI needed) ─────────────────────
# pycolmap's CMakeLists.txt uses find_package(colmap REQUIRED), so COLMAP
# must be installed before uv sync builds pycolmap.
COPY colmap /workspace/colmap
WORKDIR /workspace/colmap
RUN mkdir -p build && cd build && \
    cmake .. -GNinja \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} \
        -DCMAKE_BUILD_TYPE=Release \
        -DCOLMAP_BUILD_WITH_GUI=OFF \
        -DCOLMAP_BUILD_TESTS=OFF \
    && ninja \
    && ninja install

# Ensure nvcc is on PATH for CUDA-extension builds (devel image sets /usr/local/cuda
# but an explicit ENV is more robust across layer boundaries).
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

# ── Python environment ─────────────────────────────────────────────────────────
COPY reconstruction /workspace/reconstruction
WORKDIR /workspace/reconstruction

# Tell torch extension builds which CUDA architectures to compile for.
# Covers Turing (7.5), Ampere (8.0/8.6), Ada (8.9), Hopper (9.0),
# Blackwell (10.0 / 12.0) — same generation as CUDA 13.
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0;10.0;12.0"

# Limit parallel ninja jobs during uv sync (avoids OOM on low-RAM builders).
# Override with --build-arg MAX_JOBS=N if you have more RAM.
ARG MAX_JOBS=4
ENV MAX_JOBS=${MAX_JOBS}

# uv reads .python-version (3.10.19) and pyproject.toml.
# pycolmap resolves to ../colmap which was already installed above.
RUN uv sync

# ── Runtime ───────────────────────────────────────────────────────────────────
ENV VIRTUAL_ENV=/workspace/reconstruction/.venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# Port 8080 for viser visualizer (pass -p 8080:8080 at docker run time)
EXPOSE 8080

CMD ["/bin/bash"]
