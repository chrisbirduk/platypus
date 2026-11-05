# Platypus GPU runner for a cloud machine with an NVIDIA GPU.
#
# Build from the repository root:
#   docker build -f docker/Dockerfile -t platypus:cg26p .
# Run (mount runs/ and pile_cache/ so results and the data cache survive):
#   docker run --gpus all \
#     -v $PWD/runs:/work/runs -v $PWD/pile_cache:/work/pile_cache \
#     platypus:cg26p platypus configs/presets/cg26p.env 1
#
# No secret is baked into the image. Pass any token at run time
# (docker run -e HF_TOKEN=...), from the machine's secret store.

# Pin the base image to an exact tag when the first cloud run is set up.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PILE_CACHE=/work/pile_cache \
    RUNS_DIR=/work/runs \
    FLA_AUTO_INSTALL=0

WORKDIR /work
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

COPY env/requirements-gpu.txt /tmp/requirements-gpu.txt
# torch comes with the base image; install the rest.
RUN grep -v '^torch' /tmp/requirements-gpu.txt | pip install -r /dev/stdin

COPY . /work/

ENTRYPOINT ["bash", "/work/docker/entrypoint.sh"]
