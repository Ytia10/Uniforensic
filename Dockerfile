# UniForensic Workbench - container image with every N1-N5 dependency.
# Ubuntu 24.04 gives Python 3.12.3 and SQLite 3.45.1: the exact pair the N1-N5 runs and
# the workbench were verified with (SQLite writes the simulated phones' databases).
# The project folder itself is NOT copied in: docker-compose.yml mounts it at /work,
# so the app reads your own N1-N5 outputs and writes only to app_data/.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3.12 python3.12-venv ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && python3.12 -m venv /opt/venv

WORKDIR /opt/requirements
COPY requirements.txt requirements_n4.txt requirements_n5.txt requirements_app.txt ./
# CPU build of torch first (the default build pulls ~3 GB of CUDA libraries), then the rest.
RUN pip install --upgrade pip \
 && pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements_app.txt \
 && pip check

WORKDIR /work
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "ufapp.main:app", "--host", "0.0.0.0", "--port", "8000"]
