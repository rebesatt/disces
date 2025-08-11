# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    MPLCONFIGDIR=/tmp/mplconfig \
    PIP_NO_CACHE_DIR=1

# System deps: LaTeX toolchain, fonts; git no longer needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    make \
    latexmk \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-science \
    texlive-luatex \
    ghostscript \
    fonts-dejavu \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps first (better layer caching)
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

# Then copy the rest of the repo
COPY . /app

# Default command
CMD ["python", "reproduce_papyer.py"]
