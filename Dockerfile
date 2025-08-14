# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    MPLCONFIGDIR=/tmp/mplconfig \
    PIP_NO_CACHE_DIR=1

# System deps: LaTeX stack (slim but safe), fonts, build helpers
RUN apt-get update && apt-get install -y --no-install-recommends \
    make \
    latexmk \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-fonts-extra \
    texlive-science \
    texlive-luatex \
    texlive-pictures \
    texlive-bibtex-extra \
    texlive-latex-recommended \
    texlive-humanities \
    texlive-plain-generic \
    ghostscript \
    fonts-dejavu \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps from the repo's requirements.txt first (better caching)
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

# Copy the rest of the repo
COPY . /app

# (Optional) fail fast if the script is missing
RUN test -f reproduce_paper.py || (echo "reproduce_paper.py not found in /app" && ls -la && exit 1)

# # Default command: run the paper reproduction script
# CMD ["python", "reproduce_paper.py"]

# Always run the script; compose `command:` will append flags (e.g., ["-ilm"])
ENTRYPOINT ["python", "-u", "reproduce_paper.py"]
CMD []
