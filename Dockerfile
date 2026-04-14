# ==========================================
# STAGE 1: Build the Rust GUI
# ==========================================
FROM rust:slim-bullseye as builder

RUN apt-get update && apt-get install -y \
    pkg-config libssl-dev libfontconfig1-dev libxcb1-dev libx11-dev g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY launcher/ ./launcher/
WORKDIR /build/launcher
RUN cargo build --release

# ==========================================
# STAGE 2: The Runtime Environment
# ==========================================
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

# 1. SYSTEM LIBS
# Added 'x11-utils' to check display permissions
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3.10-dev git \
    libfontconfig1 libxcb1 libx11-6 libgl1-mesa-glx libglib2.0-0 \
    libxcursor1 libxi6 libxrandr2 libxkbcommon-x11-0 \
    tini x11-utils \
    && rm -rf /var/lib/apt/lists/*

# 2. PYTHON SETUP
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# 3. PIP INSTALLS (Cached)
RUN python -m pip install --upgrade pip
RUN python -m pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
RUN python -m pip install --no-deps "xformers<0.0.26" "trl<0.9.0" peft accelerate bitsandbytes pandas datasets scipy

# 4. COPY CODE
WORKDIR /app
COPY engine/ ./engine/
COPY tools/ ./tools/
COPY configs/ ./configs/
RUN mkdir -p outputs

# 5. COPY BINARY
COPY --from=builder /build/launcher/target/release/launcher ./UnslothStudio

# 6. CONFIG
RUN echo '{"python_path": "/usr/bin/python3"}' > settings.json

# 7. CREATE ENTRYPOINT SCRIPT (THE SMART CHECK)
# This script runs inside Docker every time it starts.
RUN echo '#!/bin/bash\n\
\n\
# Check if we can talk to the X Server\n\
if ! timeout 1s xset q >/dev/null 2>&1; then\n\
    echo -e "\\033[0;31m"\n\
    echo "======================================================="\n\
    echo " ❌ ERROR: GUI PERMISSION DENIED"\n\
    echo "======================================================="\n\
    echo " Docker cannot draw the window on your screen."\n\
    echo " "\n\
    echo " Please run this command on your HOST terminal:"\n\
    echo " "\n\
    echo "     xhost +local:root"\n\
    echo " "\n\
    echo " Then try running docker compose up again."\n\
    echo "======================================================="\\033[0m\n\
    exit 1\n\
fi\n\
\n\
# If check passes, launch the app\n\
exec ./UnslothStudio\n\
' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# 8. RUN
ENTRYPOINT ["/usr/bin/tini", "--", "/app/entrypoint.sh"]