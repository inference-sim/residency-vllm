ARG VLLM_BASE_VERSION=v0.23.0
FROM vllm/vllm-openai:${VLLM_BASE_VERSION}

# prometheus_client is already in the base image — no extra pip install needed

# Find where vLLM is installed
RUN VLLM_LOCATION=$(python3 -c "import vllm; import os; print(os.path.dirname(vllm.__file__))") && \
    echo "$VLLM_LOCATION" > /tmp/vllm_location.txt

# Copy only the patched files
COPY vllm/ /tmp/vllm-patch/

# Overlay patched files onto the installed package.
# cp -r merges directories — only the files we provide get replaced,
# all other files (including compiled .so extensions) stay from the base.
RUN VLLM_LOCATION=$(cat /tmp/vllm_location.txt) && \
    cp -r /tmp/vllm-patch/* "$VLLM_LOCATION/" && \
    rm -rf /tmp/vllm-patch

# Create prometheus multiprocess directory (shared between API server + EngineCore)
RUN mkdir -p /tmp/prometheus_multiproc

ENTRYPOINT ["vllm", "serve"]
