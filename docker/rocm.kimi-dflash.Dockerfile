ARG BASE_IMAGE=sglang-v0.5.15.15.post1
FROM ${BASE_IMAGE}

# The base ROCm image enables its fused MLA decode adapter by default. It is
# incompatible with an explicitly selected Triton attention backend.
ENV SGLANG_USE_AITER=0 \
    SGLANG_ROCM_FUSED_DECODE_MLA=0

EXPOSE 30000

ENTRYPOINT ["sglang", "serve"]
CMD ["--model-path", "/opt/models/moonshotai/Kimi-K2.7-Code", \
     "--trust-remote-code", \
     "--attention-backend", "triton", \
     "--linear-attn-prefill-backend", "triton", \
     "--linear-attn-decode-backend", "triton", \
     "--mamba-radix-cache-strategy", "no_buffer", \
     "--disable-overlap-schedule", \
     "--tp-size", "4", \
     "--max-running-requests", "32", \
     "--disable-cuda-graph", \
     "--flashinfer-allreduce-fusion-backend", "auto", \
     "--mem-fraction-static", "0.8", \
     "--host", "0.0.0.0", \
     "--port", "30000", \
     "--speculative-algorithm", "DFLASH", \
     "--speculative-draft-model-path", "/opt/models/nvidia/Kimi-K2.7-Code-DFlash", \
     "--speculative-dflash-block-size", "8", \
     "--speculative-draft-attention-backend", "triton"]
