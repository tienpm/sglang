FROM lmsysorg/sglang:v0.5.14-rocm720-mi35x

ENV SGLANG_ENABLE_OVERLAP_PLAN_STREAM=0
ENV MODEL_PATH=/models/Qwen3.5-397B-A17B
ENV DRAFT_MODEL_PATH=/models/Qwen3.5-397B-A17B-DFlash
ENV TP_SIZE=8
ENV MEM_FRACTION_STATIC=0.8
ENV SPECULATIVE_DRAFT_ATTENTION_BACKEND=triton

EXPOSE 30000

CMD ["sh", "-c", "exec python -m sglang.launch_server \
    --model-path \"$MODEL_PATH\" \
    --trust-remote-code \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path \"$DRAFT_MODEL_PATH\" \
    --speculative-dflash-block-size 8 \
    --speculative-draft-attention-backend \"$SPECULATIVE_DRAFT_ATTENTION_BACKEND\" \
    --attention-backend triton \
    --linear-attn-prefill-backend triton \
    --linear-attn-decode-backend triton \
    --mamba-radix-cache-strategy no_buffer \
    --disable-overlap-schedule \
    --tp-size \"$TP_SIZE\" \
    --max-running-requests 32 \
    --cuda-graph-max-bs-decode 32 \
    --cuda-graph-backend-prefill tc_piecewise \
    --flashinfer-allreduce-fusion-backend auto \
    --mem-fraction-static \"$MEM_FRACTION_STATIC\" \
    --host 0.0.0.0"]
