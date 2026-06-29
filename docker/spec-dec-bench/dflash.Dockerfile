FROM lmsysorg/sglang:v0.5.14-rocm72

ENV SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
ENV MODEL_PATH=/models/Qwen3.5-397B-A17B
ENV TP_SIZE=8

EXPOSE 30000

CMD exec python -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --trust-remote-code \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path modal-labs/Qwen3.5-397B-A17B-DFlash \
    --speculative-dflash-block-size 8 \
    --speculative-draft-attention-backend fa4 \
    --attention-backend trtllm_mha \
    --linear-attn-prefill-backend triton \
    --linear-attn-decode-backend flashinfer \
    --mamba-scheduler-strategy extra_buffer \
    --tp-size "${TP_SIZE}" \
    --max-running-requests 32 \
    --cuda-graph-max-bs-decode 32 \
    --cuda-graph-backend-prefill tc_piecewise \
    --enable-flashinfer-allreduce-fusion \
    --mem-fraction-static 0.8 \
    --host 0.0.0.0
