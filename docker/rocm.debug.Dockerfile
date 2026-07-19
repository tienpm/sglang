ARG BASE_IMAGE=lmsysorg/sglang:v0.5.15.post1-rocm720-mi35x
FROM ${BASE_IMAGE}

ARG SGLANG_SOURCE_COMMIT=unknown
ARG SETUPTOOLS_SCM_PRETEND_VERSION=0.5.15.post1

LABEL org.opencontainers.image.revision=${SGLANG_SOURCE_COMMIT}

RUN rm -rf /sgl-workspace/sglang
COPY . /sgl-workspace/sglang

WORKDIR /sgl-workspace/sglang

# Rebuild the ROCm extension and install the Python package from this checkout
# while retaining the base image's pinned ROCm, Torch, Triton, and AITER stack.
RUN python -m pip uninstall -y sgl_kernel sglang \
    && cd sgl-kernel \
    && rm -f pyproject.toml \
    && cp pyproject_rocm.toml pyproject.toml \
    && AMDGPU_TARGET=gfx950 python setup_rocm.py install \
    && cd .. \
    && rm -f python/pyproject.toml \
    && cp python/pyproject_other.toml python/pyproject.toml \
    && SETUPTOOLS_SCM_PRETEND_VERSION=${SETUPTOOLS_SCM_PRETEND_VERSION} \
       python -m pip install --no-deps -e python \
    && python -m pip cache purge

CMD ["/bin/bash"]
