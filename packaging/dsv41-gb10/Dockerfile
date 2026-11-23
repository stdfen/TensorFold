# syntax=docker/dockerfile:1.7

ARG CUDA_VERSION=13.0.1
ARG UBUNTU_VERSION=24.04
ARG CUDA_DEVEL_IMAGE=nvcr.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}@sha256:7d2f6a8c2071d911524f95061a0db363e24d27aa51ec831fcccf9e76eb72bc92
ARG CUDA_RUNTIME_IMAGE=nvcr.io/nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}@sha256:c3fde347d52d578c84fd644bc177bc7ec333feaf11550d990da4084d7612e4c7

FROM ${CUDA_DEVEL_IMAGE} AS toolchain

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates curl minisign xz-utils \
    && rm -rf /var/lib/apt/lists/*

ARG ZIG_VERSION=0.17.0
ARG ZIG_MINISIGN_KEY=RWSGOq2NVecA2UPNdBUZykf1CCb147pkmdtYxgb3Ti+JO/wCYvhbAb/U
ARG ZIG_SHA256=9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8
ARG ZIG_MIRROR=https://pkg.hexops.org/zig

RUN set -eux; \
    arch="$(dpkg --print-architecture)"; \
    test "$arch" = arm64; \
    archive="zig-aarch64-linux-${ZIG_VERSION}.tar.xz"; \
    curl --fail --location --retry 5 --output "/tmp/${archive}" \
        "${ZIG_MIRROR}/${archive}"; \
    curl --fail --location --retry 5 --output "/tmp/${archive}.minisig" \
        "https://ziglang.org/download/${ZIG_VERSION}/${archive}.minisig"; \
    echo "${ZIG_SHA256}  /tmp/${archive}" | sha256sum --check --strict; \
    minisign -Vm "/tmp/${archive}" -P "${ZIG_MINISIGN_KEY}"; \
    mkdir -p /opt/zig; \
    tar --extract --xz --file "/tmp/${archive}" --strip-components=1 --directory /opt/zig; \
    rm -f "/tmp/${archive}" "/tmp/${archive}.minisig"

ENV PATH=/opt/zig:/usr/local/cuda/bin:${PATH}

RUN test "$(zig version)" = "${ZIG_VERSION}" \
    && nvcc --version

FROM toolchain AS builder

ARG BUILD_JOBS=2
WORKDIR /src
COPY . .

RUN zig build dsv41-assets dsv41-engram-pack test-dsv41-assets-host test-dsv41-token-map test-dsv41-shared -j"${BUILD_JOBS}"
RUN zig build dsv41-serve \
        -Doptimize=ReleaseSafe \
        -Dnvcc=/usr/local/cuda/bin/nvcc \
        -Ddsv41-nvcc=/usr/local/cuda/bin/nvcc \
        -Dsm=121 \
        -j"${BUILD_JOBS}"

# The served DSV41 engine loads the TP mailbox/RoCE image from
# TF_TP_KERNEL_IMAGE. Keep it from the same source revision and CUDA
# toolchain as the host binary; an older recipe image is not ABI evidence.
RUN install -d /src/zig-out/share \
    && nvcc -fatbin -O3 -std=c++17 \
        -gencode=arch=compute_120,code=sm_120 \
        -gencode=arch=compute_121,code=sm_121 \
        -gencode=arch=compute_90,code=compute_90 \
        -o /src/zig-out/share/tp_mailbox.fatbin \
        /src/zig/src/fabric/tp/mailbox.cu

FROM ${CUDA_RUNTIME_IMAGE} AS runtime

ARG TENSORFOLD_VERSION=1.2.1
ARG TENSORFOLD_COMMIT=1db3edbe7e692e83c7f5056754dc4e69e79a6c22

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates curl ibverbs-providers libibverbs1 librdmacm1 \
    && rm -rf /var/lib/apt/lists/* \
    && ldconfig -p | grep -q 'libnccl.so.2' \
    && groupadd --gid 10001 tensorfold \
    && useradd --uid 10001 --gid tensorfold --create-home --shell /usr/sbin/nologin tensorfold \
    && install -d -o tensorfold -g tensorfold /var/lib/tensorfold

COPY --from=builder /src/zig-out/bin/tensorfold-dsv41 /usr/local/bin/tensorfold-dsv41
COPY --from=builder /src/zig-out/bin/tf-dsv41 /usr/local/bin/tf-dsv41
COPY --from=builder /src/zig-out/bin/tf-dsv41-assets /usr/local/bin/tf-dsv41-assets
COPY --from=builder /src/zig-out/bin/tf-dsv41-token-map /usr/local/bin/tf-dsv41-token-map
COPY --from=builder /src/zig-out/bin/tf-dsv41-engram-pack /usr/local/bin/tf-dsv41-engram-pack
COPY --from=builder /src/zig-out/share/tp_mailbox.fatbin /opt/tensorfold/share/tp_mailbox.fatbin
COPY packaging/dsv41-gb10/entrypoint.sh /usr/local/bin/tensorfold-entrypoint
COPY packaging/dsv41-gb10/healthcheck.sh /usr/local/bin/tensorfold-healthcheck
COPY packaging/dsv41-gb10/preflight.sh /usr/local/bin/tensorfold-preflight

RUN chmod 0755 \
        /usr/local/bin/tensorfold-dsv41 \
        /usr/local/bin/tf-dsv41 \
        /usr/local/bin/tf-dsv41-assets \
        /usr/local/bin/tf-dsv41-token-map \
        /usr/local/bin/tf-dsv41-engram-pack \
        /usr/local/bin/tensorfold-entrypoint \
        /usr/local/bin/tensorfold-healthcheck \
        /usr/local/bin/tensorfold-preflight \
    && printf '%s\n' \
        "tensorfold_version=${TENSORFOLD_VERSION}" \
        "tensorfold_commit=${TENSORFOLD_COMMIT}" \
        "cuda=13.0.1" \
        "sm=121" \
        "zig=0.17.0" \
        > /opt/tensorfold-build-info

LABEL org.opencontainers.image.title="TensorFold DeepSeek-V4.1 GB10 TP2" \
      org.opencontainers.image.version="${TENSORFOLD_VERSION}" \
      org.opencontainers.image.revision="${TENSORFOLD_COMMIT}" \
      org.opencontainers.image.source="https://github.com/stdfen/TensorFold" \
      com.tensorfold.cuda="13.0.1" \
      com.tensorfold.cuda.sm="121"

ENV TF_MODEL_DIR=/models/model \
    TF_DSV41_ASSETS=/assets \
    TF_DSV41_ENGRAM_DIR=/engram \
    TF_EXPERIMENTAL=deepseek \
    TF_TP_WORLD=2 \
    TF_TP_PORT=29500 \
    TF_TP_DEVICE=0 \
    TF_TP_FAILFAST=1 \
    TF_COMM_BACKEND=nccl \
    TF_HTTP_HOST=0.0.0.0 \
    TF_HTTP_PORT=8080 \
    TF_CONTEXT=32768 \
    TF_PARALLEL=1 \
    TF_DSV41_DRAFTS=0 \
    TF_REQUIRE_ASSET_HASHES=1

VOLUME ["/models", "/assets", "/engram", "/var/lib/tensorfold"]
EXPOSE 8080 29500 29507
STOPSIGNAL SIGTERM
HEALTHCHECK --interval=30s --timeout=10s --start-period=20m --retries=3 CMD ["tensorfold-healthcheck"]
ENTRYPOINT ["tensorfold-entrypoint"]

