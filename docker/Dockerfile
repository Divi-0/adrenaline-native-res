# nativeres-sdk: vitasdk + pspdev + Rust, everything the build needs (scripts/build.sh).
#
# Rust nightly is pinned because the Vita is a tier-3 target: its core library
# must be compiled from source with `-Z build-std`, which only nightly allows.
# Our own code is plain stable Rust; PC-side tests use the stable toolchain.
FROM vitasdk/vitasdk:latest

RUN wget https://github.com/pspdev/pspdev/releases/latest/download/pspdev-ubuntu-latest-x86_64.tar.gz && \
    mkdir -p /usr/local/pspdev && \
    tar -xzf pspdev-ubuntu-latest-x86_64.tar.gz -C /usr/local/ && \
    rm pspdev-ubuntu-latest-x86_64.tar.gz

ENV PSPDEV=/usr/local/pspdev
ENV PATH=${PSPDEV}/bin:$PATH

RUN echo "export PSPDEV=${PSPDEV}" > /etc/profile.d/vitasdk.sh && \
    echo 'export PATH=$PSPDEV/bin:$PATH'  >> /etc/profile.d/vitasdk.sh

ARG RUST_NIGHTLY=nightly-2026-10-01
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH
RUN (command -v apt-get >/dev/null && apt-get update && apt-get install -y --no-install-recommends curl ca-certificates gcc libc6-dev || \
     apk add --no-cache curl ca-certificates gcc musl-dev) && \
    curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable && \
    rustup toolchain install ${RUST_NIGHTLY} --profile minimal --component rust-src && \
    echo 'export PATH=/usr/local/cargo/bin:$PATH' >> /etc/profile.d/vitasdk.sh && \
    chmod -R a+w /usr/local/cargo /usr/local/rustup
