FROM ubuntu:16.04

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    ca-certificates curl file build-essential \
    autoconf automake autotools-dev libtool xutils-dev 

ARG channel=stable
RUN curl https://sh.rustup.rs -sSf | sh -s -- --default-toolchain $channel -y

ENV PATH=/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin:/sbin

ADD ci/installs.sh /usr/local/bin/ci-installs
RUN chmod +x /usr/local/bin/ci-installs && \
    ci-installs && rm /usr/local/bin/ci-installs

RUN apt-get remove -y ca-certificates curl file build-essential \
    autoconf automake autotools-dev libtool xutils-dev && \
    rm -rf /var/lib/apt/lists/*

