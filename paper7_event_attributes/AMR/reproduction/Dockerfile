# 论文七 AMR 独立环境：Python 2.7 + Java 8，不与 Python 3 的 LLM 环境混装。
# 基础镜像及大文件资源的准备命令见 ../README.md。
# The historical CAMR code is Python 2 and JAMR's sbt build requires Java 8.
# The official Miniconda2 installer is placed in this build context by setup.
# Only the runtime changes; the author source is kept outside the image.
FROM amr-ubuntu-base:local AS apt-base

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/conda/bin:$PATH \
    JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONIOENCODING=utf-8

RUN sed -i \
        -e 's@http://archive.ubuntu.com/ubuntu/@http://mirrors.aliyun.com/ubuntu/@g' \
        -e 's@http://security.ubuntu.com/ubuntu/@http://mirrors.aliyun.com/ubuntu/@g' \
        /etc/apt/sources.list.d/ubuntu.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        openjdk-8-jdk-headless build-essential swig git wget unzip \
        ca-certificates tar gzip \
    && rm -rf /var/lib/apt/lists/*

FROM apt-base AS runtime

COPY Miniconda2-latest-Linux-x86_64.sh /tmp/miniconda2.sh
RUN echo 'b820dde1a0ba868c4c948fe6ace7300a252b33b5befd078a15d4a017476b8979  /tmp/miniconda2.sh' | sha256sum -c - \
    && bash /tmp/miniconda2.sh -b -p /opt/conda \
    && rm /tmp/miniconda2.sh

RUN ln -sf /usr/bin/ld /opt/conda/compiler_compat/ld \
    && python -m pip install --no-cache-dir \
    --index-url https://mirrors.aliyun.com/pypi/simple/ \
    'numpy==1.16.6' 'nltk==3.4.5' 'six==1.16.0' \
    'bllipparser==2016.9.11'

ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/opt/conda/lib

WORKDIR /work
