FROM mcr.microsoft.com/devcontainers/python:0-3.11
RUN apt-get update \
    && apt-get install -y wget \
    bash \
    git \
    gcc \
    g++ \
    gfortran \
    liblapack-dev \
    libamd2 \
    libcholmod3 \
    libmetis-dev \
    libsuitesparse-dev \
    libnauty2-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && wget -nH https://raw.githubusercontent.com/coin-or/coinbrew/master/coinbrew \
    && chmod u+x coinbrew \
    && bash coinbrew fetch Cbc@master \
    && bash coinbrew build Cbc@master --no-prompt --prefix=/usr/local --tests=none --enable-cbc-parallel
ENV PMIP_CBC_LIBRARY="/usr/local/lib/libCbc.so"
ENV LD_LIBRARY_PATH="/home/haroldo/prog/lib"