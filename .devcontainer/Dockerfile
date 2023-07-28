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
    libgl1-mesa-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/
