# ubuntu:20.04
ARG IMAGE_BASE=ubuntu:20.04
FROM ${IMAGE_BASE}

ENV DEBIAN_FRONTEND=noninteractive

# install common tools
RUN apt-get update && \
    apt-get install -y git vim wget time \
                        build-essential \
                        gcc \
                        g++ \
                        cmake \
                        ninja-build \
                        clang-10 \
                        python3 \
                        python3-pip \
                        python3-dev \
                        libprotobuf-dev  \
                        protobuf-compiler \
                        libssl-dev  \
                        libgmp3-dev  \
                        libtool \
                        libomp5 \
                        libomp-dev \
                        libntl-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip3 install torch==2.0.1 && \
    pip3 install onnx==1.14.1 && \
    pip3 install onnxruntime==1.15.1 && \
    pip3 install matplotlib==3.7.5 && \
    pip3 install numpy==1.24.4

WORKDIR /app/cifar

RUN wget -qO- https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz | tar xzv --strip-components=1 && \
    wget -qO- https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz | tar xzv --strip-components=1


WORKDIR /app

# copy soucecode & scipt
COPY ./ .

ENTRYPOINT ["/app/scripts/entrypoint.sh"]

