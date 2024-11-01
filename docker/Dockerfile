FROM ubuntu:22.04

# Set the environment variable to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Set the default Python version as an environment variable
ENV PYTHON_VERSION=3.10.10

# Install system dependencies and tools needed to build Python from source
RUN apt-get update --fix-missing && \
    apt-get install -y \
    wget \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libncurses5-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libffi-dev \
    liblzma-dev \
    tk-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    sudo \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Download and install Python from source using the specified version
RUN wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && \
    tar -xvf Python-${PYTHON_VERSION}.tgz && \
    cd Python-${PYTHON_VERSION} && \
    ./configure --enable-optimizations && \
    make -j $(nproc) && \
    make altinstall && \
    cd .. && \
    rm -rf Python-${PYTHON_VERSION} Python-${PYTHON_VERSION}.tgz

# Extract major.minor version from PYTHON_VERSION
RUN PYTHON_MAJOR_MINOR=$(echo $PYTHON_VERSION | cut -d. -f1,2) && \
    update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python$PYTHON_MAJOR_MINOR 1

# Upgrade pip and install Python dependencies in a single RUN to reduce image layers
RUN PYTHON_MAJOR_MINOR=$(echo $PYTHON_VERSION | cut -d. -f1,2) && \
    /usr/local/bin/python$PYTHON_MAJOR_MINOR -m ensurepip && \
    /usr/local/bin/python$PYTHON_MAJOR_MINOR -m pip install --upgrade pip && \
    /usr/local/bin/python$PYTHON_MAJOR_MINOR -m pip install python-dotenv requests websocket-client jupyterlab tavily-python  && \
    /usr/local/bin/python$PYTHON_MAJOR_MINOR -m pip install git+https://github.com/openai/swarm.git

# Set build arguments for USER_UID and USER_GID
ARG USER_UID=1000
ARG USER_GID=1000

# Create a user with the specified USER_UID and USER_GID
RUN groupadd -g 1000 quetzal && \
    useradd -u 1000 -g 1000 -ms /bin/bash quetzal

# Add quetzal to sudoers
RUN echo 'quetzal ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Change ownership of necessary directories to quetzal
RUN chown -R quetzal:quetzal /home/quetzal

# Switch to "quetzal"
USER quetzal
# Set environment variables
ENV OPENAI_BASE_URL=http://localhost:11434/v1
ENV OPENAI_MODEL_NAME=llama3
ENV OPENAI_API_KEY=1234
ENV TAVILY_API_KEY=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
