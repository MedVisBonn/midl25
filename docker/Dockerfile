FROM nvcr.io/nvidia/pytorch:24.10-py3

### install standard capabilities
RUN apt-get update && apt-get install -y \
    git \
    vim \
    binutils \
    tmux

### install pip packages 
RUN pip install \
    wandb==0.18.5 \
    scikit-learn==1.5.2 \
    torchmetrics==1.6.1 \
    lightning==2.4.0 \
    monai==1.4.0 \
    tqdm==4.66.5 \
    seaborn==0.13.2 \
    omegaconf==2.3.0 \
    batchgenerators==0.25.1 \
    nibabel==5.3.2 \
    hydra-core==1.3.2 \
    threadpoolctl==3.5.0 \
    pandas==2.2.2 \
    torchvision==0.20.0a0 \
    matplotlib==3.9.2 \
    numpy==1.24.4 \
    cupy-cuda12x==12.3.0

## GIT setup
WORKDIR /root/workspace
RUN mv * nvidia || mkdir nvidia
WORKDIR /root/workspace/repos
RUN git clone https://github.com/MedVisBonn/midl25.git
RUN git clone https://github.com/deepmind/surface-distance.git
RUN pip install surface-distance/

## Cleanup
RUN apt-get autoremove -y && rm -rf /tmp/* /var/tmp/* && apt-get clean && rm -rf /var/lib/apt/lists/*

### expose ports
EXPOSE 22
EXPOSE 42421
EXPOSE 6006
EXPOSE 8889

### start interactive console on attach
ENV DEBIAN_FRONTEND=teletype
