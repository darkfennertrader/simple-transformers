FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

RUN apt update && \
    apt install -y bash \
    build-essential \
    git \
    curl \
    ca-certificates \
    python3 \
    python3-pip && \
    rm -rf /var/lib/apt/lists

# create soft link
RUN cd /usr/bin && ln -s python3.6 python

RUN python3 -m pip install --no-cache-dir --upgrade pip

RUN pip install notebook

RUN pip install torch==1.7.0+cu101 torchvision==0.8.1+cu101 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install simpletransformers --use-feature=2020-resolver

WORKDIR /simple-transformers

# install additional requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root"]