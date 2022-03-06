#!/usr/bin/env bash

# run to install on TPU VM

sudo apt update

sudo apt install -y \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev

mkdir -p ~/.mujoco

wget https://www.roboti.us/download/mujoco200_linux.zip -O ~/mujoco.zip \
    && unzip ~/mujoco.zip -d ~/.mujoco \
    && rm ~/mujoco.zip
wget https://www.roboti.us/download/mjpro150_linux.zip -O ~/mujoco.zip \
    && unzip ~/mujoco.zip -d ~/.mujoco \
    && rm ~/mujoco.zip
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O ~/mujoco.tar.gz \
    && tar -xf ~/mujoco.tar.gz -C ~/.mujoco \
    && rm ~/mujoco.tar.gz
wget https://github.com/deepmind/mujoco/releases/download/2.1.1/mujoco-2.1.1-linux-x86_64.tar.gz -O ~/mujoco.tar.gz \
    && tar -xf ~/mujoco.tar.gz -C ~/.mujoco \
    && rm ~/mujoco.tar.gz


cp ./gcp/vendor/mjkey.txt ~/.mujoco/mjkey.txt
ln -s ~/.mujoco/mujoco200_linux ~/.mujoco/mujoco200

echo 'export LD_LIBRARY_PATH=~/.mujoco/mjpro150/bin:${LD_LIBRARY_PATH}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=~/.mujoco/mujoco200/bin:${LD_LIBRARY_PATH}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=~/.mujoco/mujoco200_linux/bin:${LD_LIBRARY_PATH}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=~/.mujoco/mujoco-2.1.1/bin:${LD_LIBRARY_PATH}' >> ~/.bashrc
export LD_LIBRARY_PATH=~/.mujoco/mjpro150/bin:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=~/.mujoco/mujoco200/bin:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=~/.mujoco/mujoco200_linux/bin:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=~/.mujoco/mujoco-2.1.1/bin:${LD_LIBRARY_PATH}

wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
    && bash ~/miniconda.sh -b -p ~/miniconda3 \
    && rm ~/miniconda.sh

export PATH="~/miniconda3/bin:$PATH"
source ~/miniconda3/etc/profile.d/conda.sh
conda init bash

conda update -y --name base conda && conda clean --all -y
conda env create -f ./environment.yml
echo 'conda activate rjax' >> ~/.bashrc

conda activate rjax \
    && pip install --upgrade "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html \
    && pip install -e .
