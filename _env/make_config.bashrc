#!/bin/bash


if [ -f ~/project_config.bashrc ] ; then
    source ~/project_config.bashrc
    source ~/machine_config.bashrc
elif [ -f ./_env/project_config.bashrc ] ; then
    source ./_env/project_config.bashrc
    source ./_env/machine_config.bashrc
fi


TMP_TORCHEXT=$(mktemp -d -t ci-XXXXXXXXXX)
# TMP_DNNLIB=$(mktemp -d -t ci-XXXXXXXXXX)
TMP_NUMBA_CACHE_DIR=$(mktemp -d -t ci-XXXXXXXXXX)

export OPTIONS_DOCKER_RUN="
    --rm
    --gpus=all
    --ipc=host
    --net=host
    -w $PROJECT_DN
    -e DISPLAY=$DISPLAY
    -e NUMBA_CACHE_DIR=$TMP_NUMBA_CACHE_DIR
    -v /dev/shm:/dev/shms
    -v $PROJECT_DN:$PROJECT_DN
    -v $PROJECT_DN/_env/project_config.bashrc:/root/project_config.bashrc
    -v $PROJECT_DN/_env/machine_config.bashrc:/root/machine_config.bashrc
    -v $PROJECT_DN/_env/make_config.bashrc:/root/make_config.bashrc
    -v $PROJECT_DN/_env/home/bin:/root/bin
    -v $PROJECT_DN/_env/home/.bashrc:/root/.bashrc
    -v $PROJECT_DN/_env/home/.sensitive:/root/.sensitive
    -v $PROJECT_DN/_env/home/.jupyter:/root/.jupyter
    -v $PROJECT_DN/_env/home/.config/matplotlib:/root/.config/matplotlib
    -v $PROJECT_DN/_env/home/.config/wandb:/root/.config/wandb
    -v $PROJECT_DN/_env/home/.cache/matplotlib:/root/.cache/matplotlib
    -v $PROJECT_DN/_env/home/.cache/torch:/root/.cache/torch
    -v $PROJECT_DN/_env/home/.cache/clip:/root/.cache/clip
    -v $PROJECT_DN/_env/home/.cache/dnnlib:/root/.cache/dnnlib
    -v $PROJECT_DN/_env/home/.torch:/root/.torch
    -v $PROJECT_DN/_env/home/.cupy:/root/.cupy
    -v $TMP_TORCHEXT:/root/.cache/torch_extensions
    -v $TMP_NUMBA_CACHE_DIR:$TMP_NUMBA_CACHE_DIR
"
    # -v $TMP_DNNLIB:/root/.cache/dnnlib

export OPTIONS_SINGULARITY_EXEC="
    --nv
    --no-home
    --containall
    --home /root
    -W $PROJECT_DN
    --env DISPLAY=$DISPLAY
    --env NUMBA_CACHE_DIR=$TMP_NUMBA_CACHE_DIR
    -B /tmp:/tmp
    -B /var/tmp:/var/tmp
    -B $PROJECT_DN:$PROJECT_DN
    -B $PROJECT_DN/_env/project_config.bashrc:/root/project_config.bashrc
    -B $PROJECT_DN/_env/machine_config.bashrc:/root/machine_config.bashrc
    -B $PROJECT_DN/_env/make_config.bashrc:/root/make_config.bashrc
    -B $PROJECT_DN/_env/home/bin:/root/bin
    -B $PROJECT_DN/_env/home/.bashrc:/root/.bashrc
    -B $PROJECT_DN/_env/home/.sensitive:/root/.sensitive
    -B $PROJECT_DN/_env/home/.jupyter:/root/.jupyter
    -B $PROJECT_DN/_env/home/.config/matplotlib:/root/.config/matplotlib
    -B $PROJECT_DN/_env/home/.config/wandb:/root/.config/wandb
    -B $PROJECT_DN/_env/home/.cache/matplotlib:/root/.cache/matplotlib
    -B $PROJECT_DN/_env/home/.cache/torch:/root/.cache/torch
    -B $PROJECT_DN/_env/home/.cache/clip:/root/.cache/clip
    -B $PROJECT_DN/_env/home/.cache/dnnlib:/root/.cache/dnnlib
    -B $PROJECT_DN/_env/home/.torch:/root/.torch
    -B $PROJECT_DN/_env/home/.cupy:/root/.cupy
    -B $TMP_TORCHEXT:/root/.cache/torch_extensions
    -B $TMP_NUMBA_CACHE_DIR:$TMP_NUMBA_CACHE_DIR
"
    # -B $TMP_DNNLIB:/root/.cache/dnnlib








