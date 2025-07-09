#!/bin/bash

docker run --rm -it \
  --device /dev/dri \
  --device /dev/kfd \
  -p 8263:8263 \
  --group-add video \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --privileged \
  -v $HOME/.ssh:/root/.ssh \
  -v $HOME:$HOME \
  --shm-size 128G \
  --name rocm6.4_rc_test \
  -w $PWD \
  yushengsuthu/rocm-6.4-rc-debug:latest \
  /bin/bash


export PYTHONFAULTHANDLER=1
export AMD_LOG_LEVEL=4
export HSAKMT_VERBOSE_LEVEL=7
export AMD_LOG_LEVEL_FILE=/home/yushensu/projects/docker_rocm_test/hip_log_ 


export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1

python vllm_test.py 2>&1 | tee test.log

# python vllm_test.py
