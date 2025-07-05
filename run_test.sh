export PYTHONFAULTHANDLER=1


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



python vllm_test.py
