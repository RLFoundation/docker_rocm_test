# docker_rocm_test

## Issue: `[rank0]: torch.OutOfMemoryError: HIP out of memory. Tried to allocate 186.00 MiB. GPU 4 has a total capacity of 191.45 GiB of which 188.69 GiB is free. Of the allocated memory 0 bytes is allocated by PyTorch, and 2.00 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_HIP_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)`

## Re-implement this error:

### Docker
I've prepared the Dockerfile.rocm and the built image
- Dockerfile: `Dockerfile.rocm`
- Build image: `docker pull yushengsuthu/rocm-6.4-rc-debug:latest`

### Test case 
The `vllm_test.py` will excute the sleep mde within vllm

### Test Script 
You can run `run_test.sh` to re-implement the error. It will `pull the image`, `launch the image`, and then `excute the vllm_test.py`

