# export VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1
# export VLLM_USE_V1=1
# export VLLM_USE_TRITON_FLASH_ATTN=1

import torch
import time
from contextlib import contextmanager
from amdsmi import (amdsmi_get_gpu_vram_usage,
                    amdsmi_get_processor_handles, amdsmi_init,
                    amdsmi_shut_down)
from vllm import LLM
import os
import socket

# Disable NCCL logs
os.environ["NCCL_DEBUG"] = "NONE"

def get_node_id():
    """Get a unique identifier for this node"""
    return socket.gethostname()

if __name__ == "__main__":
    # Create inference results file
    node_id = get_node_id()
    inference_file = f"inference_results_{node_id}.txt"

    @contextmanager
    def _rocm():
        try:
            amdsmi_init()
            yield
        finally:
            amdsmi_shut_down()

    def get_physical_device_indices(devices):
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible_devices is None:
            return devices

        visible_indices = [int(x) for x in visible_devices.split(",")]
        index_mapping = {i: physical for i, physical in enumerate(visible_indices)}
        return [index_mapping[i] for i in devices if i in index_mapping]

    def print_gpu_memory():
        with _rocm():
            devices = list(range(torch.cuda.device_count()))
            devices = get_physical_device_indices(devices)
            start_time = time.time()

            output: dict[int, str] = {}
            output_raw: dict[int, float] = {}
            for device in devices:
                dev_handle = amdsmi_get_processor_handles()[device]
                mem_info = amdsmi_get_gpu_vram_usage(dev_handle)
                gb_used = mem_info["vram_used"] / 2**10
                output_raw[device] = gb_used
                output[device] = f'{gb_used:.02f}'

            print('gpu memory used (GB): ', end='')
            for k, v in output.items():
                print(f'{k}={v}; ', end='')
            print('')

            dur_s = time.time() - start_time

            time.sleep(5)

    def print_memory_usage(stage):
        torch.cuda.synchronize()  # Ensure all operations are complete
        print(f"CUDA Memory Usage ({stage}):")
        print_gpu_memory()

    def run_inference(prompt):
        """Run inference and save results to file"""
        print(f"Running inference with prompt: {prompt}")

        # Write to both console and file
        with open(inference_file, 'a') as f:
            f.write(f"\n=== Inference at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            f.write(f"Node: {node_id}\n")
            f.write(f"Prompt: {prompt}\n")

            outputs = llm.generate(prompt)
            for output in outputs:
                prompt_text = output.prompt
                generated_text = output.outputs[0].text

                # Print to console
                print(f"Prompt: {prompt_text!r}, Generated text: {generated_text!r}")

                # Write to file
                f.write(f"Generated text: {generated_text}\n")
                f.write("-" * 50 + "\n")

    # model_list = [("Qwen/Qwen2.5-14B-Instruct", 0.7, 1)]
    model_list = [("Qwen/Qwen2.5-14B-Instruct", 0.7, 2)]
    # model_list = [("Qwen/Qwen2.5-14B-Instruct", 0.7, 8)]
    model, gpu_memory_utilization, tensor_parallel_size = model_list[0]
    print(f"Testing {model} with {tensor_parallel_size} tensor parallel size and {gpu_memory_utilization} GPU memory utilization")

    # Initialize inference results file
    with open(inference_file, 'w') as f:
        f.write(f"=== Inference Results for Node {node_id} ===\n")
        f.write(f"Model: {model}\n")
        f.write(f"Tensor Parallel Size: {tensor_parallel_size}\n")
        f.write(f"GPU Memory Utilization: {gpu_memory_utilization}\n")
        f.write(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n")

    # dowload model in parallel
    from huggingface_hub import snapshot_download
    snapshot_download(model, max_workers=32)

    llm = LLM(model=model, enable_sleep_mode=True,
          tensor_parallel_size=tensor_parallel_size, gpu_memory_utilization=gpu_memory_utilization,
          disable_custom_all_reduce=True,
          skip_tokenizer_init=False,
          max_model_len=8192,
          max_num_batched_tokens=8192,
          compilation_config={"level": 3, "cudagraph_capture_sizes": [1]}
          )

    print_memory_usage("initial")

    # First inference
    run_inference("San Francisco is")
    llm.sleep()

    time.sleep(10)
    print_memory_usage("after sleep")

    llm.wake_up()

    time.sleep(10)
    print_memory_usage("after wakeup")

    # Second inference
    run_inference("Paris is")

    print(f"Inference results saved to: {inference_file}")
