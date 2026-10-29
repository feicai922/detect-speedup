import subprocess

import pathlib as _pl, sys as _sys
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # repo root
import re

def get_all_gpu_info():
    # 调用 nvidia-smi 输出XML格式信息，便于解析
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.free,memory.used",
        "--format=csv,noheader,nounits"
    ]
    result = subprocess.check_output(cmd, encoding="utf-8")
    lines = result.strip().splitlines()
    gpu_list = []
    for line in lines:
        idx, name, total_mem, free_mem, used_mem = [x.strip() for x in line.split(",")]
        gpu_list.append({
            "cuda_id": int(idx),
            "gpu_model": name,
            "total_memory_MiB": int(total_mem),
            "free_memory_MiB": int(free_mem),
            "used_memory_MiB": int(used_mem)
        })
    return gpu_list

if __name__ == "__main__":
    try:
        gpus = get_all_gpu_info()
        print("=" * 80)
        print(f"检测到可用 GPU 总数: {len(gpus)}")
        print("=" * 80)
        for gpu in gpus:
            print(f"CUDA ID: {gpu['cuda_id']}")
            print(f"显卡型号: {gpu['gpu_model']}")
            print(f"总显存:   {gpu['total_memory_MiB']} MiB ({gpu['total_memory_MiB']/1024:.2f} GB)")
            print(f"已用显存: {gpu['used_memory_MiB']} MiB ({gpu['used_memory_MiB']/1024:.2f} GB)")
            print(f"空闲显存: {gpu['free_memory_MiB']} MiB ({gpu['free_memory_MiB']/1024:.2f} GB)")
            print("-" * 50)
    except Exception as e:
        print(f"获取GPU信息失败: {e}")
        print("确认当前环境已安装 NVIDIA 驱动、nvidia-smi 可用")