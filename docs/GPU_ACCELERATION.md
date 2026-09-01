# GPU加速配置

## 当前电脑

当前项目配置为CPU：

```yaml
models:
  embedding_device: cpu
  reranker_device: cpu
asr:
  providers:
    qwen3_asr:
      device: cpu
      dtype: bfloat16
```

当前PyTorch是CPU构建。Intel Arc 130T可供独立llama-server通过Vulkan运行GGUF，但当前Python ASR、BGE和Reranker没有配置Intel XPU运行时，不能仅把`device`改成`cuda`。

## NVIDIA CUDA

1. 安装与显卡驱动和目标CUDA版本匹配的官方PyTorch CUDA wheel。
2. 验证：

```powershell
.\venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

3. 修改`config.yaml`：

```yaml
models:
  embedding_device: cuda
  reranker_device: cuda
asr:
  providers:
    qwen3_asr:
      device: cuda
      dtype: bfloat16
```

较老、没有良好BF16支持的NVIDIA卡可改用`float16`。显存不足时先保持BGE和Reranker为CPU，只将Qwen3-ASR/ForcedAligner设为CUDA。项目入库时仍会先卸载问答模型，再运行ASR，完成后恢复问答模型。

llama-server的GPU卸载由`config.yaml`中的`llm.gpu_layers`控制；它不读取上述PyTorch设备配置。Intel/AMD使用Vulkan构建，NVIDIA电脑可换用官方CUDA构建并保持同一接口。

## 更换设备后的必要操作

设备从CPU改成GPU不会改变Embedding数学模型和维度，现有索引通常可以继续用，但应运行`rag.py doctor`和固定问题集验证数值排序。更换Embedding模型、量化版本或归一化方式则必须重建整个FAISS索引。
