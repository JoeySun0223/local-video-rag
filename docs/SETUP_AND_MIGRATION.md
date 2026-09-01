# 工程安装、模型与迁移

## 运行环境

- Windows 10/11 64位
- Python 3.11（当前为3.11.9）
- FFmpeg 7.1 essentials，项目内路径 `tools/ffmpeg/bin/ffmpeg.exe`
- 项目内独立 llama.cpp Vulkan 构建：`tools/llama.cpp/b10621/llama-server.exe`
- 当前CPU环境核心版本：PyTorch 2.13.0+cpu、Transformers 5.15.1、Sentence Transformers 6.0.0、FAISS CPU 1.15.0、NumPy 2.4.6、FastAPI 0.135.1

建议在目标电脑重新创建虚拟环境，不要直接复制`venv`：

```powershell
py -3.11 -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements-local-rag.txt
```

若不直接复制`tools/llama.cpp/b10621/`，可从llama.cpp官方发布页下载
`llama-b10621-bin-win-vulkan-x64.zip`并解压到该目录；NVIDIA电脑可改用同版本CUDA构建。

PyTorch CPU或CUDA版本应按目标电脑单独安装，安装完成后运行：

```powershell
.\venv\Scripts\python.exe .\rag.py doctor --hashes
```

## 当前完整模型清单

### 新视频入库

- `models/qwen3-asr-1.7b-hf`：`Qwen/Qwen3-ASR-1.7B-hf`，BF16。
- `models/qwen3-forced-aligner-0.6b-hf`：句/词真实时间对齐。
- `models/asr`、`models/vad`、`models/punctuation`：旧FunASR回退方案，当前默认不使用但保留。

### 检索和回答

- `models/embedding-v1.5`：`BAAI/bge-large-zh-v1.5`。
- `models/reranker`：`BAAI/bge-reranker-base`。
- `models/generator/Qwen3.5-4B-Q4_K_M.gguf`：本地回答与入库分章模型，由独立 llama-server 加载。

权重哈希记录在 `models/manifest.json`。模型目录被Git忽略，上传GitHub时只提交清单和下载/放置说明，不提交大权重。

## 配置与路径

项目路径全部相对 `config.yaml`。`knowledge_base/`可整体迁移。生成服务路径由
`llm.server_path` 指定，GGUF由 `llm.model_path` 指定；两者默认均位于项目目录，不需要安装LM Studio。

## 整个项目迁移

1. 使用`关闭本地视频知识库.cmd`停止Web和模型。
2. 复制项目目录；正式知识资产全部位于`knowledge_base/`。
3. 在目标电脑重建`venv`。
4. 放置`models/`与`tools/ffmpeg/`，或修改`config.yaml`为目标路径。
5. 保留`tools/llama.cpp/`和`models/generator/`，或修改`config.yaml`为目标路径。
6. 运行`rag.py doctor --hashes`。
7. 双击`启动本地视频知识库.cmd`。

## GitHub工程约定

源码、配置示例、测试和文档可以提交。`.gitignore`默认排除`venv/`、`models/`、知识资产内容、音视频、日志和本机私有配置。不要把正式视频、转写、查询日志或模型权重推送到公共仓库。
