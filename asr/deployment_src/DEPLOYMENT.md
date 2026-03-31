# TikTalk ASR Cloud Deployment Guide

本文档说明如何将 LoRA 微调后的 Whisper large-v3 模型部署到 Modal 云平台。

---

## 1. 项目代码结构

```
tiktalk_asr_cloud_deploy/
├── best_adapter/              # LoRA 微调产出文件
│   ├── adapter_config.json    # LoRA 配置 (r=16, alpha=32, dropout=0.05)
│   ├── adapter_model.safetensors  # LoRA 权重文件
│   ├── processor_config.json  # Whisper processor 配置
│   ├── tokenizer.json         # Tokenizer 词表
│   └── tokenizer_config.json  # Tokenizer 配置
├── merge_and_convert.py       # 本地脚本：合并权重 + 转换格式
├── serve.py                   # Modal 部署脚本：推理服务
├── pyproject.toml             # Python 依赖声明
├── DEPLOYMENT.md              # 本文档
├── FineTuneInstruction.md     # 微调过程记录
└── project_document.md        # 项目需求文档
```

### 生成的目录（不提交到 Git）

| 目录            | 说明                                         |
| --------------- | -------------------------------------------- |
| `merged_model/` | 合并 LoRA 权重后的完整 Whisper large-v3 模型 |
| `ct2_model/`    | CTranslate2 格式模型，供 faster-whisper 使用 |

---

## 2. 代码说明

### 2.1 `merge_and_convert.py` — 权重合并与格式转换

**作用**：在本地完成两步操作，为部署做准备。

| 步骤        | 操作                                        | 核心技术                          |
| ----------- | ------------------------------------------- | --------------------------------- |
| Step 1 合并 | 加载 whisper-large-v3 + LoRA adapter → 合并 | PEFT `merge_and_unload()`         |
| Step 2 转换 | 将合并模型转为 CTranslate2 格式             | `ct2-transformers-converter` CLI  |

**关键参数**：

- **CTranslate2 量化**：`float16` — 保持推理精度的同时减少模型大小

### 2.2 `serve.py` — Modal 推理服务

**作用**：定义 Modal 应用，在云端 T4 GPU 上运行 faster-whisper 推理。

#### 推理超参数

| 参数           | 值       | 说明                                       |
| -------------- | -------- | ------------------------------------------ |
| `beam_size`    | 5        | Beam search 宽度，越大越精确但越慢         |
| `language`     | `"en"`   | 目标语言（英语，儿童语音）                 |
| `vad_filter`   | `True`   | 启用语音活动检测，自动跳过静音段           |
| `min_silence_duration_ms` | 500 | VAD 最小静音时长（毫秒）           |
| `compute_type` | `float16`| GPU 推理精度                               |

#### Modal 资源配置

| 参数                      | 值    | 说明                     |
| ------------------------- | ----- | ------------------------ |
| `gpu`                     | `T4`  | 使用 T4 GPU              |
| `timeout`                 | 300s  | 单次请求超时             |
| `container_idle_timeout`  | 120s  | 容器空闲超时后自动关闭   |
| `allow_concurrent_inputs` | 5     | 允许并发请求数           |

#### API 接口

- **Method**: POST
- **Content-Type**: `multipart/form-data`
- **字段名**: `audio`
- **支持格式**: `.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`, `.webm`

**响应示例**：

```json
{
  "text": "hello how are you doing today",
  "segments": [
    { "start": 0.0, "end": 2.5, "text": "hello how are you" },
    { "start": 2.5, "end": 4.0, "text": "doing today" }
  ],
  "language": "en",
  "language_probability": 0.98,
  "duration": 4.0
}
```

---

## 3. 部署流程（命令行步骤）

### 前提条件

- 已安装 [uv](https://docs.astral.sh/uv/)
- 已注册 [Modal](https://modal.com/) 账号
- `best_adapter/` 目录中包含微调后的 LoRA 文件

### Step 1: 安装依赖

```bash
uv sync
```

### Step 2: 设置 Modal

注册/登录 Modal 后，运行：

```bash
uv run modal setup
```

按提示在浏览器中完成认证。

### Step 3: 合并权重并转换格式

```bash
uv run python merge_and_convert.py
```

这一步会：
1. 从 Hugging Face 下载 `openai/whisper-large-v3` 基座模型（~6GB，首次运行）
2. 合并 LoRA 权重 → 输出到 `merged_model/`
3. 转换为 CTranslate2 格式 → 输出到 `ct2_model/`

> ⏱️ 首次运行可能需要 10-20 分钟（取决于网络和硬件）

### Step 4: 上传模型到 Modal Volume

```bash
uv run modal volume create tiktalk-asr-model
uv run modal volume put tiktalk-asr-model ct2_model/ /
```

### Step 5: 部署到 Modal

```bash
uv run modal deploy serve.py
```

部署成功后会输出 Web endpoint URL，格式类似：
```
https://your-workspace--tiktalk-asr-transcribe-endpoint.modal.run
```

### Step 6（开发模式，可选）: 热重载调试

```bash
uv run modal serve serve.py
```

这会启动一个临时 endpoint，代码修改后自动重新部署。

---

## 4. Postman 测试

1. 打开 Postman，创建新请求
2. 方法选择 **POST**
3. URL 填入 Modal 部署后输出的 endpoint URL
4. 切换到 **Body** tab → 选择 **form-data**
5. 添加一个字段：
   - **Key**: `audio`（类型选择 **File**）
   - **Value**: 选择本地的 `.wav` 或 `.mp3` 文件
6. 点击 **Send**
7. 查看返回的 JSON，包含 `text`（完整转录）和 `segments`（分段转录+时间戳）

---

## 5. 常用 Modal 命令

| 命令                                    | 说明                       |
| --------------------------------------- | -------------------------- |
| `modal deploy serve.py`                 | 部署（生产模式）           |
| `modal serve serve.py`                  | 开发调试模式               |
| `modal app list`                        | 查看已部署的应用           |
| `modal app stop tiktalk-asr`            | 停止应用                   |
| `modal volume ls tiktalk-asr-model`     | 查看 Volume 中的文件       |
| `modal volume put tiktalk-asr-model ct2_model/ /` | 上传模型到 Volume |
| `modal volume delete tiktalk-asr-model` | 删除 Volume（慎用）        |
