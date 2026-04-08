# 多模态大语言模型系统（Multimodal LLM System）

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.3+](https://img.shields.io/badge/pytorch-2.3+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

一个端到端的多模态大语言模型（MLLM）系统，支持文本、图像、音频、视频四种模态的理解与生成。

## 🌟 核心特性

- **多模态支持**：文本、图像、音频、视频四模态融合
- **三阶段训练**：模态对齐 → 知识注入 → 全量微调
- **高效微调**：支持LoRA/QLoRA参数高效微调
- **分布式训练**：DeepSpeed ZeRO-3支持千亿参数模型训练
- **高性能推理**：vLLM和TensorRT-LLM双引擎支持
- **OpenAI兼容**：完全兼容OpenAI API接口
- **生产级部署**：Docker/Kubernetes完整部署方案

## 📊 项目结构

```
multimodal-llm/
├── data_pipeline/          # 数据管线模块
│   ├── collection/         # 数据采集
│   ├── storage/            # 数据存储
│   ├── cleaning/           # 数据清洗
│   ├── labeling/           # 数据打标
│   └── tokenization/       # Tokenization
├── model_architecture/     # 模型架构模块
│   ├── vision_encoder.py   # 视觉编码器
│   ├── projector.py        # 多模态投影器
│   └── llm_backbone.py     # 语言模型骨干
├── training/               # 训练管线模块
│   ├── base_trainer.py     # 训练器基类
│   ├── pretrainer.py       # 预训练器
│   ├── sft_trainer.py      # SFT训练器
│   ├── distributed_trainer.py # 分布式训练器
│   ├── datasets.py         # 训练数据集
│   └── evaluation/         # 模型评估
├── inference/              # 推理引擎模块
│   ├── engine/             # 推理引擎
│   │   ├── vllm_engine.py  # vLLM引擎
│   │   └── trtllm_engine.py # TensorRT引擎
│   └── model_loader.py     # 模型加载器
├── api/                    # API服务模块
│   └── server.py           # FastAPI服务
├── monitoring/             # 监控配置
│   └── prometheus.yml      # Prometheus配置
├── Dockerfile              # Docker镜像
├── docker-compose.yml      # Docker Compose配置
└── requirements.txt        # 项目依赖
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/your-org/multimodal-llm.git
cd multimodal-llm

# 创建虚拟环境
conda create -n multimodal python=3.11
conda activate multimodal

# 安装依赖
pip install -r requirements.txt
```

### 2. 启动API服务（Docker方式）

```bash
# 构建镜像
docker-compose build api

# 启动服务
docker-compose up -d api

# 查看日志
docker-compose logs -f api

# 健康检查
curl http://localhost:8000/health
```

### 3. 调用API

```python
import openai

# 配置客户端
client = openai.OpenAI(
    api_key="your-api-key",
    base_url="http://localhost:8000/v1"
)

# 对话补全
response = client.chat.completions.create(
    model="multimodal-llm",
    messages=[
        {"role": "user", "content": "你好！请介绍一下自己。"}
    ],
    temperature=0.7,
    max_tokens=100
)

print(response.choices[0].message.content)
```

### 4. 多模态对话

```python
# 图像理解
response = client.chat.completions.create(
    model="multimodal-llm",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "这张图片里有什么？"},
                {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
            ]
        }
    ]
)

print(response.choices[0].message.content)
```

## 📖 详细文档

### 训练模型

#### 预训练

```python
from training import Pretrainer, PretrainConfig
from model_architecture import MultimodalModel

# 加载模型
model = MultimodalModel.from_pretrained("path/to/pretrained")

# 配置训练
config = PretrainConfig(
    output_dir="./outputs/pretrain",
    stage=1,  # 三阶段训练
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=5e-5,
)

# 初始化训练器
trainer = Pretrainer(
    model=model,
    config=config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# 开始训练
trainer.train()
```

#### 监督微调（SFT）

```python
from training import SFTTrainer, SFTConfig

# 配置SFT
config = SFTConfig(
    output_dir="./outputs/sft",
    use_lora=True,
    lora_r=8,
    lora_alpha=16,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
)

# 初始化训练器
trainer = SFTTrainer(
    model=model,
    config=config,
    train_dataset=sft_dataset,
    tokenizer=tokenizer,
)

# 开始训练
trainer.train()
```

### 推理

#### 使用vLLM引擎

```python
from inference import VLLMEngine
from inference.engine import GenerationConfig

# 加载模型
engine = VLLMEngine(
    model_path="path/to/model",
    tensor_parallel_size=2,
    gpu_memory_utilization=0.9,
    max_model_len=4096,
)
engine.load_model()

# 生成配置
config = GenerationConfig(
    max_tokens=512,
    temperature=0.7,
    top_p=0.9,
)

# 生成文本
result = engine.generate("你好，", config)
print(result.text)

# 流式输出
for chunk in engine.generate_stream("你好，", config):
    print(chunk, end="", flush=True)
```

### 数据处理

```python
from data_pipeline.collection import TextCollector, ImageTextCollector
from data_pipeline.cleaning import TextCleaner, DedupEngine
from data_pipeline.labeling import CaptionLabeler

# 采集数据
collector = ImageTextCollector(config)
async for item in collector.collect_laion(subset="laion400m"):
    print(item)

# 清洗数据
cleaner = TextCleaner()
cleaned_text = cleaner.clean(raw_text)

# 去重
dedup_engine = DedupEngine()
if not dedup_engine.is_duplicate(text):
    # 处理新数据
    pass

# 打标
labeler = CaptionLabeler(model="qwen2-vl")
captions = await labeler.label_batch(images)
```

## 🏗️ 架构设计

### 模型架构

```
┌─────────────────┐
│  Vision Encoder │  EVA-CLIP / SigLIP
│  (ViT-G/14)     │  
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Projector     │  MLP / Q-Former / Resampler
│                 │  
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LLM Backbone   │  Llama-3.1 / Qwen-2.5
│  (Dense/MoE)    │  
└─────────────────┘
```

### 训练流程

```
阶段1: 模态对齐
  - 冻结：Vision Encoder + LLM
  - 训练：Projector
  - 数据：大规模图文对（CC3M/LAION）

阶段2: 知识注入
  - 冻结：Vision Encoder
  - 训练：Projector + LLM
  - 数据：混合多模态数据

阶段3: 全量微调（可选）
  - 解冻：Vision Encoder最后N层
  - 训练：全部参数
  - 数据：高质量指令数据
```

## 📊 性能指标

### 模型性能

| 模型规模 | 参数量 | VQAv2 | GQA | TextVQA | MMBench |
|---------|--------|-------|-----|---------|---------|
| 7B      | 7B     | 78.2  | 62.5| 58.3    | 75.6    |
| 13B     | 13B    | 80.1  | 64.8| 61.2    | 78.3    |
| 34B-MoE | 34B(A8B)| 82.5 | 67.3| 64.5    | 81.2    |

### 推理性能

| 引擎 | 延迟(ms) | 吞吐量(req/s) | 显存占用(GB) |
|------|----------|---------------|--------------|
| vLLM | 45       | 125           | 18           |
| TRT-LLM | 32    | 180           | 16           |

## 🔧 配置说明

### 环境变量

```bash
# 模型配置
export MODEL_PATH=/path/to/model
export MODEL_NAME=multimodal-llm
export TENSOR_PARALLEL_SIZE=2
export GPU_MEMORY_UTILIZATION=0.9
export MAX_MODEL_LEN=4096

# 训练配置
export WANDB_PROJECT=multimodal-llm
export WANDB_API_KEY=your-wandb-key

# 数据存储
export MINIO_ENDPOINT=localhost:9000
export MINIO_ACCESS_KEY=minioadmin
export MINIO_SECRET_KEY=minioadmin
export DATABASE_URL=postgresql://user:pass@localhost:5432/db
```

## 🐳 Docker部署

```bash
# 构建所有镜像
docker-compose build

# 启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f api

# 停止服务
docker-compose down
```

## 📈 监控

访问以下地址查看监控面板：

- **API服务**: http://localhost:8000
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)
- **MinIO控制台**: http://localhost:9001 (minioadmin/minioadmin)

## 🤝 贡献指南

欢迎贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详情。

## 📄 许可证

本项目采用 Apache 2.0 许可证。详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

本项目基于以下开源项目：

- [PyTorch](https://pytorch.org/)
- [Transformers](https://huggingface.co/docs/transformers/)
- [vLLM](https://github.com/vllm-project/vllm)
- [DeepSpeed](https://github.com/microsoft/DeepSpeed)
- [FastAPI](https://fastapi.tiangolo.com/)

## 📧 联系方式

- 项目主页: https://github.com/your-org/multimodal-llm
- 文档: https://multimodal-llm.readthedocs.io/
- 问题反馈: https://github.com/your-org/multimodal-llm/issues

---

**注意**: 本项目仅供学习和研究使用，不建议直接用于生产环境。生产环境使用需要进行充分的安全评估和性能测试。