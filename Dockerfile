# 多模态大模型系统 Dockerfile
# 多阶段构建：基础镜像 -> 训练镜像 -> 推理镜像

# ============== 基础镜像 ==============
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 AS base

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    wget \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# 设置Python 3.11为默认版本
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --upgrade pip setuptools wheel \
    && pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121 \
    && pip install -r requirements.txt

# ============== 训练镜像 ==============
FROM base AS training

# 安装训练相关依赖
RUN pip install \
    deepspeed==0.14.0 \
    megatron-core \
    flash-attn==2.5.0

# 复制训练代码
COPY training/ /app/training/
COPY data_pipeline/ /app/data_pipeline/
COPY model_architecture/ /app/model_architecture/

# 设置训练环境变量
ENV PYTHONPATH=/app
ENV WANDB_PROJECT=multimodal-llm

# 训练入口脚本
COPY scripts/train.sh /app/scripts/train.sh
RUN chmod +x /app/scripts/train.sh

# 默认命令
CMD ["/bin/bash"]

# ============== 推理镜像 ==============
FROM base AS inference

# 安装推理相关依赖
RUN pip install \
    vllm==0.6.0 \
    tensorrt-llm==0.10.0

# 复制推理代码
COPY inference/ /app/inference/
COPY model_architecture/ /app/model_architecture/
COPY api/ /app/api/

# 设置推理环境变量
ENV PYTHONPATH=/app
ENV MODEL_PATH=/models
ENV MODEL_NAME=multimodal-llm

# 暴露API端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 启动API服务
CMD ["python", "-m", "api.server"]

# ============== 数据处理镜像 ==============
FROM base AS data-pipeline

# 安装数据处理依赖
RUN pip install \
    datasets==2.18.0 \
    data-juicer \
    label-studio-sdk

# 复制数据管线代码
COPY data_pipeline/ /app/data_pipeline/

# 设置数据环境变量
ENV PYTHONPATH=/app
ENV DATA_DIR=/data

# 数据处理入口脚本
COPY scripts/process_data.sh /app/scripts/process_data.sh
RUN chmod +x /app/scripts/process_data.sh

# 默认命令
CMD ["/bin/bash"]