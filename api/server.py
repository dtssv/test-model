"""
FastAPI服务主入口
提供OpenAI兼容的API接口
"""

from typing import Optional, List, Dict, Any, Union
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import logging
import time
import asyncio
import json
from datetime import datetime

from inference import VLLMEngine
from inference.engine import GenerationConfig

logger = logging.getLogger(__name__)

# 初始化FastAPI应用
app = FastAPI(
    title="Multimodal LLM API",
    description="OpenAI-compatible API for Multimodal Large Language Model",
    version="0.1.0",
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== 请求/响应模型 ==============

class Message(BaseModel):
    """对话消息"""
    role: str = Field(..., description="消息角色: system/user/assistant")
    content: Union[str, List[Dict[str, Any]]] = Field(..., description="消息内容")


class ChatCompletionRequest(BaseModel):
    """Chat Completion请求"""
    model: str = Field(..., description="模型名称")
    messages: List[Message] = Field(..., description="对话消息列表")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="温度参数")
    top_p: Optional[float] = Field(0.9, ge=0.0, le=1.0, description="Top-p采样")
    top_k: Optional[int] = Field(50, ge=0, description="Top-k采样")
    max_tokens: Optional[int] = Field(512, ge=1, le=8192, description="最大生成token数")
    stream: Optional[bool] = Field(False, description="是否流式输出")
    stop: Optional[Union[str, List[str]]] = Field(None, description="停止词")
    presence_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0, description="存在惩罚")
    frequency_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0, description="频率惩罚")
    user: Optional[str] = Field(None, description="用户ID")


class ChatCompletionChoice(BaseModel):
    """Chat Completion选择"""
    index: int
    message: Message
    finish_reason: str


class Usage(BaseModel):
    """使用统计"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """Chat Completion响应"""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage


class CompletionRequest(BaseModel):
    """Text Completion请求"""
    model: str
    prompt: Union[str, List[str]]
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(0.9, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(50, ge=0)
    max_tokens: Optional[int] = Field(512, ge=1, le=8192)
    stream: Optional[bool] = Field(False)
    stop: Optional[Union[str, List[str]]] = None
    echo: Optional[bool] = Field(False)
    user: Optional[str] = None


class CompletionChoice(BaseModel):
    """Completion选择"""
    index: int
    text: str
    finish_reason: str


class CompletionResponse(BaseModel):
    """Completion响应"""
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Usage


class ModelInfo(BaseModel):
    """模型信息"""
    id: str
    object: str = "model"
    created: int
    owned_by: str = "custom"


class ModelList(BaseModel):
    """模型列表"""
    object: str = "list"
    data: List[ModelInfo]


# ============== 全局变量 ==============

# 推理引擎实例
engine: Optional[VLLMEngine] = None
model_name: str = "multimodal-llm"


# ============== 依赖注入 ==============

def get_engine() -> VLLMEngine:
    """获取推理引擎"""
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return engine


async def verify_api_key(authorization: Optional[str] = Header(None)) -> str:
    """验证API密钥"""
    # TODO: 实现真实的API密钥验证
    if authorization is None:
        # 允许无密钥访问（开发模式）
        return "anonymous"
    
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    
    api_key = authorization[7:]
    return api_key


# ============== API端点 ==============

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "Multimodal LLM API",
        "version": "0.1.0",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "healthy",
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    """列出可用模型"""
    models = [
        ModelInfo(
            id=model_name,
            created=int(time.time()),
            owned_by="custom",
        )
    ]
    return ModelList(data=models)


@app.get("/v1/models/{model_id}", response_model=ModelInfo)
async def get_model(model_id: str):
    """获取模型信息"""
    if model_id != model_name:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return ModelInfo(
        id=model_name,
        created=int(time.time()),
        owned_by="custom",
    )


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    engine: VLLMEngine = Depends(get_engine),
    api_key: str = Depends(verify_api_key),
):
    """
    Chat Completion API
    OpenAI兼容的对话补全接口
    """
    try:
        # 转换消息格式
        messages = [
            {"role": msg.role, "content": msg.content if isinstance(msg.content, str) else str(msg.content)}
            for msg in request.messages
        ]
        
        # 构建生成配置
        config = GenerationConfig(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stop_tokens=[request.stop] if isinstance(request.stop, str) else request.stop,
            stream=request.stream,
        )
        
        if request.stream:
            # 流式输出
            return StreamingResponse(
                stream_chat_response(engine, messages, config, request.model),
                media_type="text/event-stream",
            )
        else:
            # 非流式输出
            result = engine.chat(messages, config)
            
            # 构建响应
            response = ChatCompletionResponse(
                id=f"chatcmpl-{int(time.time())}",
                created=int(time.time()),
                model=request.model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=Message(role="assistant", content=result.text),
                        finish_reason=result.finish_reason,
                    )
                ],
                usage=Usage(
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    total_tokens=result.total_tokens,
                ),
            )
            
            return response
    
    except Exception as e:
        logger.error(f"Error in chat completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/completions")
async def completions(
    request: CompletionRequest,
    engine: VLLMEngine = Depends(get_engine),
    api_key: str = Depends(verify_api_key),
):
    """
    Text Completion API
    OpenAI兼容的文本补全接口
    """
    try:
        # 处理prompt
        if isinstance(request.prompt, list):
            prompts = request.prompt
        else:
            prompts = [request.prompt]
        
        # 构建生成配置
        config = GenerationConfig(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stop_tokens=[request.stop] if isinstance(request.stop, str) else request.stop,
            stream=request.stream,
        )
        
        if request.stream:
            # 流式输出
            return StreamingResponse(
                stream_completion_response(engine, prompts[0], config, request.model),
                media_type="text/event-stream",
            )
        else:
            # 非流式输出
            result = engine.generate(prompts[0], config)
            
            # 构建响应
            response = CompletionResponse(
                id=f"cmpl-{int(time.time())}",
                created=int(time.time()),
                model=request.model,
                choices=[
                    CompletionChoice(
                        index=0,
                        text=result.text if not request.echo else prompts[0] + result.text,
                        finish_reason=result.finish_reason,
                    )
                ],
                usage=Usage(
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    total_tokens=result.total_tokens,
                ),
            )
            
            return response
    
    except Exception as e:
        logger.error(f"Error in completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== 流式输出辅助函数 ==============

async def stream_chat_response(
    engine: VLLMEngine,
    messages: List[Dict[str, str]],
    config: GenerationConfig,
    model: str,
):
    """流式输出Chat响应"""
    try:
        for chunk in engine.chat_stream(messages, config):
            data = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": chunk},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(data)}\n\n"
        
        # 发送结束标记
        yield "data: [DONE]\n\n"
    
    except Exception as e:
        logger.error(f"Error in stream: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


async def stream_completion_response(
    engine: VLLMEngine,
    prompt: str,
    config: GenerationConfig,
    model: str,
):
    """流式输出Completion响应"""
    try:
        for chunk in engine.generate_stream(prompt, config):
            data = {
                "id": f"cmpl-{int(time.time())}",
                "object": "text_completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "text": chunk,
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(data)}\n\n"
        
        # 发送结束标记
        yield "data: [DONE]\n\n"
    
    except Exception as e:
        logger.error(f"Error in stream: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


# ============== 应用生命周期 ==============

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化"""
    global engine, model_name
    
    logger.info("Starting up API server...")
    
    # TODO: 从环境变量或配置文件读取模型路径
    import os
    model_path = os.getenv("MODEL_PATH", "./models/multimodal-llm")
    model_name = os.getenv("MODEL_NAME", "multimodal-llm")
    
    try:
        # 初始化推理引擎
        engine = VLLMEngine(
            model_path=model_path,
            tensor_parallel_size=int(os.getenv("TENSOR_PARALLEL_SIZE", "1")),
            gpu_memory_utilization=float(os.getenv("GPU_MEMORY_UTILIZATION", "0.9")),
            max_model_len=int(os.getenv("MAX_MODEL_LEN", "4096")),
        )
        engine.load_model()
        
        logger.info(f"Model {model_name} loaded successfully from {model_path}")
    
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        engine = None


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时清理"""
    global engine
    
    logger.info("Shutting down API server...")
    
    if engine is not None:
        engine.release_memory()
        engine = None


# ============== 异常处理 ==============

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP异常处理"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.detail,
                "type": "invalid_request_error",
                "param": None,
                "code": None,
            }
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """通用异常处理"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Internal server error",
                "type": "internal_error",
                "param": None,
                "code": None,
            }
        },
    )


# ============== 启动命令 ==============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
    )