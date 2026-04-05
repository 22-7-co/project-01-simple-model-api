# =========================================================================
# FASTAPI 实现
# =========================================================================

'''
FastAPI 应用程序实现
FastAPI 提供自动 API 文档和更好的异步支持。
'''
import io
import uuid
import time
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
from fastapi.concurrency import asynccontextmanager

from src.config import Config
from src.model_loader import ModelLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)

config = Config()
model_loader = None


# =========================================================================
# Pydantic 模型
# =========================================================================

class PredictionItem(BaseModel):
    '''
    单个预测项。
    '''
    class_name: str
    confidence: float
    rank: int

class PredictionResponse(BaseModel):
    '''
    完整的预测响应。
    '''
    success: bool
    predictions: List[PredictionItem]
    latency_ms: float
    correlation_id: str
    timestamp: str

class ErrorResponse(BaseModel):
    '''
    错误响应模型。
    '''
    success: bool
    error: dict


# =========================================================================
# 辅助函数
# =========================================================================

def generate_correlation_id() -> str:
    '''
    生成用于请求跟踪的唯一关联 ID。
    '''
    return str(uuid.uuid4())

def format_success_response(predictions: list,
                           latency_ms: float,
                           correlation_id: str) -> dict:
    '''
    格式化成功的预测响应。
    '''
    return {
        'success': True,
        'predictions': predictions,
        'latency_ms': latency_ms,
        'correlation_id': correlation_id,
        'timestamp': datetime.now().isoformat() + 'Z'
    }

def format_error_response(error_code: str, message: str, correlation_id: str, details: Optional[dict] = None) -> dict:
    '''
    格式化错误响应。
    '''
    error_response = {
        'success': False,
        'error': {
            'code': error_code,
            'message': message,
            'correlation_id': correlation_id,
            'timestamp': datetime.now().isoformat() + 'Z'
        }
    }

    if details:
        error_response['error']['details'] = details

    return error_response


# =========================================================================
# FastAPI 应用
# =========================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    '''
    FastAPI 启动时初始化模型。
    '''
    global model_loader
    try:
        logger.info("正在初始化模型...")
        model_loader = ModelLoader(
            model_name=config.MODEL_NAME,
            device=config.DEVICE
        )
        model_loader.load()
        logger.info("模型初始化成功。")
    except Exception as e:
        logger.error(f"初始化模型时出错：{e}")
        raise
    yield
    # 关闭时清理（如需要）

app = FastAPI(
    lifespan=lifespan,
    title='模型推理 API',
    description='用于图像分类的 REST API',
    version='1.0.0',
)


# =========================================================================
# 端点
# =========================================================================

@app.get('/health')
async def health():
    '''
    健康检查端点。
    '''
    is_healthy = (model_loader is not None) and (model_loader.model is not None)
    if is_healthy:
        return JSONResponse(status_code=200, content={
            'status': 'healthy',
            'model_loaded': True,
            'model_name': config.MODEL_NAME,
            'uptime': time.time()
        })
    else:
        return JSONResponse(status_code=503, content={
            'status': 'unhealthy',
            'model_loaded': False,
            'reason': '模型未加载',
            'uptime': time.time()
        })

@app.get('/info')
async def info():
    '''
    包含模型和 API 信息的端点。
    '''
    if model_loader is None:
        raise HTTPException(status_code=503, detail={
            'error': '模型未加载',
            'uptime': time.time()
        })
    model_info = model_loader.get_model_info()
    return JSONResponse(status_code=200, content={
        'model': model_info,
        'api_version': config.API_VERSION,
        'endpoints': ['/predict', '/health', '/info'],
        'limits': {
            'max_file_size': config.MAX_FILE_SIZE,
            'max_image_dimension': config.MAX_IMAGE_DIMENSION,
            'timeout_seconds': config.REQUEST_TIMEOUT
        },
        'uptime': time.time()
    })

@app.post('/predict')
async def predict(file: UploadFile = File(...), top_k: int = config.DEFAULT_TOP_K):
    '''
    预测端点。
    '''
    correlation_id = generate_correlation_id()
    start_time = time.time()
    
    try:
        # 1. 验证请求包含文件
        if not file:
            return JSONResponse(
                status_code=400,
                content=format_error_response(
                    'MISSING_FILE',
                    '请求中未提供文件',
                    correlation_id
                )
            )
        
        # 2. 检查文件有名称
        if file.filename == '':
            return JSONResponse(
                status_code=400,
                content=format_error_response(
                    'EMPTY_FILENAME',
                    '文件名为空',
                    correlation_id
                )
            )
        
        # 3. 检查文件大小
        file_bytes = await file.read()
        file_size = len(file_bytes)
        if file_size > config.MAX_FILE_SIZE:
            return JSONResponse(
                status_code=413,
                content=format_error_response(
                    'FILE_TOO_LARGE',
                    f'文件大小 {file_size} 超过最大限制 {config.MAX_FILE_SIZE} 字节',
                    correlation_id
                )
            )
        
        # 4. 加载图像
        try:
            image = Image.open(io.BytesIO(file_bytes))
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content=format_error_response(
                    'INVALID_IMAGE_FORMAT',
                    f'无法打开图像：{str(e)}',
                    correlation_id
                )
            )
        
        # 5. 验证图像
        is_valid, error_msg = model_loader.validate_image(image)
        if not is_valid:
            return JSONResponse(
                status_code=400,
                content=format_error_response(
                    'INVALID_IMAGE',
                    error_msg,
                    correlation_id
                )
            )
        
        # 6. 验证 top_k 参数
        try:
            top_k = int(top_k)
            if top_k < 1 or top_k > config.MAX_TOP_K:
                return JSONResponse(
                    status_code=400,
                    content=format_error_response(
                        'INVALID_PARAMETER',
                        f'top_k 必须在 1 到 {config.MAX_TOP_K} 之间',
                        correlation_id
                    )
                )
        except (ValueError, TypeError):
            return JSONResponse(
                status_code=400,
                content=format_error_response(
                    'INVALID_PARAMETER',
                    'top_k 必须是整数',
                    correlation_id
                )
            )
        
        # 7. 生成预测
        predictions = model_loader.predict(image, top_k=top_k)
        
        # 8. 计算延迟
        latency_ms = (time.time() - start_time) * 1000
        
        # 9. 记录请求日志
        logger.info(f"预测成功：correlation_id={correlation_id}, "
                    f"latency={latency_ms:.2f}ms, top_class={predictions[0]['class']}")
        
        # 10. 返回响应
        return JSONResponse(
            status_code=200,
            content=format_success_response(
                predictions,
                latency_ms,
                correlation_id
            )
        )
    
    except Exception as e:
        logger.error(f'预测错误：{e}', exc_info=True)
        return JSONResponse(
            status_code=500,
            content=format_error_response(
                'PREDICTION_ERROR',
                '内部服务器错误',
                correlation_id
            )
        )


# =========================================================================
# 主入口
# =========================================================================

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(
        app,
        host=config.HOST,
        port=config.Port,
        log_level=config.LOG_LEVEL.lower(),
    )
