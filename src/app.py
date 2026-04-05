import io
import time
import uuid
import logging
from datetime import datetime
from typing import Dict, Tuple, Optional
from PIL import Image
from flask import Flask, request, jsonify
from src.config import Config
from src.model_loader import ModelLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s',
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

config = Config()
model_loader = None

def init_model():
    '''
    初始化并加载机器学习模型。

    实现模型初始化
    - 创建 ModelLoader 实例
    - 加载模型
    - 优雅处理错误
    - 这在启动时被调用一次
    '''
    global model_loader
    try:
        logger.info("Initializing model...")
        model_loader = ModelLoader(
            model_name = config.MODEL_NAME,
            device = config.DEVICE
        )
        model_loader.load()
        logger.info("Model initialized sccessfully.")
    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        raise #主动抛出一个错误，让程序停下来

    return True

@app.route('/health', methods=['GET'])
def health():
    '''
    健康检查端点。

    实现健康检查
    - 检查模型是否已加载
    - 如果模型已加载则返回健康状态
    - 如果模型未加载则返回不健康状态（503）
    - 包含模型名称和运行时间
    返回：
        包含健康状态的 JSON 响应
    '''
    is_healthy = (model_loader is not None) and (model_loader.model is not None)

    if is_healthy:
        return jsonify({
            'status': 'healthy',
            'model_loaded': True,
            'model_name': config.MODEL_NAME,
            'uptime': time.time()
        }), 200
    else:
        return jsonify({
            'status': 'unhealthy',
            'model_loaded': False,
            'reason': 'Model not loaded',
            'uptime': time.time()
        }), 503

@app.route('/info', methods=['GET'])
def info():
    '''
    模型信息端点。

    实现信息端点
    - 返回模型元数据
    - 包含 API 版本
    - 包含支持的端点
    - 包含限制（文件大小、超时等）

    返回：
        包含模型和 API 信息的 JSON 响应
    '''
    if model_loader is None:
        return jsonify({
            'error': 'Model not loaded',
            'uptime': time.time()
        }),503
    model_info = model_loader.get_model_info()

    return jsonify({
        'model': model_info,
        'api_version': config.API_VERSION,
        'endpoints': ['/predict', '/health', '/info'],
        'limits': {
            'max_file_size': config.MAX_FILE_SIZE,
            'max_image_dimension': config.MAX_IMAGE_DIMENSION,
            'timeout_seconds' : config.REQUEST_TIMEOUT
        },
        'uptime': time.time()
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    '''
    预测端点。

    实现预测端点
    1. 生成关联 ID 用于请求跟踪
    2. 验证请求包含文件
    3. 验证文件大小
    4. 加载并验证图像
    5. 获取 top_k 参数（可选）
    6. 调用 model_loader.predict()
    7. 测量延迟
    8. 格式化响应
    9. 记录请求
    10. 优雅处理所有错误

    返回：
        包含预测结果或错误的 JSON 响应
    '''
    correlation_id = generate_correlation_id()
    start_time = time.time()
    try:
        # 1. 验证请求包含文件
        if 'file' not in request.files:
            return jsonify(format_error_response(
                'MISSING_FILE',
                '请求中未提供文件',
                correlation_id
            )), 400
        file = request.files['file']

        # 2. 检查文件有名称
        if file.filename == '':
            return jsonify(format_error_response(
                'EMPTY_FILENAME',
                '文件名为空',
                correlation_id
            )),400
        
        # 3. 检查文件大小
        file.seek(0, 2)
        file_size = file.tell()
        file.seek(0)
        if file_size > config.MAX_FILE_SIZE:
            return jsonify(format_error_response(
                'FILE_TOO_LARGE',
                f'文件大小 {file_size} 超过最大限制 {config.MAX_FILE_SIZE} 字节',
                correlation_id
            )),413 

        # 4. 加载图像
        try:
            file_bytes = file.read()
            image = Image.open(io.BytesIO(file_bytes))
        except Exception as e:
            return jsonify(format_error_response(
                'INVALID_IMAGE_FORMAT',
                f'无法打开图像：{str(e)}',
                correlation_id
            )),400
        
        # 5. 验证图像
        is_valid, error_msg = model_loader.validate_image(image)
        if not is_valid:
            return jsonify(format_error_response(
                'INVALID_IMAGE',
                error_msg,
                correlation_id
            )),400
        
        # 6. 获取 top_k 参数
        top_k = request.form.get('top_k', config.DEFAULT_TOP_K)
        try:
            top_k = int(top_k)
            if top_k < 1 or top_k > config.MAX_TOP_K:
                return jsonify(format_error_response(
                    'INVALID_PARAMETER',
                    f'top_k 必须在 1 到 {config.MAX_TOP_K} 之间',
                    correlation_id
                )), 400
        except ValueError:
            return jsonify(format_error_response(
                'INVALID_PARAMETER',
                'top_k 必须是整数',
                correlation_id
            )), 400
        
        # 7. 生成预测
        predictions = model_loader.predict(image, top_k=top_k)

        # 8. 计算延迟
        latency_ms = (time.time() - start_time) * 1000

        # 9. 记录请求日志
        logger.info(f"预测成功：correlation_id={correlation_id}, "
                    f"latency={latency_ms:.2f}ms, top_class={predictions[0]['class']}")
        
        # 10. 返回响应
        return jsonify(format_success_response(
            predictions,
            latency_ms,
            correlation_id
        )), 200
    except Exception as e:
        logger.error(f'预测错误：{e}', exc_info=True)
        return jsonify(format_error_response(
            'PREDICTION_ERROR',
            '内部服务器错误',
            correlation_id
        )), 500

# =========================================================================
# 错误处理器
# =========================================================================
@app.errorhandler(404)
def not_found(error):
    ''' 
    处理 404 未找到错误。
    '''
    return jsonify({
        'success': False,
        'error': {
            'code': 'NOT_FOUND',
            'message': 'Endpoint not found'
        }
    }), 404

#app.errorhandler(405)
def method_not_allowed(error):
    ''' 
    处理 405 方法不允许错误。
    '''
    return jsonify({
        'success': False,
        'error': {
            'code': 'METHOD_NOT_ALLOWED',
            'message': 'Method not allowed for this endpoint'
        }
    }), 405

@app.errorhandler(500)
def internal_error(error):
    ''' 
    处理 500 内部服务器错误。
    '''
    return jsonify({
        'success': False,
        'error': {
            'code': 'INTERNAL_SERVER_ERROR',
            'message': '发生未预期的错误'
        }
    }), 500


# =========================================================================
# 辅助函数
# =========================================================================

def generate_correlation_id() -> str:
    '''
    生成用于请求跟踪的唯一关联 ID。

    实现关联 ID 生成
    - 使用 UUID 确保唯一性
    - 格式为 'req-<8-char-hex>'
    - 用于在日志中跟踪请求

    返回：
        关联 ID 字符串
    '''
    return str(uuid.uuid4())

def format_success_response(predictions: list,
                           latency_ms: float,
                           correlation_id: str) -> dict:
    '''
    格式化成功的预测响应。

    实现成功响应格式化
    - 包含 success=True
    - 包含预测列表
    - 包含延迟
    - 包含 correlation_id
    - 包含时间戳

    参数：
        predictions: 预测字典列表
        latency_ms: 请求延迟（毫秒）
        correlation_id: 请求关联 ID

    返回：
        格式化响应字典
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

    待实现：实现错误响应格式化
    - 包含 success=False
    - 包含包含代码、消息的错误对象
    - 包含用于跟踪的 correlation_id
    - 包含时间戳
    - 可选包含详细信息

    参数：
        error_code: 错误代码（例如 'INVALID_IMAGE'）
        message: 人类可读的错误消息
        correlation_id: 请求关联 ID
        details: 可选的附加详细信息

    返回：
        格式化的错误响应字典
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

def validate_image(file) -> Tuple[bool, Optional[str]]:
    '''
    验证上传的文件是有效的图像。

    实现文件验证
    - 检查文件不为 None
    - 检查文件有内容
    - 尝试用 PIL 打开
    - 返回 (is_valid, error_message)

    参数：
        file: Flask 文件对象

    返回：
        (is_valid, error_message) 元组
    '''
    pass

# =========================================================================
# 应用程序启动
# =========================================================================


if __name__ == '__main__':
    '''
     运行 Flask 应用程序。

    实现应用程序启动
    - 初始化模型
    - 启动 Flask 服务器
    - 使用配置的主机和端口
    '''
    try:
        init_model()
        logger.info(f"Starting application on {config.HOST}:{config.Port}")
        app.run(host=config.HOST, port=config.Port, debug=config.DEBUG)
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        exit(1)
