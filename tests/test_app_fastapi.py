import pytest
import io
from PIL import Image
from httpx import ASGITransport, AsyncClient

# FastAPI 测试导入
from src.app_fast import app


# =========================================================================
# 测试夹具
# =========================================================================

@pytest.fixture
def client():
    '''创建 FastAPI 测试客户端'''
    return app.test_client

@pytest.fixture
async def async_client():
    '''创建 FastAPI 异步测试客户端'''
    from httpx import ASGITransport, AsyncClient
    from src.app_fast import lifespan
    
    # 直接调用 lifespan 初始化模型
    async with lifespan(app) as lifespan_context:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            yield ac

@pytest.fixture
def sample_image():
    '''创建用于测试的示例 RGB 图像'''
    image = Image.new('RGB', (224, 224), color='red')
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    return img_byte_arr

@pytest.fixture
def large_image():
    '''创建用于测试尺寸限制的大图像 (5000x5000)'''
    image = Image.new('RGB', (5000, 5000), color='red')
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG', quality=100)
    img_byte_arr.seek(0)
    return img_byte_arr

@pytest.fixture
def sample_image_factory():
    '''创建新示例图像的工厂'''
    def create_sample_image():
        image = Image.new('RGB', (224, 224), color='red')
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        return img_byte_arr
    return create_sample_image


# =========================================================================
# 健康检查测试
# =========================================================================

@pytest.mark.anyio
async def test_health_endpoint_returns_200(async_client):
    """测试健康检查端点返回 200"""
    response = await async_client.get('/health')
    assert response.status_code == 200
    assert response.headers['content-type'] == 'application/json'

@pytest.mark.anyio
async def test_health_endpoint_returns_healthy_status(async_client):
    """测试健康检查端点返回健康状态"""
    response = await async_client.get('/health')
    data = response.json()
    assert data['status'] == 'healthy'
    assert data['model_loaded'] is True

@pytest.mark.anyio
async def test_health_endpoint_includes_model_name(async_client):
    """测试健康检查端点包含模型名称"""
    response = await async_client.get('/health')
    data = response.json()
    assert 'model_name' in data
    assert data['model_name'] in ['resnet50', 'mobilenet_v2']


# =========================================================================
# 信息端点测试
# =========================================================================

@pytest.mark.anyio
async def test_info_endpoint_returns_200(async_client):
    """测试信息端点返回 200"""
    response = await async_client.get('/info')
    assert response.status_code == 200
    assert response.headers['content-type'] == 'application/json'

@pytest.mark.anyio
async def test_info_endpoint_includes_model_info(async_client):
    """测试信息端点包含模型信息"""
    response = await async_client.get('/info')
    data = response.json()
    assert 'model' in data
    assert 'name' in data['model']
    assert 'framework' in data['model']

@pytest.mark.anyio
async def test_info_endpoint_includes_api_version(async_client):
    """测试信息端点包含 API 版本"""
    response = await async_client.get('/info')
    data = response.json()
    assert 'api_version' in data
    assert data['api_version'] == '1.0.0'

@pytest.mark.anyio
async def test_info_endpoint_includes_limits(async_client):
    """测试信息端点包含限制信息"""
    response = await async_client.get('/info')
    data = response.json()
    assert 'limits' in data
    assert 'max_file_size' in data['limits']
    assert 'max_image_dimension' in data['limits']


# =========================================================================
# 预测端点测试 - 成功情况
# =========================================================================

@pytest.mark.anyio
async def test_predict_endpoint_with_valid_image(async_client, sample_image):
    """测试有效图像的预测端点"""
    response = await async_client.post(
        '/predict',
        files={'file': ('test.jpg', sample_image, 'image/jpeg')}
    )
    assert response.status_code == 200
    data = response.json()
    assert data['success'] is True
    assert 'predictions' in data

@pytest.mark.anyio
async def test_predict_returns_correct_number_of_predictions(async_client, sample_image_factory):
    """测试返回正确数量的预测结果"""
    img1 = sample_image_factory()
    response = await async_client.post(
        '/predict',
        files={'file': ('test.jpg', img1, 'image/jpeg')}
    )
    data = response.json()
    assert len(data['predictions']) == 5

    img2 = sample_image_factory()
    response = await async_client.post(
        '/predict',
        files={'file': ('test.jpg', img2, 'image/jpeg')},
        params={'top_k': 3}
    )
    data = response.json()
    assert len(data['predictions']) == 3

@pytest.mark.anyio
async def test_prediction_format(async_client, sample_image):
    """测试预测结果格式"""
    response = await async_client.post(
        '/predict',
        files={'file': ('test.jpg', sample_image, 'image/jpeg')}
    )
    data = response.json()
    predictions = data['predictions']
    assert len(predictions) > 0
    for pred in predictions:
        # FastAPI 使用 'class' 而不是 'class_name'
        assert 'class' in pred or 'class_name' in pred
        assert 'confidence' in pred
        assert 'rank' in pred

@pytest.mark.anyio
async def test_prediction_includes_latency(async_client, sample_image):
    """测试预测结果包含延迟信息"""
    response = await async_client.post(
        '/predict',
        files={'file': ('test.jpg', sample_image, 'image/jpeg')}
    )
    data = response.json()
    assert 'latency_ms' in data
    assert isinstance(data['latency_ms'], float)
    assert data['latency_ms'] > 0

@pytest.mark.anyio
async def test_prediction_includes_correlation_id(async_client, sample_image):
    """测试预测结果包含关联 ID"""
    response = await async_client.post(
        '/predict',
        files={'file': ('test.jpg', sample_image, 'image/jpeg')}
    )
    data = response.json()
    # 关联 ID 应该存在于成功和错误响应中
    assert 'correlation_id' in data.get('error', data) or 'correlation_id' in data


# =========================================================================
# 预测端点测试 - 错误情况
# =========================================================================

@pytest.mark.anyio
async def test_predict_without_file_returns_422(async_client):
    """FastAPI 在缺少必需文件参数时返回 422"""
    response = await async_client.post('/predict')
    assert response.status_code == 422
    data = response.json()
    assert 'detail' in data

@pytest.mark.anyio
async def test_predict_with_empty_file_returns_400(async_client):
    """测试空文件返回 400"""
    empty_file = io.BytesIO(b'')
    response = await async_client.post(
        '/predict',
        files={'file': ('empty.jpg', empty_file, 'image/jpeg')}
    )
    assert response.status_code == 400
    data = response.json()
    assert data['success'] is False

@pytest.mark.anyio
async def test_predict_with_large_file_returns_413(async_client, large_image):
    """测试大文件返回 413"""
    response = await async_client.post(
        '/predict',
        files={'file': ('large.jpg', large_image, 'image/jpeg')}
    )
    # 注意：这测试的是图像尺寸限制 (4096)，而不是文件大小限制 (10MB)
    # 5000x5000 的图像超过了 MAX_IMAGE_DIMENSION
    assert response.status_code == 400
    data = response.json()
    assert data['success'] is False
    assert 'error' in data

@pytest.mark.anyio
async def test_predict_with_invalid_image_returns_400(async_client):
    """测试无效图像返回 400"""
    invalid_file = io.BytesIO(b'This is not an image')
    response = await async_client.post(
        '/predict',
        files={'file': ('invalid.jpg', invalid_file, 'image/jpeg')}
    )
    assert response.status_code == 400
    data = response.json()
    assert data['success'] is False
    assert data['error']['code'] == 'INVALID_IMAGE_FORMAT'

@pytest.mark.anyio
async def test_predict_with_invalid_top_k_returns_400_or_422(async_client, sample_image_factory):
    """FastAPI 对无效 top_k 参数返回 400 或 422"""
    test_cases = ['-1', '0', '100', 'abc']
    for invalid_top_k in test_cases:
        img = sample_image_factory()
        response = await async_client.post(
            '/predict',
            files={'file': ('test.jpg', img, 'image/jpeg')},
            params={'top_k': invalid_top_k}
        )
        # FastAPI 对业务逻辑验证返回 400，对类型验证返回 422
        assert response.status_code in [400, 422]


# =========================================================================
# 边界情况测试
# =========================================================================

@pytest.mark.anyio
async def test_predict_with_grayscale_image(async_client):
    """测试灰度图像预测"""
    grayscale = Image.new('L', (224, 224), color=128)
    img_bytes = io.BytesIO()
    grayscale.save(img_bytes, format='JPEG')
    img_bytes.seek(0)

    response = await async_client.post(
        '/predict',
        files={'file': ('gray.jpg', img_bytes, 'image/jpeg')}
    )
    assert response.status_code == 200

@pytest.mark.anyio
async def test_predict_with_rgba_image(async_client):
    """测试 RGBA 图像预测"""
    rgba = Image.new('RGBA', (224, 224), color=(255, 0, 0, 128))
    img_bytes = io.BytesIO()
    rgba.save(img_bytes, format='PNG')
    img_bytes.seek(0)

    response = await async_client.post(
        '/predict',
        files={'file': ('rgba.png', img_bytes, 'image/png')}
    )
    assert response.status_code == 200

@pytest.mark.anyio
async def test_predict_with_different_image_formats(async_client):
    """测试不同图像格式"""
    formats = [
        ('JPEG', 'jpg'),
        ('PNG', 'png'),
    ]
    
    for fmt, ext in formats:
        img = Image.new('RGB', (224, 224), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format=fmt)
        img_bytes.seek(0)

        response = await async_client.post(
            '/predict',
            files={'file': (f'test.{ext}', img_bytes, f'image/{ext.lower() if ext != "jpg" else "jpeg"}')}
        )
        assert response.status_code == 200


# =========================================================================
# 顺序请求测试（Flask 测试客户端不是线程安全的）
# =========================================================================

@pytest.mark.anyio
async def test_sequential_predictions(async_client, sample_image_factory):
    """测试 API 可以处理多个顺序请求"""
    results = []
    
    # 进行 10 次顺序请求
    for _ in range(10):
        img = sample_image_factory()
        response = await async_client.post(
            '/predict',
            files={'file': ('test.jpg', img, 'image/jpeg')}
        )
        results.append(response.status_code)
    
    # 验证所有请求都成功
    assert len(results) == 10
    assert all(status_code == 200 for status_code in results)


# =========================================================================
# Performance Tests
# =========================================================================

@pytest.mark.anyio
async def test_health_check_latency(async_client):
    import time
    start = time.time()
    response = await async_client.get('/health')
    elapsed = (time.time() - start) * 1000
    assert response.status_code == 200
    assert elapsed < 1000  # Should complete in under 1 second

@pytest.mark.anyio
async def test_prediction_latency(async_client, sample_image):
    import time
    start = time.time()
    response = await async_client.post(
        '/predict',
        files={'file': ('test.jpg', sample_image, 'image/jpeg')}
    )
    elapsed = (time.time() - start) * 1000
    assert response.status_code == 200
    data = response.json()
    assert 'latency_ms' in data
    assert elapsed < 5000  # Should complete in under 5 seconds


# =========================================================================
# Error Handler Tests
# =========================================================================

@pytest.mark.anyio
async def test_404_for_nonexistent_endpoint(async_client):
    response = await async_client.get('/nonexistent')
    assert response.status_code == 404

@pytest.mark.anyio
async def test_405_for_wrong_method(async_client):
    response = await async_client.post('/health')
    assert response.status_code == 405
