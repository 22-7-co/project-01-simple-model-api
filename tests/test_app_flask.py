import pytest
import io
from PIL import Image

# Flask 测试导入
from src.app import app, init_model

# 初始化 Flask 测试用的模型
init_model()


# =========================================================================
# 测试夹具
# =========================================================================

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def sample_image():
    image = Image.new('RGB', (224, 224), color='red')
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    return img_byte_arr

@pytest.fixture
def sample_image_factory():
    def create_sample_image():
        image = Image.new('RGB', (224, 224), color='red')
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        return img_byte_arr
    return create_sample_image

@pytest.fixture
def large_image():
    image = Image.new('RGB', (5000, 5000), color='red')
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG', quality=100)
    img_byte_arr.seek(0)
    return img_byte_arr


# =========================================================================
# 健康检查测试
# =========================================================================

def test_health_endpoint_returns_200(client):
    """测试健康检查端点返回 200 状态码"""
    response = client.get('/health')
    assert response.status_code == 200
    assert response.is_json

def test_health_endpoint_returns_healthy_status(client):
    """测试健康检查端点返回健康状态"""
    response = client.get('/health')
    data = response.get_json()
    assert data['status'] == 'healthy'
    assert data['model_loaded'] is True

def  test_health_endpoint_includes_model_name(client):
    """测试健康检查端点包含模型名称"""
    response = client.get('/health')
    data = response.get_json()
    assert 'model_name' in data
    assert data['model_name'] in ['resnet50', 'mobilenet_v2']


# =========================================================================
# 信息端点测试
# =========================================================================


def test_info_endpoint_returns_200(client):
    """测试信息端点返回 200 状态码"""
    response = client.get('/info')
    assert response.status_code == 200
    assert response.is_json

def test_info_endpoint_includes_model_info(client):
    """测试信息端点包含模型信息"""
    response = client.get('/info')
    data = response.get_json()
    assert 'model' in data
    assert 'name' in data['model']
    assert 'framework' in data['model']


def test_info_endpoint_includes_api_version(client):
    """测试信息端点包含 API 版本"""
    response = client.get('/info')
    data = response.get_json()
    assert 'api_version' in data
    assert data['api_version'] == '1.0.0'

def test_info_endpoint_includes_limits(client):
    """测试信息端点包含限制信息"""
    response = client.get('/info')
    data = response.get_json()
    assert 'limits' in data
    assert 'max_file_size' in data['limits']
    assert 'max_image_dimension' in data['limits']

# =========================================================================
# 预测端点测试 - 成功情况
# =========================================================================

def test_predict_endpoint_with_valid_image(client, sample_image):
    """测试有效图像的预测端点"""
    response = client.post(
        '/predict',
        data = {
            'file':(sample_image, 'test.jpg')},
            content_type='multipart/form-data'
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data['success'] is True
    assert 'predictions' in data


def test_predict_returns_correct_number_of_predictions(client, sample_image_factory):
    """测试返回正确数量的预测结果"""
    img1 = sample_image_factory()
    response = client.post(
        '/predict',
        data = {
            'file':(img1, 'test.jpg')},
    )
    data = response.get_json()
    assert len(data['predictions']) == 5

    img2 = sample_image_factory()
    response = client.post(
        '/predict',
        data = {
            'file':(img2, 'test.jpg'),
            'top_k': 3
        },
    )
    data = response.get_json()
    assert len(data['predictions']) == 3

def test_prediction_format(client, sample_image):
    """测试预测结果格式"""
    response = client.post(
        '/predict',
        data = {
            'file':(sample_image, 'test.jpg')},
    )
    data = response.get_json()
    predictions = data['predictions']

    for i, pred in enumerate(predictions, start=1):
        assert 'class' in pred
        assert 'confidence' in pred
        assert 'rank' in pred
        assert 0 <= pred['confidence'] <= 1
        assert pred['rank'] == i

def test_prediction_includes_latency(client, sample_image):
    """测试预测结果包含延迟信息"""
    response = client.post(
        '/predict',
        data = {
            'file':(sample_image, 'test.jpg')},
    )
    data = response.get_json()
    assert 'latency_ms' in data
    assert isinstance(data['latency_ms'], (int, float))
    assert data['latency_ms'] >= 0

def test_prediction_includes_correlation_id(client, sample_image):
    """测试预测结果包含关联 ID"""
    response = client.post(
        '/predict',
        data = {
            'file':(sample_image, 'test.jpg')},
    )
    data = response.get_json()
    assert 'correlation_id' in data
    assert isinstance(data['correlation_id'], str)
    assert len(data['correlation_id']) > 0


# =========================================================================
# 预测端点测试 - 错误情况
# =========================================================================

def test_predict_without_file_returns_400(client):
    """测试未提供文件时返回 400"""
    response = client.post('/predict', data={})
    assert response.status_code == 400
    data = response.get_json()
    assert data['success'] is False
    assert data['error']['code'] == 'MISSING_FILE'

def test_predict_with_empty_file_returns_400(client):
    """测试空文件时返回 400"""
    empty_file =  io.BytesIO()
    empty_file = io.BytesIO(b'')
    response = client.post(
        '/predict',
        data = {
            'file':(empty_file, 'empty.jpg')},
            content_type='multipart/form-data'
    )
    assert response.status_code == 400

def test_predict_with_large_file_returns_413(client, large_image):
    """测试大文件返回 413"""
    response = client.post(
        '/predict',
        data = {
            'file':(large_image, 'large.jpg')},
            content_type='multipart/form-data'
    )
    # 注意：这测试的是图像尺寸限制 (4096)，而不是文件大小限制 (10MB)
    # 5000x5000 的图像超过了 MAX_IMAGE_DIMENSION
    assert response.status_code == 400
    data = response.get_json()
    assert data['success'] is False
    assert 'error' in data

def test_predict_with_invalid_image_returns_400(client):
    """测试无效图像返回 400"""
    invalid_file = io.BytesIO(b'This is not an image')
    response = client.post(
        '/predict',
        data = {
            'file':(invalid_file, 'fake.jpg')},
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data['error']['code'] == 'INVALID_IMAGE_FORMAT'

def test_predict_with_invalid_top_k_returns_400(client, sample_image_factory):
    """测试无效 top_k 参数返回 400"""
    test_cases = ['-1', '0', '100', 'abc']
    for invalid_top_k in test_cases:
        img = sample_image_factory()
        response = client.post(
            '/predict',
            data = {
                'file':(img, 'test.jpg'),
                'top_k': invalid_top_k
            },
        )
        assert response.status_code == 400

    
# =========================================================================
# 边界情况测试
# =========================================================================

def test_predict_with_grayscale_image(client):
    """测试灰度图像预测"""
    grayscale = Image.new('L', (224, 224), color=128)
    img_bytes = io.BytesIO()
    grayscale.save(img_bytes, format='JPEG')
    img_bytes.seek(0)

    response = client.post(
        '/predict',
        data = {
            'file':(img_bytes, 'gray.jpg')},
    )
    assert response.status_code == 200

def test_predict_with_rgba_image(client):
    """测试 RGBA 图像预测"""
    rgba = Image.new('RGBA', (224, 224), color=(255, 0, 0, 128))
    img_bytes = io.BytesIO()
    rgba.save(img_bytes, format='PNG')
    img_bytes.seek(0)

    response = client.post(
        '/predict',
        data = {
            'file':(img_bytes, 'rgba.png')},
    )
    assert response.status_code == 200


def test_predict_with_different_image_formats(client):
    """测试不同图像格式"""
    formats = ['JPEG', 'PNG', 'BMP']
    for fmt in formats:
        image = Image.new('RGB', (224, 224), color='blue')
        img_bytes = io.BytesIO()
        image.save(img_bytes, format=fmt)
        img_bytes.seek(0)

        response = client.post(
            '/predict',
            data = {
                'file':(img_bytes, f'test.{fmt.lower()}')},
        )
        assert response.status_code == 200
    

# =========================================================================
# 并发请求测试
# =========================================================================

def test_concurrent_predictions(client, sample_image_factory):
    """测试 API 可以处理多个顺序请求（注意：Flask 测试客户端不是线程安全的）"""
    results = []
    
    # 进行 10 次顺序请求以测试基本并发处理
    for _ in range(10):
        img = sample_image_factory()
        response = client.post(
            '/predict',
            data = {
                'file':(img, 'test.jpg')},
        )
        results.append(response.status_code)
    
    # 验证所有请求都成功
    assert len(results) == 10
    assert all(status_code == 200 for status_code in results)


# =========================================================================
# 性能测试
# =========================================================================

def test_health_check_latency(client):
    """测试健康检查延迟"""
    import time
    start_time = time.time()
    response = client.get('/health')
    latency_ms = (time.time() - start_time) * 1000
    assert latency_ms < 100
    assert response.status_code == 200

def test_prediction_latency(client, sample_image):
    """测试预测延迟"""
    response = client.post(
        '/predict',
        data = {
            'file':(sample_image, 'test.jpg')},
    )
    data = response.get_json()
    assert data['latency_ms'] < 1000


# =========================================================================
# Error Handler Tests
# =========================================================================

def test_404_for_nonexistent_endpoint(client):
    response = client.get('/nonexistent')
    assert response.status_code == 404

def test_405_for_wrong_method(client):
    response = client.put('/predict')
    assert response.status_code == 405


# =========================================================================
# Run Tests
# =========================================================================

if __name__ == "__main__":
   pytest.main([__file__, '-v'])