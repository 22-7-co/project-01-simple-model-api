"""
模型测试模块

这个模块包含了模型加载和推理功能的单元测试。

命令: pytest tests/test_model.py

作者: 22-7
许可证: MIT
"""

'''
================================= test session starts =================================
platform linux -- Python 3.11.15, pytest-9.0.2, pluggy-1.5.0 -- /home/22-7/.conda/envs/tf-gpu/bin/python
cachedir: .pytest_cache
rootdir: /home/22-7/Dev/ai-infra-learn/ai-infra-project/junior-engineer/project-01-simple-model-api
configfile: pytest.ini
collected 26 items                                                                    

tests/test_model.py::test_model_loader_initialization PASSED                    [  3%]
tests/test_model.py::test_model_loader_with_invalid_model_name PASSED           [  7%]
tests/test_model.py::test_model_loads_successfully PASSED                       [ 11%]
tests/test_model.py::test_class_labels_loaded PASSED                            [ 15%]
tests/test_model.py::test_model_on_correct_device PASSED                        [ 19%]
tests/test_model.py::test_preprocess_rgb_image PASSED                           [ 23%]
tests/test_model.py::test_preprocess_grayscale_image PASSED                     [ 26%]
tests/test_model.py::test_preprocess_rgba_image PASSED                          [ 30%]
tests/test_model.py::test_preprocess_different_sizes PASSED                     [ 34%]
tests/test_model.py::test_preprocess_none_image_raises_error PASSED             [ 38%]
tests/test_model.py::test_predict_returns_correct_number PASSED                 [ 42%]
tests/test_model.py::test_prediction_format PASSED                              [ 46%]
tests/test_model.py::test_prediction_confidence_valid PASSED                    [ 50%]
tests/test_model.py::test_prediction_ranks_sequential PASSED                    [ 53%]
tests/test_model.py::test_prediction_sorted_by_confidence PASSED                [ 57%]
tests/test_model.py::test_predict_without_loaded_model_raises_error PASSED      [ 61%]
tests/test_model.py::test_prediction_deterministic PASSED                       [ 65%]
tests/test_model.py::test_get_model_info PASSED                                 [ 69%]
tests/test_model.py::test_get_model_info_before_loading PASSED                  [ 73%]
tests/test_model.py::test_validate_image_with_valid_image PASSED                [ 76%]
tests/test_model.py::test_validate_image_with_none PASSED                       [ 80%]
tests/test_model.py::test_validate_image_with_large_dimensions PASSED           [ 84%]
tests/test_model.py::test_prediction_performance PASSED                         [ 88%]
tests/test_model.py::test_preprocessing_performance PASSED                      [ 92%]
tests/test_model.py::test_model_memory_usage PASSED                             [ 96%]
tests/test_model.py::test_full_prediction_pipeline PASSED                       [100%]

================================= 26 passed in 23.78s =================================
'''

import pytest
from PIL import Image
import torch
import numpy as np

from src.model_loader import ModelLoader
from src.config import Config

@pytest.fixture()
def model_loader():
    '''
    为测试创建 ModelLoader 实例。

    测试夹具（fixture）
    - 使用 resnet50 创建 ModelLoader
    - 使用 CPU 设备（测试速度更快）
    - 暂不加载模型（测试用例将根据需要自行加载）

    返回：
        ModelLoader 实例
    '''
    return ModelLoader(model_name="resnet50", device="cpu")



@pytest.fixture()  
def loaded_model_loader(model_loader):
    '''
    创建一个**已完成模型加载**的 ModelLoader 实例。

    测试夹具
    - 依赖已有的 model_loader 夹具
    - 调用其 load() 方法完成模型加载
    - 返回加载完成后的实例

    返回：
        已加载好模型的 ModelLoader 实例
    '''
    model_loader.load()
    return model_loader

@pytest.fixture()
def sample_image():
    '''
    创建用于测试的示例 RGB 图像。

    待实现：测试夹具
    - 创建 RGB 图像 (模式='RGB')
    - 返回 PIL Image 对象

    返回：
        PIL Image 对象（RGB 格式）
    '''
    return Image.new('RGB', (224, 224), color='red')

@pytest.fixture()
def grayscale_image():
    '''
    创建用于测试的灰度图像。

    待实现：测试夹具
    - 创建灰度图像 (模式='L')
    - 返回 PIL Image 对象

    返回：
        PIL Image 对象（灰度格式）
    '''
    return Image.new('L', (224, 224), color=128) 

def test_model_loader_initialization(model_loader):
    '''
    测试 ModelLoader 的初始化。

    测试点：
    - 确保 model_name 和 device 正确设置
    - 确保 model、transform、class_labels 初始化为 None

    参数：
        model_loader: ModelLoader 夹具实例
    '''
    loader = ModelLoader(model_name="resnet50", device="cpu")
    assert model_loader.model_name == "resnet50"
    assert model_loader.device == "cpu"
    assert model_loader.model is None
    assert model_loader.transform is None
    assert model_loader.class_labels is None
    return

def test_model_loader_with_invalid_model_name():
    '''
    测试使用无效模型名称初始化 ModelLoader 时是否抛出 ValueError。

    测试点：
    - 使用不支持的模型名称创建 ModelLoader
    - 调用 load() 方法应抛出 ValueError
    '''
    loader = ModelLoader(model_name="invalid_model", device="cpu")
    with pytest.raises(ValueError):
        loader.load()
    return

def test_model_loads_successfully(model_loader):
    '''
    测试模型是否成功加载。

    测试点：
    - 模型对象不为 None
    - 模型是 torch.nn.Module 的实例
    - 模型不在训练模式（应为评估模式）
    '''
    model_loader.load()
    assert model_loader.model is not None
    assert isinstance(model_loader.model, torch.nn.Module)
    assert not model_loader.model.training
    return

def test_class_labels_loaded(model_loader):
    '''
    测试 ImageNet 标签是否成功加载。

    实现测试
    - 加载模型
    - 断言 class_labels 不为 None
    - 断言 class_labels 是字典类型
    - 断言有 1000 个类别（ImageNet 标准）
    - 断言标签 0 存在
    - 断言标签 999 存在
    '''
    model_loader.load()
    assert model_loader.class_labels is not None
    assert isinstance(model_loader.class_labels, dict)
    assert len(model_loader.class_labels) == 1000
    assert 0 in model_loader.class_labels
    assert 999 in model_loader.class_labels
    return

def test_model_on_correct_device(model_loader):
    """
    测试模型是否在正确的设备上。

    实现测试
    - 加载模型
    - 获取第一个参数的设备
    - 断言与预期设备匹配
    """
    model_loader.load()
    param_device = next(model_loader.model.parameters()).device
    expected_device = torch.device(model_loader.device)
    assert param_device.type == expected_device.type
    pass


# =========================================================================
# 预处理测试
# =========================================================================

def test_preprocess_rgb_image(loaded_model_loader, sample_image):
    """
    测试 RGB 图像预处理。

    实现测试
    - 预处理示例 RGB 图像
    - 断言输出为 torch.Tensor
    - 断言形状为 (1, 3, 224, 224)
    - 断言值已归一化（约 -2 到 2）
    """
    tensor = loaded_model_loader.preprocess(sample_image)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, 224, 224)
    # 检查归一化（归一化后的值应在约 -2 到 2）
    assert tensor.min() >= -3
    assert tensor.max() <= 3
    pass


def test_preprocess_grayscale_image(loaded_model_loader, grayscale_image):
    """
    测试灰度图像预处理（应转换为 RGB）。

    实现测试
    - 预处理灰度图像
    - 断言输出形状为 (1, 3, 224, 224) 而非 (1, 1, 224, 224)
    - 灰度图应转换为 RGB
    """
    tensor = loaded_model_loader.preprocess(grayscale_image)
    assert tensor.shape == (1, 3, 224, 224)
    pass


def test_preprocess_rgba_image(loaded_model_loader):
    """
    测试 RGBA 图像预处理（应转换为 RGB）。

    实现测试
    - 创建 RGBA 图像
    - 预处理
    - 断言输出形状为 (1, 3, 224, 224)
    """
    rgba_image = Image.new('RGBA', (224, 224), color=(255, 0, 0, 128))
    tensor = loaded_model_loader.preprocess(rgba_image)
    assert tensor.shape == (1, 3, 224, 224)
    pass


def test_preprocess_different_sizes(loaded_model_loader):
    """
    测试不同尺寸图像的预处理。

    实现测试
    - 测试 100x100 图像
    - 测试 500x500 图像
    - 测试非正方形图像 (300x200)
    - 所有输出应为 (1, 3, 224, 224)
    """
    sizes = [(100, 100), (500, 500), (300, 200), (1000, 500)]
    for size in sizes:
        image = Image.new('RGB', size, color='blue')
        tensor = loaded_model_loader.preprocess(image)
        assert tensor.shape == (1, 3, 224, 224)
    pass


def test_preprocess_none_image_raises_error(loaded_model_loader):
    """
    测试预处理 None 会抛出 ValueError。

    实现测试
    - 调用 preprocess(None)
    - 断言抛出 ValueError
    """
    with pytest.raises(ValueError):
        loaded_model_loader.preprocess(None)
    pass


# =========================================================================
# 预测测试
# =========================================================================

def test_predict_returns_correct_number(loaded_model_loader, sample_image):
    """
    测试 predict 返回正确数量的预测结果。

    实现测试
    - 使用 top_k=5 调用 predict
    - 断言返回长度为 5 的列表
    - 使用 top_k=10 测试
    - 断言返回长度为 10 的列表
    """
    predictions = loaded_model_loader.predict(sample_image, top_k=5)
    assert len(predictions) == 5
    
    predictions = loaded_model_loader.predict(sample_image, top_k=10)
    assert len(predictions) == 10
    pass


def test_prediction_format(loaded_model_loader, sample_image):
    """
    测试预测输出格式。

    实现测试
    - 进行预测
    - 断言每个预测是字典
    - 断言包含键：'class'、'confidence'、'rank'
    - 断言 class 为字符串
    - 断言 confidence 为浮点数
    - 断言 rank 为整数
    """
    predictions = loaded_model_loader.predict(sample_image, top_k=5)
    
    for pred in predictions:
        assert isinstance(pred, dict)
        assert 'class' in pred
        assert 'confidence' in pred
        assert 'rank' in pred
        assert isinstance(pred['class'], str)
        assert isinstance(pred['confidence'], float)
        assert isinstance(pred['rank'], int)
    pass


def test_prediction_confidence_valid(loaded_model_loader, sample_image):
    """
    测试置信度分数是否为有效的概率。

    实现测试
    - 进行预测
    - 断言每个置信度在 0 到 1 之间
    - 断言置信度总和约为 1（误差范围内）
    """
    predictions = loaded_model_loader.predict(sample_image, top_k=5)
    
    for pred in predictions:
        assert 0 <= pred['confidence'] <= 1
    
    # 注意：Top-K 置信度总和不会是 1，但应该是合理的
    total = sum(p['confidence'] for p in predictions)
    assert 0 < total <= 1
    pass


def test_prediction_ranks_sequential(loaded_model_loader, sample_image):
    """
    测试排名是否连续且正确。

    实现测试
    - 使用 top_k=5 进行预测
    - 断言排名为 [1, 2, 3, 4, 5]
    """
    predictions = loaded_model_loader.predict(sample_image, top_k=5)
    ranks = [p['rank'] for p in predictions]
    assert ranks == [1, 2, 3, 4, 5]
    pass


def test_prediction_sorted_by_confidence(loaded_model_loader, sample_image):
    """
    测试预测是否按置信度排序（降序）。

    实现测试
    - 进行预测
    - 断言置信度为降序排列
    """
    predictions = loaded_model_loader.predict(sample_image, top_k=5)
    confidences = [p['confidence'] for p in predictions]
    assert confidences == sorted(confidences, reverse=True)
    pass


def test_predict_without_loaded_model_raises_error(model_loader, sample_image):
    """
    测试未加载模型进行预测会抛出错误。

    实现测试
    - 不调用 load()
    - 调用 predict()
    - 断言抛出 RuntimeError
    """
    # model_loader is not loaded (no load() call)
    with pytest.raises(RuntimeError):
        model_loader.predict(sample_image)
    pass


def test_prediction_deterministic(loaded_model_loader, sample_image):
    """
    测试预测是否确定性的（相同输入 = 相同输出）。

    实现测试
    - 用同一图像进行两次预测
    - 断言预测结果相同
    """
    pred1 = loaded_model_loader.predict(sample_image, top_k=5)
    pred2 = loaded_model_loader.predict(sample_image, top_k=5)
    
    assert len(pred1) == len(pred2)
    for p1, p2 in zip(pred1, pred2):
        assert p1['class'] == p2['class']
        assert abs(p1['confidence'] - p2['confidence']) < 1e-6
    pass


# =========================================================================
# 模型信息测试
# =========================================================================

def test_get_model_info(loaded_model_loader):
    """
    测试 get_model_info 是否返回正确的信息。

    实现测试
    - 调用 get_model_info()
    - 断言返回字典
    - 断言包含：name、framework、version、input_shape、output_classes
    """
    info = loaded_model_loader.get_model_info()
    assert isinstance(info, dict)
    assert 'name' in info
    assert 'framework' in info
    assert 'version' in info
    assert 'input_shape' in info
    assert 'output_classes' in info
    assert info['name'] == 'resnet50'
    assert info['framework'] == 'PyTorch'
    assert info['input_shape'] == [224, 224, 3]
    assert info['output_classes'] == 1000
    pass


def test_get_model_info_before_loading(model_loader):
    """
    测试未加载模型需情况下的 get_model_info 。

    实现测试
    - 在 load() 之前调用 get_model_info()
    - 仍然应返回信息
    - 断言 'loaded' 键的值为 False
    """
    info = model_loader.get_model_info()
    assert info['loaded'] is False
    pass


# =========================================================================
# 图像验证测试
# =========================================================================

def test_validate_image_with_valid_image(loaded_model_loader, sample_image):
    """
    测试有效图像的校验。

    实现测试
    - 校验示例图像
    - 断言返回 (True, None)
    """
    is_valid, error = loaded_model_loader.validate_image(sample_image)
    assert is_valid is True
    assert error is None
    pass


def test_validate_image_with_none(loaded_model_loader):
    """
    测试当为 None 时的图像校验。

    实现测试
    - 校验 None
    - 断言返回 (False, 错误信息)
    """
    is_valid, error = loaded_model_loader.validate_image(None)
    assert is_valid is False
    assert error is not None
    pass


def test_validate_image_with_large_dimensions(loaded_model_loader):
    """
    测试超大尺寸图像的校验。

    实现测试
    - 创建比 MAX_IMAGE_DIMENSION 更大的图像
    - 校验
    - 断言返回 (False, 错误信息)
    """
    large_image = Image.new('RGB', (15000, 15000), color='red')
    is_valid, error = loaded_model_loader.validate_image(large_image)
    assert is_valid is False
    assert 'too large' in error.lower()
    pass


# =========================================================================
# 性能测试
# =========================================================================

def test_prediction_performance(loaded_model_loader, sample_image):
    """
    测试预测是否在可接受的时间内完成。

    实现测试
    - 测量预测时间
    - 断言在 1 秒内完成（P99 目标）
    """
    import time
    start = time.time()
    loaded_model_loader.predict(sample_image, top_k=5)
    elapsed = time.time() - start
    assert elapsed < 1.0  # 1 second
    pass


def test_preprocessing_performance(loaded_model_loader, sample_image):
    """
    测试预处理是否很快。

    实现测试
    - 测量预处理时间
    - 断言在 100毫秒内完成
    """
    import time
    start = time.time()
    loaded_model_loader.preprocess(sample_image)
    elapsed = time.time() - start
    assert elapsed < 0.1  # 100ms
    pass


# =========================================================================
# 内存测试
# =========================================================================

def test_model_memory_usage(loaded_model_loader):
    """
    测试模型内存使用是否合理。

    实现测试（可选）
    - 加载模型
    - 检查内存使用情况
    - 断言 < 2GB（需求）
    - 此项需要 psutil 或类似工具
    """
    import psutil
    import os
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    assert memory_mb < 2000  # 2GB
    pass


# =========================================================================
# 集成测试
# =========================================================================

def test_full_prediction_pipeline(model_loader):
    """
    从零开始测试完整预测管道。

    实现测试
    - 初始化 ModelLoader
    - 加载模型
    - 创建图像
    - 预处理
    - 预测
    - 断言所有步骤成功
    """
    # 初始化
    loader = ModelLoader(model_name='resnet50', device='cpu')
    
    # 加载
    loader.load()
    assert loader.model is not None
    
    # 创建图像
    image = Image.new('RGB', (500, 500), color='blue')
    
    # 预测
    predictions = loader.predict(image, top_k=5)
    
    # 验证
    assert len(predictions) == 5
    assert all(0 <= p['confidence'] <= 1 for p in predictions)
    return


# =========================================================================
# 运行测试
# =========================================================================

if __name__ == "__main__":
    """
    使用 pytest 运行测试。

    执行：pytest tests/test_model.py -v
    """
    pytest.main([__file__, '-v'])