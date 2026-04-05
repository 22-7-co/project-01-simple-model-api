# 源代码目录

本目录包含项目的核心源代码实现。

## 📁 文件说明

### app.py - Flask 应用实现
Flask 框架的 REST API 实现，包含：
- 应用初始化和配置
- 路由定义（/health, /info, /predict）
- 错误处理器
- 请求/响应格式化
- 日志记录

**主要特性：**
- 同步请求处理
- Flask 原生文件上传处理
- 错误处理和验证
- 关联 ID 跟踪

**使用示例：**
```python
from src.app import app

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

### app_fast.py - FastAPI 应用实现
FastAPI 框架的 REST API 实现，包含：
- 异步请求处理
- Pydantic 模型验证
- 自动 API 文档生成
- 生命周期管理（lifespan）

**主要特性：**
- 异步支持（async/await）
- 自动请求验证
- Swagger UI 和 ReDoc 文档
- 更好的性能

**使用示例：**
```python
from src.app_fast import app
import uvicorn

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=5000)
```

### config.py - 配置管理
环境变量和配置管理类。

**功能：**
- 从 .env 文件加载配置
- 环境变量读取和验证
- 类型转换（int, bool, str）
- 配置验证

**配置项：**
```python
MODEL_NAME          # 模型名称
DEVICE             # 推理设备
HOST               # 服务器地址
PORT               # 服务器端口
MAX_FILE_SIZE      # 最大文件大小
MAX_IMAGE_DIMENSION # 最大图像尺寸
DEFAULT_TOP_K      # 默认预测数量
MAX_TOP_K          # 最大预测数量
LOG_LEVEL          # 日志级别
```

**使用示例：**
```python
from src.config import Config

config = Config()
print(f"Model: {config.MODEL_NAME}")
print(f"Device: {config.DEVICE}")
```

### model_loader.py - 模型加载器
管理机器学习模型的生命周期和推理过程。

**核心功能：**
1. **模型加载**
   - 从 torchvision 加载预训练模型
   - 支持 ResNet-50 和 MobileNetV2
   - 设备管理（CPU/GPU/MPS）

2. **图像预处理**
   - 尺寸调整（256x256）
   - 中心裁剪（224x224）
   - 归一化（ImageNet 均值和方差）
   - 格式转换（RGB/灰度/RGBA）

3. **模型推理**
   - Top-K 预测
   - 置信度计算
   - 类别映射

4. **图像验证**
   - 尺寸检查
   - 格式验证
   - 错误处理

**使用示例：**
```python
from src.model_loader import ModelLoader
from PIL import Image

# 初始化和加载
loader = ModelLoader(model_name='resnet50', device='cpu')
loader.load()

# 加载图像
image = Image.open('cat.jpg')

# 验证图像
is_valid, error = loader.validate_image(image)
if is_valid:
    # 生成预测
    predictions = loader.predict(image, top_k=5)
    for pred in predictions:
        print(f"{pred['rank']}: {pred['class']} ({pred['confidence']:.2%})")
```

## 🔧 模块依赖关系

```
app.py / app_fast.py
    ├── config.py (配置管理)
    └── model_loader.py (模型加载和推理)

model_loader.py
    ├── config.py (配置读取)
    ├── torch (PyTorch 框架)
    ├── torchvision (模型和变换)
    ├── PIL (图像处理)
    └── requests (下载标签)
```

## 📝 编码规范

### 命名约定
- 类名：大驼峰命名（如 `ModelLoader`, `Config`）
- 函数名：小写 + 下划线（如 `load_model`, `preprocess_image`）
- 常量：全大写 + 下划线（如 `MAX_FILE_SIZE`）
- 私有方法：双下划线前缀（如 `__load_image_labels`）

### 文档字符串
所有公共函数和类都应包含文档字符串：
```python
def predict(image: Image.Image, top_k: int = 5) -> List[Dict]:
    """
    对输入图片生成 Top-K 预测结果。
    
    参数:
        image: PIL Image 对象
        top_k: 返回前 K 个预测结果
    
    返回:
        预测结果列表，每个结果包含 class, confidence, rank
    """
```

### 错误处理
- 使用具体的异常类型
- 提供有意义的错误消息
- 记录详细的日志
- 返回标准化的错误响应

## 🧪 测试

每个模块都有对应的测试：
- `test_app_flask.py` - Flask 应用测试
- `test_app_fastapi.py` - FastAPI 应用测试
- `test_model.py` - 模型加载器测试

运行测试：
```bash
pytest tests/ -v
```

## 🚀 性能优化

### 模型加载
- 使用预训练权重
- 模型缓存在本地
- 懒加载策略

### 推理优化
- 批量处理（可选）
- GPU 加速（如可用）
- 异步处理（FastAPI）

### 内存管理
- 及时释放不用的张量
- 使用 `torch.no_grad()` 进行推理
- 限制图像尺寸防止内存溢出

## 🔐 安全考虑

1. **文件上传**
   - 限制文件大小（10MB）
   - 限制图像尺寸（4096x4096）
   - 验证文件类型

2. **输入验证**
   - 参数范围检查
   - 类型验证
   - 格式验证

3. **错误处理**
   - 不泄露内部实现细节
   - 使用关联 ID 跟踪错误
   - 记录详细日志

## 📚 扩展指南

### 添加新模型
1. 在 `model_loader.py` 的 `load()` 方法中添加模型加载逻辑
2. 更新 `Config` 类支持新模型名称
3. 添加测试用例

### 添加新端点
1. 在 `app.py` 或 `app_fast.py` 中定义路由
2. 实现业务逻辑
3. 添加错误处理
4. 编写测试

### 自定义预处理
1. 修改 `model_loader.py` 中的 `create_transform()` 方法
2. 添加新的变换步骤
3. 更新测试

## 📖 参考资料

- [Flask 文档](https://flask.palletsprojects.com/)
- [FastAPI 文档](https://fastapi.tiangolo.com/)
- [PyTorch 文档](https://pytorch.org/docs/)
- [torchvision 模型](https://pytorch.org/vision/stable/models.html)
- [PIL 文档](https://pillow.readthedocs.io/)
