# 模型推理 API 项目

一个基于 Python 的图像分类 REST API 服务，支持 Flask 和 FastAPI 两种框架。

## 📋 项目简介

本项目提供了一个完整的图像分类 API 服务，使用预训练的 ResNet-50 或 MobileNetV2 模型对上传的图像进行识别和分类。支持 Docker 容器化部署，包含完整的测试套件。

## ✨ 主要特性

- **双框架支持**: 同时提供 Flask 和 FastAPI 两种实现
- **预训练模型**: 使用 PyTorch torchvision 的 ResNet-50 或 MobileNetV2
- **RESTful API**: 标准的 REST API 接口，包含健康检查、模型信息和预测端点
- **Docker 部署**: 优化的多阶段 Docker 构建，减小镜像体积
- **完整测试**: 包含单元测试、集成测试和性能测试
- **配置灵活**: 支持环境变量和 .env 文件配置
- **错误处理**: 完善的错误处理和请求验证
- **请求跟踪**: 使用关联 ID 跟踪请求日志

## 🚀 快速开始

### 环境要求

- Python 3.11+
- Docker 和 Docker Compose（可选）
- 至少 4GB 可用内存
- 至少 10GB 可用磁盘空间

### 方法一：本地运行

1. **克隆项目**
```bash
git clone <repository-url>
cd project-01-simple-model-api
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **配置环境变量**
```bash
cp .env.example .env
# 根据需要修改 .env 文件中的配置
```

4. **运行服务**

Flask 版本：
```bash
python src/app.py
```

FastAPI 版本：
```bash
python src/app_fast.py
```

5. **访问服务**
- API 文档（FastAPI）：http://localhost:5000/docs
- 健康检查：http://localhost:5000/health
- 模型信息：http://localhost:5000/info

### 方法二：Docker 运行

1. **构建镜像**

标准版本（Flask）：
```bash
make build
# 或
docker build -f docker/Dockerfile -t model-api:v1.0 .
```

优化版本（FastAPI）：
```bash
make build-promax
# 或
docker build -f docker/Dockerfile.promax -t model-api:v1.0.promax .
```

2. **运行容器**
```bash
# Flask 版本
docker run -d -p 5000:5000 --name model-api-container model-api:v1.0

# FastAPI 版本
docker run -d -p 5000:5000 --name model-api-fastapi-container model-api:v1.0.promax
```

3. **使用 Docker Compose**
```bash
cd docker
docker-compose up -d
```

## 📚 API 文档

### 端点列表

#### 1. GET /health
健康检查端点，验证服务是否正常运行。

**响应示例：**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "resnet50",
  "uptime": 1633024800.123
}
```

#### 2. GET /info
获取模型和 API 的详细信息。

**响应示例：**
```json
{
  "model": {
    "name": "resnet50",
    "framework": "PyTorch",
    "version": "2.7.1",
    "input_shape": [224, 224, 3],
    "output_classes": 1000
  },
  "api_version": "1.0.0",
  "endpoints": ["/predict", "/health", "/info"],
  "limits": {
    "max_file_size": 10485760,
    "max_image_dimension": 4096,
    "timeout_seconds": 30
  }
}
```

#### 3. POST /predict
图像分类预测端点。

**请求参数：**
- `file`: 上传的图像文件（必需）
- `top_k`: 返回前 K 个预测结果（可选，默认 5，范围 1-10）

**请求示例（curl）：**
```bash
curl -X POST http://localhost:5000/predict \
  -F "file=@image.jpg" \
  -F "top_k=5"
```

**成功响应：**
```json
{
  "success": true,
  "predictions": [
    {
      "class": "tabby cat",
      "rank": 1,
      "confidence": 0.8234
    },
    {
      "class": "Egyptian cat",
      "rank": 2,
      "confidence": 0.1234
    }
  ],
  "latency_ms": 45.67,
  "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2024-01-01T12:00:00.000Z"
}
```

**错误响应：**
```json
{
  "success": false,
  "error": {
    "code": "INVALID_IMAGE_FORMAT",
    "message": "无法打开图像：文件格式不支持",
    "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
    "timestamp": "2024-01-01T12:00:00.000Z"
  }
}
```

### 错误代码

| 错误代码 | HTTP 状态码 | 说明 |
|---------|-----------|------|
| MISSING_FILE | 400 | 请求中未提供文件 |
| EMPTY_FILENAME | 400 | 文件名为空 |
| FILE_TOO_LARGE | 413 | 文件大小超过限制 |
| INVALID_IMAGE_FORMAT | 400 | 图像格式无效 |
| INVALID_IMAGE | 400 | 图像验证失败 |
| INVALID_PARAMETER | 400 | 参数无效 |
| PREDICTION_ERROR | 500 | 预测过程出错 |

## 🧪 运行测试

### 运行所有测试
```bash
make test
# 或
pytest tests/ -v
```

### 运行特定测试
```bash
# Flask 测试
make test-flask
# 或
pytest tests/test_app_flask.py -v

# FastAPI 测试
make test-fastapi
# 或
pytest tests/test_app_fastapi.py -v

# 模型测试
pytest tests/test_model.py -v
```

## ⚙️ 配置选项

通过环境变量或 `.env` 文件配置：

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| MODEL_NAME | resnet50 | 模型名称（resnet50 或 mobilenet_v2） |
| DEVICE | cpu | 推理设备（cpu、cuda 或 mps） |
| HOST | 0.0.0.0 | 服务器绑定地址 |
| PORT | 5000 | 服务器端口 |
| LOG_LEVEL | INFO | 日志级别 |
| MAX_FILE_SIZE | 10485760 | 最大文件大小（字节） |
| MAX_IMAGE_DIMENSION | 4096 | 最大图像尺寸（像素） |
| DEFAULT_TOP_K | 5 | 默认返回预测数量 |
| MAX_TOP_K | 10 | 最大预测数量 |

## 📁 项目结构

```
project-01-simple-model-api/
├── src/                      # 源代码目录
│   ├── app.py               # Flask 应用实现
│   ├── app_fast.py          # FastAPI 应用实现
│   ├── config.py            # 配置管理
│   └── model_loader.py      # 模型加载和推理
├── tests/                    # 测试文件
│   ├── test_app_flask.py    # Flask 测试
│   ├── test_app_fastapi.py  # FastAPI 测试
│   ├── test_model.py        # 模型测试
│   └── conftest.py          # 测试夹具
├── docker/                   # Docker 配置
│   ├── Dockerfile           # Flask Dockerfile
│   ├── Dockerfile.promax    # FastAPI 优化 Dockerfile
│   └── docker-compose.yml   # Docker Compose 配置
├── .env.example             # 环境变量示例
├── requirements.txt         # Python 依赖
├── Makefile                 # 构建和测试命令
└── README.md                # 项目文档
```

## 🐳 Docker 镜像优化

项目使用多阶段构建来减小镜像体积：

- **标准镜像** (~9GB): 包含 Flask 应用和所有依赖
- **优化镜像** (~8.8GB): 使用 FastAPI，预加载模型到镜像中

优化策略：
1. 多阶段构建减少最终镜像大小
2. 预下载模型权重，避免容器启动时下载
3. 使用 slim 基础镜像
4. 清理 apt 缓存

## 🔧 开发指南

### 添加新模型

1. 在 `src/model_loader.py` 中添加模型加载逻辑
2. 更新 `Config` 类支持新模型名称
3. 添加相应的测试用例

### 添加新端点

1. 在 `src/app.py` 或 `src/app_fast.py` 中添加路由
2. 实现请求验证和错误处理
3. 添加对应的测试用例

## 📝 常见问题

### Q: 为什么容器启动很慢？
A: 首次启动需要下载模型权重（约 100MB）。使用优化版 Dockerfile 可以预加载模型。

### Q: 如何使用 GPU 加速？
A: 设置 `DEVICE=cuda` 并确保 Docker 容器支持 CUDA。需要使用 NVIDIA Container Toolkit。

### Q: 如何自定义模型？
A: 修改 `src/model_loader.py` 中的 `load()` 方法，加载自定义模型权重。

## 📄 许可证

MIT License

## 👥 作者

22-7

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！
