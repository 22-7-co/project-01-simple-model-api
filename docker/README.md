# Docker 部署指南

本目录包含项目的 Docker 配置文件和部署脚本。

## 📁 文件说明

### Dockerfile - Flask 标准版本
用于构建 Flask 应用的 Docker 镜像。

**特点：**
- 基于 Python 3.11-slim-bookworm
- 预下载 ResNet-50 模型权重
- 安装必要的系统库（libgomp1, libglib2.0-0 等）
- 非 root 用户运行（安全性）
- 健康检查配置

**镜像大小：** ~9GB

**构建命令：**
```bash
docker build -f docker/Dockerfile -t model-api:v1.0 .
```

### Dockerfile.promax - FastAPI 优化版本
使用多阶段构建优化的 FastAPI 应用镜像。

**优化策略：**
1. **多阶段构建**
   - Builder 阶段：安装依赖和下载模型
   - Runtime 阶段：只复制必要的文件

2. **模型预加载**
   - 在构建时下载模型权重
   - 避免容器启动时下载延迟

3. **依赖优化**
   - 使用系统 site-packages
   - 正确的权限设置

4. **镜像清理**
   - 清理 apt 缓存
   - 使用 slim 基础镜像

**镜像大小：** ~8.8GB

**构建命令：**
```bash
docker build -f docker/Dockerfile.promax -t model-api:v1.0.promax .
```

### docker-compose.yml - Docker Compose 配置
用于本地开发和测试的容器编排配置。

**功能：**
- 服务定义和资源限制
- 环境变量配置
- 卷挂载（代码、日志、模型缓存）
- 健康检查
- 日志轮转
- 网络配置

**启动服务：**
```bash
cd docker
docker-compose up -d
```

**查看日志：**
```bash
docker-compose logs -f
```

**停止服务：**
```bash
docker-compose down
```

## 🚀 快速开始

### 方法一：使用 Makefile

```bash
# 构建 Flask 镜像
make build

# 构建 FastAPI 镜像（优化版）
make build-promax

# 运行 Flask 容器
make run

# 运行 FastAPI 容器
make run-fastapi

# 停止所有容器
make stop

# 清理所有 Docker 资源
make clean
```

### 方法二：直接使用 Docker 命令

#### 1. 构建镜像

**Flask 版本：**
```bash
docker build -f docker/Dockerfile -t model-api:v1.0 .
```

**FastAPI 版本（推荐）：**
```bash
docker build -f docker/Dockerfile.promax -t model-api:v1.0.promax .
```

#### 2. 运行容器

**基本运行：**
```bash
# Flask
docker run -d -p 5000:5000 --name model-api-container model-api:v1.0

# FastAPI
docker run -d -p 5000:5000 --name model-api-fastapi-container model-api:v1.0.promax
```

**带环境变量：**
```bash
docker run -d -p 5000:5000 \
  -e MODEL_NAME=resnet50 \
  -e DEVICE=cpu \
  -e LOG_LEVEL=INFO \
  --name model-api-container \
  model-api:v1.0
```

**挂载本地目录：**
```bash
docker run -d -p 5000:5000 \
  -v $(pwd)/logs:/app/logs \
  -v ~/.cache/torch:/home/apiuser/.cache/torch \
  --name model-api-container \
  model-api:v1.0
```

#### 3. 测试服务

```bash
# 健康检查
curl http://localhost:5000/health

# 获取模型信息
curl http://localhost:5000/info

# 上传测试图片
curl -X POST http://localhost:5000/predict \
  -F "file=@test.jpg" \
  -F "top_k=5"
```

#### 4. 查看日志

```bash
# 实时日志
docker logs -f model-api-container

# 最近 100 行
docker logs --tail 100 model-api-container

# 带时间戳
docker logs -f -t model-api-container
```

#### 5. 进入容器调试

```bash
docker exec -it model-api-container /bin/bash
```

## ⚙️ 配置选项

### 环境变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| MODEL_NAME | resnet50 | 模型名称 |
| DEVICE | cpu | 推理设备（cpu/cuda/mps） |
| HOST | 0.0.0.0 | 服务器绑定地址 |
| PORT | 5000 | 服务器端口 |
| LOG_LEVEL | INFO | 日志级别 |
| DEBUG | false | 调试模式 |

### 资源限制

在 docker-compose.yml 中配置：

```yaml
deploy:
  resources:
    limits:
      cpus: '2.0'      # CPU 限制
      memory: 4G       # 内存限制
    reservations:
      cpus: '1.0'
      memory: 2G
```

## 🔧 高级配置

### GPU 支持

使用 NVIDIA GPU 进行加速：

**前提条件：**
- NVIDIA GPU
- NVIDIA Driver
- NVIDIA Container Toolkit

**运行命令：**
```bash
docker run -d -p 5000:5000 \
  --gpus all \
  -e DEVICE=cuda \
  --name model-api-gpu \
  model-api:v1.0.promax
```

**docker-compose 配置：**
```yaml
services:
  model-api:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### 生产环境部署

#### 1. 使用 Docker Swarm

```bash
# 初始化 Swarm
docker swarm init

# 部署服务
docker service create \
  --name model-api \
  --replicas 3 \
  -p 5000:5000 \
  --env MODEL_NAME=resnet50 \
  --env DEVICE=cpu \
  model-api:v1.0.promax
```

#### 2. 使用 Kubernetes

创建 deployment.yaml：
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: model-api
  template:
    metadata:
      labels:
        app: model-api
    spec:
      containers:
      - name: model-api
        image: model-api:v1.0.promax
        ports:
        - containerPort: 5000
        env:
        - name: MODEL_NAME
          value: "resnet50"
        - name: DEVICE
          value: "cpu"
        resources:
          limits:
            memory: "4Gi"
            cpu: "2"
          requests:
            memory: "2Gi"
            cpu: "1"
        livenessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 30
          periodSeconds: 10
```

应用配置：
```bash
kubectl apply -f deployment.yaml
```

## 🐛 故障排查

### 容器无法启动

**检查日志：**
```bash
docker logs model-api-container
```

**常见错误：**
1. **端口被占用**
   ```bash
   # 查看端口占用
   lsof -i :5000
   # 使用不同端口
   docker run -p 5001:5000 ...
   ```

2. **权限问题**
   ```bash
   # 确保 Docker 有权限访问目录
   sudo chown -R $USER:$USER ./logs
   ```

3. **内存不足**
   ```bash
   # 检查系统内存
   free -h
   # 增加 Docker 内存限制
   ```

### 模型下载失败

**使用预构建镜像：**
```bash
docker build -f docker/Dockerfile.promax -t model-api:v1.0.promax .
```

**手动下载模型：**
```bash
# 下载模型到本地
mkdir -p ~/.cache/torch/hub/checkpoints
cd ~/.cache/torch/hub/checkpoints
wget https://download.pytorch.org/models/resnet50-11ad3fa6.pth
```

### 健康检查失败

**检查服务状态：**
```bash
docker inspect --format='{{.State.Health.Status}}' model-api-container
```

**手动测试健康端点：**
```bash
curl http://localhost:5000/health
```

## 📊 性能优化

### 镜像大小优化

1. **使用多阶段构建**（已实现）
2. **使用 slim 基础镜像**（已实现）
3. **清理缓存**（已实现）
4. **减少层数**

查看镜像大小：
```bash
docker images | grep model-api
```

### 启动速度优化

1. **预加载模型**（Dockerfile.promax 已实现）
2. **使用本地卷缓存模型**
   ```bash
   docker run -v ~/.cache/torch:/home/apiuser/.cache/torch ...
   ```

### 运行时优化

1. **使用 GPU 加速**
2. **调整并发设置**
3. **优化内存使用**

## 🔐 安全最佳实践

### 1. 非 root 用户运行

Dockerfile 中已配置：
```dockerfile
RUN useradd -m -u 1000 apiuser
USER apiuser
```

### 2. 只读文件系统

```bash
docker run --read-only \
  --tmpfs /tmp \
  model-api:v1.0
```

### 3. 限制能力

```bash
docker run --cap-drop=ALL \
  --cap-add=NET_BIND_SERVICE \
  model-api:v1.0
```

### 4. 扫描镜像漏洞

```bash
# 使用 Docker Scout
docker scout cve model-api:v1.0

# 使用 Trivy
trivy image model-api:v1.0
```

## 📝 开发指南

### 本地开发挂载代码

```bash
docker run -d -p 5000:5000 \
  -v $(pwd)/src:/app/src:ro \
  -v $(pwd)/logs:/app/logs \
  --name model-api-dev \
  model-api:v1.0
```

### 热重载（开发模式）

使用 Flask 的调试模式：
```bash
docker run -d -p 5000:5000 \
  -e DEBUG=true \
  -v $(pwd)/src:/app/src \
  model-api:v1.0
```

### 构建特定版本

```bash
# 带标签构建
docker build -f docker/Dockerfile \
  -t model-api:v1.0.0 \
  --build-arg VERSION=1.0.0 \
  .
```

## 📚 参考资料

- [Docker 官方文档](https://docs.docker.com/)
- [Docker Compose 文档](https://docs.docker.com/compose/)
- [Dockerfile 最佳实践](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)
- [多阶段构建](https://docs.docker.com/build/building/multi-stage/)
- [Docker 网络](https://docs.docker.com/network/)
- [Docker 存储卷](https://docs.docker.com/storage/volumes/)
