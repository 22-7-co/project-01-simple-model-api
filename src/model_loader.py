'''
模型加载器模块

本模块用于加载预训练模型、预处理图像，
以及生成预测结果。

作者: 22-7
许可证: MIT
'''
import logging
from pprint import pprint
import sys
from typing import List, Dict, Optional, Tuple
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import requests
from src.config import Config

logger = logging.getLogger(__name__)

class ModelLoader:
    '''
    管理机器学习模型的生命周期与推理过程。
    本类负责处理：
        从 torchvision 加载预训练模型
        图像预处理（尺寸调整、归一化等）
        执行模型推理
        对预测结果进行后处理
        加载 ImageNet 类别标签
    '''

    def __init__(self, model_name:str = "resnet50", device:str = "cpu"):
        '''
        初始化 ModelLoader。

        完善初始化方法
        - 存储 model_name 和 device
        - 初始化 model 为 None(稍后加载)
        - 初始化 transform 为 None
        - 初始化 class_labels 为 None
        - 设置其它需要的成员变量
        '''
        self.config = Config()
        self.model_name = model_name
        self.device = device
        self.model = None
        self.transform = None
        self.class_labels = None
        logger.info(f"Initialized ModelLoader with model: {model_name}, device: {device}")
    
    def load(self) -> None:
        """
        加载模型权重并准备推理。

        实现模型加载
        1. 按照 self.model_name 加载模型
           - 可以使用 torchvision.models.resnet50() 或 mobilenet_v2()
           - 设置 pretrained=True（新版 PyTorch 可以用 weights='DEFAULT'）
        2. 将模型移动到 self.device
        3. 设置为评估模式（model.eval()）
        4. 创建预处理变换流水线
        5. 加载 ImageNet 类别标签
        6. 日志记录加载成功
        """
        if self.model_name == "resnet50":
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif self.model_name == "mobilenet_v2":
            self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        else:
            logger.error(f"Unsupported model name: {self.model_name}")
            raise ValueError(f"Unsupported model name: {self.model_name}")
        self.model.to(self.device)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.transform = self.create_transform()
        self.class_labels = self.__load_image_labels()
        logger.info(f"Model {self.model_name} loaded successfully on {self.device}")

    def create_transform(self) -> transforms.Compose:
        """
        创建图像预处理变换流水线。

        实现预处理流程
        - 缩放到 256x256（再中心裁剪到 224x224）
        - 中心裁剪为 224x224
        - 转为 tensor
        - 按 ImageNet 均值和方差归一化
          均值: [0.485, 0.456, 0.406]
          方差: [0.229, 0.224, 0.225]
        """
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]),
        ])

    def __load_image_labels(self) -> Dict[int, str]:
        """
        加载 ImageNet 类别标签。

        实现标签加载
        - 通过 IMAGENET_LABELS_URL 下载标签文件（用 requests 库）
        - 解析文本文件（每行一个标签）
        - 创建 index 到标签的字典
        - 优雅处理下载失败
        - 可以考虑本地缓存标签
        """
        try:
            url = 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'
            # url = 'https://raw.gitmirror.com/pytorch/hub/master/imagenet_classes.txt'
            response = requests.get(url, timeout=100)
            response.raise_for_status()
            labels = response.text.strip().split('\n')
            return {i: label for i, label in enumerate(labels)}
        except Exception as e:
            logger.error(f"Failed to load ImageNet labels: {e}")
            raise RuntimeError("Could not load class labels")
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """
        对模型输入图像预处理。

        实现预处理
        - 验证 image 非 None
        - 转为 RGB（兼容灰度、RGBA）
        - 应用 transform 流程
        - 增加 batch 维度（unsqueeze）
        - 移到指定设备
        - 返回处理后的 tensor
        """
        if image is None:
            raise ValueError("Input image is None")
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            tensor = self.transform(image)
            tensor = tensor.unsqueeze(0)  # 增加 batch 维度
            tensor = tensor.to(self.device)
            return tensor
        except Exception as e:
            logger.error(f"Error during image preprocessing: {e}")
            raise ValueError(f"Error during image preprocessing: {e}")
        return image

    def predict(self, image: Image.Image, top_k: int = 5) -> List[Dict[str, any]]:
        """
        对输入图片生成 Top-K 预测结果。

        实现预测
        1. 验证模型已经加载
        2. 预处理图像
        3. 运行推理（torch.no_grad()）
        4. 使用 softmax 得到概率
        5. 获取 top-K 预测
        6. 映射索引到类别标签
        7. 以字典列表格式输出
        8. 返回预测结果
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load() before predict().")
        try:
            tensor = self.preprocess(image)
            with torch.no_grad():
                outputs = self.model(tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            top_probs, top_indices = torch.topk(probabilities, top_k)
            predictions = []
            for rank, (prob, idx) in enumerate(zip(top_probs, top_indices), start=1):
                predictions.append({
                    'class': self.class_labels[idx.item()],
                    "rank": rank ,
                    'confidence': float(prob.item())
                })
            return predictions
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise ValueError(f"Failed to generate prediction: {e}")
    def get_model_info(self) -> Dict[str, any]:
        """
        获取模型的元数据和相关信息。

        实现模型信息
        - 返回包含模型元数据的字典
        - 包含内容：name（名称）、framework（框架）、version（版本）、input shape（输入形状）、output classes（类别数）
        - 兼容模型未加载的情况
        """
        return{
            "name": self.model_name,
            "framework": "PyTorch",
            "version": torch.__version__,
            "input_shape": [224, 224, 3],
            "output_classes": 1000,
            "divice": self.device,
            "loaded": self.model is not None
        }
    def validate_image(self, image: Image.Image) -> Tuple[bool, Optional[str]]:
        """
        校验图片是否符合要求。

        实现图片校验
        - 检查 image 不为 None
        - 检查图片尺寸是否合理（小于 MAX_IMAGE_DIMENSION）
        - 检查图片模式是否合法
        - 返回 (is_valid, error_message)
        """
        if image is None:
            return False, "Image is None"
        width, height = image.size
        max_dim = self.config.MAX_IMAGE_DIMENSION
        print("max_dim: ", max_dim)
        if max_dim is None:
            return False, "MAX_IMAGE_DIMENSION is not set in config"
        if width > max_dim or height > max_dim:
                return False, f"Image dimensions too large: {width}x{height}"
        if image.mode not in ['RGB', 'RGBA', 'L', 'R']:
            return False, f"Unsupported image mode: {image.mode}"
        return True, None
    
    def __repr__(self) -> str:
        """
        ModelLoader 的字符串描述。

        实现 __repr__
        - 返回包含模型状态的信息字符串
        - 包括模型名、设备、是否已加载
        """
        loaded = self.model is not None
        return f"ModelLoader(model_name={self.model_name}, device={self.device}, loaded={loaded})"

def load_model_from_url(path: str, model_name:str, device: str = 'cpu') -> nn.Module:
    """
    从自定义路径加载模型（进阶用法）。

    实现自定义模型加载（可选）
    - 从自定义 checkpoint 文件加载模型
    - 适用于加载微调后的模型
    - 本项目基础部分可以不实现
    """
    model = load_model_from_url(path, model_name, device)
    return model

def download_file(url: str, local_path:str, timeout: int = 10) -> None:
    """
    从 URL 下载文件至本地路径。

    实现文件下载（可选）
    - 带进度跟踪下载文件
    - 适用于下载自定义模型或标签
    - 本项目基础部分可以不实现
    """
    success = download_file(url, local_path, timeout)
    if not success:
        raise RuntimeError(f"Failed to download file from {url}")
    return

if __name__ == "__main__":
    """
    测试
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    try:
        # 测试模型加载
        model_loader = ModelLoader(model_name="resnet50", device="cpu")
        print(f"初始化 模型加载器: {model_loader}")
        
        # 加载模型
        print("加载 模型中...")
        model_loader.load()
        print(f"模型{model_loader.model_name}加载成功")
        
        # 测试模型信息
        info = model_loader.get_model_info()
        print('模型信息: ')
        pprint(info,indent=4)

        # 
        if len(sys.argv) > 1:
            image_path = sys.argv[1]
            print(f"测试图片路径: {image_path}")
            image = Image.open(image_path)

            # 验证图片
            is_valid, error_message = model_loader.validate_image(image)
            if not is_valid:
                print(f"图片验证失败: {error_message}")
                sys.exit(1)
            
            # 生成预测
            print("生成预测...")
            predictions = model_loader.predict(image, top_k=5)
            print("预测结果:")
            for pred in predictions:
                print(f"Rank: {pred['rank']}, Label: {pred['label']}, Probability: {pred['probability']:.4f}")
        else:
            print("未提供图片路径用于预测。")
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)