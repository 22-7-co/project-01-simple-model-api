'''
Config Management Module
Author: 22-7
License: MIT
'''

import json
import os
from typing import Optional
from dotenv import load_dotenv

class Config:

    def __init__(self):
        # print(os.path.join(os.path.dirname(__file__), '../.env'))
        load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../.env'))
        self.MODEL_NAME: str = os.getenv('MODEL_NAME')
        self.MODEL_PATH: str = os.getenv('MODEL_PATH')
        self.DEVICE: str = os.getenv('DEVICE')
        self.HOST: str = os.getenv('HOST')
        self.Port: int = int(os.getenv('PORT'))
        self.DEBUG: bool = bool(os.getenv('DEBUG'))
        self.API_VERSION: str = os.getenv('API_VERSION')
        self.MAX_FILE_SIZE: int = int(os.getenv('MAX_FILE_SIZE'))
        self.MAX_IMAGE_DIMENSION: int = int(os.getenv('MAX_IMAGE_DIMENSION'))
        self.REQUEST_TIMEOUT: int = int(os.getenv('REQUEST_TIMEOUT'))
        self.DEFAULT_TOP_K: int = int(os.getenv('DEFAULT_TOP_K'))
        self.MAX_TOP_K: int = int(os.getenv('MAX_TOP_K'))
        self.LOG_LEVEL: str = os.getenv('LOG_LEVEL')
        self.LOG_FORMAT: str = os.getenv('LOG_FORMAT')
        self.IMAGE_LABELS_URL: str = os.getenv('IMAGE_LABELS_URL')

    @classmethod
    def validate(cls) -> bool:
        if not cls.MODEL_NAME:
            return False
        if not cls.MODEL_PATH:
            return False
        if not cls.DEVICE:
            return False
        if not cls.HOST:
            return False
        if not cls.Port:
            return False
        return True
    
    @classmethod # 不用实例类可以直接调用
    def to_dict(cls) -> dict:
        return {
            'MODEL_NAME': cls.MODEL_NAME,
            'MODEL_PATH': cls.MODEL_PATH,
            'DEVICE': cls.DEVICE,
            'HOST': cls.HOST,
            'Port': cls.Port,
            'DEBUG': cls.DEBUG,
            'API_VERSION': cls.API_VERSION,
            'MAX_FILE_SIZE': cls.MAX_FILE_SIZE,
            'NAX_IMAGE_DIMENSION': cls.NAX_IMAGE_DIMENSION,
            'REQUEST_TIMEOUT': cls.REQUEST_TIMEOUT,
            'DEFAULT_TOP_K': cls.DEFAULT_TOP_K,
            'MAX_TOP_K': cls.MAX_TOP_K,
            'LOG_LEVEL': cls.LOG_LEVEL,
            'LOG_FORMAT': cls.LOG_FORMAT,
            'IMAGE_LABELS_URL': cls.IMAGE_LABELS_URL
        }
    
    def __repr__(self):
        '''
        打印显示的内容
        '''
        return json.dumps(self.to_dict(), indent=4, ensure_ascii=False, sort_keys=False)

def get_env_bool(key: str, default: bool = False) -> bool:
    """
    从环境变量读取布尔值。

    TODO: 实现布尔转换
    - 获取环境变量的值
    - 转换为布尔类型
    - 未设置则返回默认值
    - 'true', '1', 'yes', 'on' 视为 True（不区分大小写）
    - 其它值均视为 False
    """
    return os.environ[key] == bool

def get_env_int(key: str, default: int) -> int:
    """
    从环境变量读取整数。

    TODO: 实现整数转换
    - 获取环境变量的值
    - 转换为整数
    - 未设置或非法时返回默认值
    - 对非数字值需要处理 ValueError
    """
    return os.environ[key] == default

if __name__ == "__main__":
    '''
    测试 配置 加载
    '''
    config = Config()
    print(f'config.validate():{config.validate()}')
    print(config)