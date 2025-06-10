"""
数据加载模块
为了兼容原有代码结构而创建的数据加载类
"""

import numpy as np

class Data:
    """数据加载类，用于兼容原有代码结构"""
    
    def __init__(self, data_source=None):
        """
        初始化数据加载器
        
        Args:
            data_source: 数据源标识
        """
        self.data_source = data_source
        self.data = {}
        
    def load_data(self):
        """加载数据（占位方法）"""
        # 对于无人机环境，不需要外部数据
        # 这里只是为了兼容性
        pass
        
    def get_data(self):
        """获取数据"""
        return self.data
        
    def preprocess_data(self):
        """数据预处理（占位方法）"""
        pass 