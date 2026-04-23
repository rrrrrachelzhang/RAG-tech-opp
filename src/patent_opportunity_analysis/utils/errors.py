# src/utils/errors.py
"""
自定义异常类
"""

class PatentAnalysisError(Exception):
    """专利分析基础异常类"""
    pass

class DataLoadingError(PatentAnalysisError):
    """数据加载错误"""
    pass

class DKNBuildError(PatentAnalysisError):
    """DKN构建错误"""
    pass

class FeatureExtractionError(PatentAnalysisError):
    """特征提取错误"""
    pass

class RegressionModelError(PatentAnalysisError):
    """回归模型错误"""
    pass

class ACOSearchError(PatentAnalysisError):
    """ACO搜索错误"""
    pass

