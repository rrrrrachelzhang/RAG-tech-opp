# src/utils/cache.py
"""
缓存工具模块
用于缓存NLP处理结果，避免重复计算
"""

from functools import lru_cache
from typing import Dict, Optional
import hashlib
import pickle
from pathlib import Path
from loguru import logger

class NLPCache:
    """NLP处理结果缓存"""
    
    def __init__(self, cache_dir: Optional[Path] = None, max_size: int = 10000):
        """
        初始化缓存
        
        Args:
            cache_dir: 缓存文件目录，如果为None则使用内存缓存
            max_size: 最大缓存条目数（仅用于内存缓存）
        """
        self.cache_dir = cache_dir
        self.max_size = max_size
        self.memory_cache: Dict[str, any] = {}
        
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"NLP缓存目录: {cache_dir}")
        else:
            logger.info("使用内存NLP缓存")
    
    def _get_key(self, text: str, patent_id: str = None) -> str:
        """
        生成缓存键
        
        Args:
            text: 文本内容
            patent_id: 专利ID（如果提供，优先使用）
        """
        if patent_id:
            return f"patent_{patent_id}"
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def get(self, text: str, patent_id: str = None) -> Optional[any]:
        """获取缓存结果"""
        key = self._get_key(text, patent_id)
        
        # 先检查内存缓存
        if key in self.memory_cache:
            return self.memory_cache[key]
        
        # 检查文件缓存
        if self.cache_dir:
            cache_file = self.cache_dir / f"{key}.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        result = pickle.load(f)
                    # 同时存入内存缓存
                    if len(self.memory_cache) < self.max_size:
                        self.memory_cache[key] = result
                    return result
                except Exception as e:
                    logger.warning(f"读取缓存文件失败: {e}")
        
        return None
    
    def set(self, text: str, value: any, patent_id: str = None):
        """设置缓存结果"""
        key = self._get_key(text, patent_id)
        
        # 存入内存缓存
        if len(self.memory_cache) >= self.max_size:
            # 简单的FIFO策略：删除最旧的条目
            oldest_key = next(iter(self.memory_cache))
            del self.memory_cache[oldest_key]
        self.memory_cache[key] = value
        
        # 存入文件缓存
        if self.cache_dir:
            cache_file = self.cache_dir / f"{key}.pkl"
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(value, f)
            except Exception as e:
                logger.warning(f"写入缓存文件失败: {e}")
    def clear(self):
        """清空缓存"""
        self.memory_cache.clear()
        if self.cache_dir:
            for cache_file in self.cache_dir.glob("*.pkl"):
                try:
                    cache_file.unlink()
                except Exception as e:
                    logger.warning(f"删除缓存文件失败: {e}")
        logger.info("缓存已清空")
    
    def get_stats(self) -> Dict[str, int]:
        """获取缓存统计信息"""
        stats = {
            'memory_entries': len(self.memory_cache),
            'file_entries': 0
        }
        if self.cache_dir:
            stats['file_entries'] = len(list(self.cache_dir.glob("*.pkl")))
        return stats

# 全局缓存实例
_global_nlp_cache: Optional[NLPCache] = None

def get_nlp_cache(cache_dir: Optional[Path] = None) -> NLPCache:
    """获取全局NLP缓存实例"""
    global _global_nlp_cache
    if _global_nlp_cache is None:
        if cache_dir is None:
            cache_dir = Path(__file__).resolve().parent.parent.parent / "cache" / "nlp"
        _global_nlp_cache = NLPCache(cache_dir)
    return _global_nlp_cache
