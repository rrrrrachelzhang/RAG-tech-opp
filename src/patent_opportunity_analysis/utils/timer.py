# src/utils/timer.py
"""
时间统计模块
用于统计各模块的运行时间
"""

import time
from contextlib import contextmanager
from typing import Dict, Optional
from loguru import logger

class Timer:
    """时间统计器"""
    
    def __init__(self):
        self.timings: Dict[str, float] = {}
        self.start_times: Dict[str, float] = {}
    
    @contextmanager
    def time_block(self, name: str):
        """
        上下文管理器，用于统计代码块执行时间
        
        Usage:
            with timer.time_block("data_loading"):
                # 代码块
                pass
        """
        start = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start
            self.timings[name] = elapsed
            logger.info(f"⏱️  [{name}] 耗时: {elapsed:.2f} 秒")
    
    def start(self, name: str):
        """开始计时"""
        self.start_times[name] = time.time()
    
    def stop(self, name: str) -> float:
        """停止计时并返回耗时"""
        if name not in self.start_times:
            logger.warning(f"未找到计时器: {name}")
            return 0.0
        
        elapsed = time.time() - self.start_times[name]
        if name in self.timings:
            self.timings[name] += elapsed
        else:
            self.timings[name] = elapsed
        
        del self.start_times[name]
        logger.info(f"⏱️  [{name}] 耗时: {elapsed:.2f} 秒")
        return elapsed
    
    def get_timings(self) -> Dict[str, float]:
        """获取所有计时结果"""
        return self.timings.copy()
    
    def get_total_time(self) -> float:
        """获取总耗时"""
        return sum(self.timings.values())
    
    def print_summary(self):
        """打印时间统计摘要"""
        logger.info("=" * 60)
        logger.info("⏱️  运行时间统计摘要")
        logger.info("=" * 60)
        total = self.get_total_time()
        for name, elapsed in sorted(self.timings.items(), key=lambda x: x[1], reverse=True):
            percentage = (elapsed / total * 100) if total > 0 else 0
            logger.info(f"  {name:30s}: {elapsed:8.2f} 秒 ({percentage:5.1f}%)")
        logger.info("-" * 60)
        logger.info(f"  总计: {total:8.2f} 秒")
        logger.info("=" * 60)

# 全局计时器实例
global_timer = Timer()

