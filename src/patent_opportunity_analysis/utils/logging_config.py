# src/utils/logging_config.py
"""
统一日志配置文件
提供项目级别的日志配置
"""

from pathlib import Path
from loguru import logger
import sys
from typing import Optional

def setup_project_logging(
    log_dir: Optional[Path] = None,
    log_level: str = "INFO",
    rotation: str = "10 MB",
    retention: str = "7 days",
    enable_console: bool = True,
    enable_file: bool = True
):
    """
    配置项目级别的日志系统
    
    Args:
        log_dir: 日志文件目录，如果为None则使用项目根目录下的logs文件夹
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        rotation: 日志文件轮转大小
        retention: 日志文件保留时间
        enable_console: 是否启用控制台输出
        enable_file: 是否启用文件输出
    """
    # 移除默认的handler
    logger.remove()
    
    # 控制台输出（带颜色）
    if enable_console:
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=log_level,
            colorize=True
        )
    
    # 文件输出
    if enable_file:
        if log_dir is None:
            log_dir = Path(__file__).resolve().parent.parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # 详细日志文件（包含DEBUG）
        debug_log_file = log_dir / "pipeline_debug_{time:YYYY-MM-DD}.log"
        logger.add(
            debug_log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG",
            rotation=rotation,
            retention=retention,
            encoding="utf-8"
        )
        
        # 信息日志文件（INFO及以上）
        info_log_file = log_dir / "pipeline_{time:YYYY-MM-DD}.log"
        logger.add(
            info_log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=log_level,
            rotation=rotation,
            retention=retention,
            encoding="utf-8"
        )
    
    logger.info("=" * 60)
    logger.info("日志系统已初始化")
    logger.info(f"日志级别: {log_level}")
    if enable_file:
        logger.info(f"日志目录: {log_dir}")
    logger.info("=" * 60)
    
    return logger

# 模块级别的日志装饰器
def log_function_call(func):
    """装饰器：自动记录函数调用、耗时和异常"""
    from functools import wraps
    import time
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_name = f"{func.__module__}.{func.__name__}"
        logger.debug(f"🔵 开始执行: {func_name}")
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.debug(f"✅ 完成执行: {func_name} (耗时: {elapsed:.2f}秒)")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"❌ 执行失败: {func_name} (耗时: {elapsed:.2f}秒) - {type(e).__name__}: {e}")
            raise
    
    return wrapper

