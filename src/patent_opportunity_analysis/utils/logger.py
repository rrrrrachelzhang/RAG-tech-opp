# src/utils/logger.py
"""
日志系统模块
使用 loguru 提供统一的日志管理
"""

from pathlib import Path
from loguru import logger
import sys
from typing import Optional

def setup_logger(
    log_dir: Optional[Path] = None,
    log_level: str = "INFO",
    rotation: str = "10 MB",
    retention: str = "7 days"
):
    """
    配置日志系统
    
    Args:
        log_dir: 日志文件目录，如果为None则使用项目根目录下的logs文件夹
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR)
        rotation: 日志文件轮转大小
        retention: 日志文件保留时间
    """
    # 移除默认的handler
    logger.remove()
    
    # 控制台输出（带颜色）
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True
    )
    
    # 文件输出
    if log_dir is None:
        log_dir = Path(__file__).resolve().parent.parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / "pipeline_{time:YYYY-MM-DD}.log"
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=log_level,
        rotation=rotation,
        retention=retention,
        encoding="utf-8"
    )
    
    logger.info(f"日志系统已初始化，日志文件保存在: {log_dir}")
    return logger

# 默认初始化
setup_logger()

