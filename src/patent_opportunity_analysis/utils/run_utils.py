"""
运行工具模块

提供run_id生成、路径管理等工具函数。
"""
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional

from .paths import OUTPUTS_DIR, RUNS_DIR


def generate_run_id(
    data_hash: Optional[str] = None,
    hist_end_year: Optional[int] = None,
    max_year: Optional[int] = None
) -> str:
    """生成运行ID
    
    Args:
        data_hash: 数据哈希（可选）
        hist_end_year: 历史截止年份（可选）
        max_year: 最大年份（可选）
        
    Returns:
        运行ID字符串，格式：YYYYmmdd_HHMMSS[_hash]
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if data_hash or hist_end_year or max_year:
        # 生成短哈希
        hash_parts = []
        if data_hash:
            hash_parts.append(f"d{data_hash[:8]}")
        if hist_end_year:
            hash_parts.append(f"h{hist_end_year}")
        if max_year:
            hash_parts.append(f"m{max_year}")
        
        hash_str = "_".join(hash_parts)
        return f"{timestamp}_{hash_str}"
    
    return timestamp


def get_run_dir(run_id: str, base_dir: Optional[Path] = None) -> Path:
    """获取运行目录路径
    
    Args:
        run_id: 运行ID
        base_dir: 基础目录（默认：outputs/runs/）
        
    Returns:
        运行目录路径
    """
    if base_dir is None:
        base_dir = RUNS_DIR
    
    return base_dir / run_id


def get_step_dir(run_dir: Path, step_name: str) -> Path:
    """获取步骤目录路径
    
    Args:
        run_dir: 运行目录
        step_name: 步骤名称（如 "01_networks"）
        
    Returns:
        步骤目录路径
    """
    return run_dir / step_name


def ensure_run_dirs(run_dir: Path) -> dict:
    """确保运行目录结构存在
    
    Args:
        run_dir: 运行目录
        
    Returns:
        包含各步骤目录路径的字典
    """
    dirs = {
        "run_dir": run_dir,
        "networks_dir": run_dir / "01_networks",
        "regression_dir": run_dir / "02_regression",
        "aco_dir": run_dir / "03_aco",
        "rag_dir": run_dir / "04_rag_reports",
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs
