# src/patent_opportunity_analysis/utils/cache_io.py
"""
缓存IO模块：支持parquet优先，pickle兜底
"""

import pickle
import gzip
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
from loguru import logger
import json
from datetime import datetime

def _try_import_parquet():
    """尝试导入parquet库"""
    try:
        import pyarrow.parquet as pq
        return True, 'pyarrow'
    except ImportError:
        try:
            import fastparquet
            return True, 'fastparquet'
        except ImportError:
            return False, None

def save_table(df: pd.DataFrame, path_base: Path, metadata: Optional[Dict[str, Any]] = None):
    """
    保存DataFrame到缓存（parquet优先，pickle兜底）
    
    Args:
        df: 要保存的DataFrame
        path_base: 基础路径（不带后缀）
        metadata: 可选的元数据字典
    """
    path_base = Path(path_base)
    path_base.parent.mkdir(parents=True, exist_ok=True)
    
    # 尝试使用parquet
    parquet_available, parquet_lib = _try_import_parquet()
    
    if parquet_available:
        try:
            parquet_path = path_base.with_suffix('.parquet')
            if parquet_lib == 'pyarrow':
                df.to_parquet(parquet_path, engine='pyarrow', compression='snappy')
            else:  # fastparquet
                df.to_parquet(parquet_path, engine='fastparquet', compression='snappy')
            logger.debug(f"保存到parquet: {parquet_path}")
            
            # 保存元数据（如果有）
            if metadata:
                meta_path = path_base.with_suffix('.parquet.meta.json')
                with open(meta_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, default=str, ensure_ascii=False)
            return
        except Exception as e:
            logger.warning(f"保存parquet失败: {e}，回退到pickle")
    
    # 回退到pickle
    pkl_path = path_base.with_suffix('.pkl.gz')
    try:
        with gzip.open(pkl_path, 'wb') as f:
            pickle.dump({
                'data': df,
                'metadata': metadata or {}
            }, f)
        logger.debug(f"保存到pickle: {pkl_path}")
    except Exception as e:
        logger.error(f"保存pickle失败: {e}")
        raise

def load_table(path_base: Path) -> tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """
    从缓存加载DataFrame（优先parquet，失败则尝试pickle）
    
    Args:
        path_base: 基础路径（不带后缀）
    
    Returns:
        (DataFrame, metadata_dict)
    """
    path_base = Path(path_base)
    
    # 优先尝试parquet
    parquet_path = path_base.with_suffix('.parquet')
    if parquet_path.exists():
        try:
            df = pd.read_parquet(parquet_path)
            # 尝试加载元数据
            meta_path = path_base.with_suffix('.parquet.meta.json')
            metadata = None
            if meta_path.exists():
                try:
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                except Exception as e:
                    logger.debug(f"读取parquet元数据失败: {e}")
            logger.debug(f"从parquet加载: {parquet_path}")
            return df, metadata
        except Exception as e:
            logger.warning(f"读取parquet失败: {e}，尝试pickle")
    
    # 尝试pickle
    pkl_path = path_base.with_suffix('.pkl.gz')
    if pkl_path.exists():
        try:
            with gzip.open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, dict) and 'data' in data:
                df = data['data']
                metadata = data.get('metadata')
            else:
                # 旧格式：直接是DataFrame
                df = data
                metadata = None
            logger.debug(f"从pickle加载: {pkl_path}")
            return df, metadata
        except Exception as e:
            logger.error(f"读取pickle失败: {e}")
            raise
    
    raise FileNotFoundError(f"缓存文件不存在: {parquet_path} 或 {pkl_path}")

def table_exists(path_base: Path) -> bool:
    """检查缓存文件是否存在"""
    path_base = Path(path_base)
    return (
        path_base.with_suffix('.parquet').exists() or
        path_base.with_suffix('.pkl.gz').exists()
    )
