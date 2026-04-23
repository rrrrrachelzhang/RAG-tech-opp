# src/patent_opportunity_analysis/utils/cache_manifest.py
"""
缓存manifest管理模块
记录缓存文件的元数据，用于一致性校验
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import pandas as pd
from loguru import logger
import hashlib

MANIFEST_FILE = "manifest.json"

def compute_data_hash(data: Any) -> str:
    """计算数据的哈希值"""
    data_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.md5(data_str.encode('utf-8')).hexdigest()[:16]

def load_manifest(cache_dir: Path) -> Optional[Dict[str, Any]]:
    """加载manifest文件"""
    manifest_path = cache_dir / MANIFEST_FILE
    if not manifest_path.exists():
        return None
    
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"读取manifest失败: {e}")
        return None

def save_manifest(cache_dir: Path, manifest: Dict[str, Any]):
    """保存manifest文件"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / MANIFEST_FILE
    
    try:
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, default=str, ensure_ascii=False)
        logger.debug(f"保存manifest: {manifest_path}")
    except Exception as e:
        logger.error(f"保存manifest失败: {e}")
        raise

def update_manifest_entry(
    cache_dir: Path,
    cache_key: str,
    file_info: Dict[str, Any]
):
    """更新manifest中的某个缓存文件条目"""
    manifest = load_manifest(cache_dir) or {
        'version': '1.0',
        'created_at': datetime.now().isoformat(),
        'cache_files': {}
    }
    
    if 'cache_files' not in manifest:
        manifest['cache_files'] = {}
    
    manifest['cache_files'][cache_key] = {
        **file_info,
        'updated_at': datetime.now().isoformat()
    }
    
    manifest['updated_at'] = datetime.now().isoformat()
    save_manifest(cache_dir, manifest)

def check_cache_validity(
    cache_dir: Path,
    expected_config: Dict[str, Any],
    auto_rebuild: bool = True
) -> tuple[bool, Optional[str]]:
    """
    检查缓存是否有效（与当前配置一致）
    
    Args:
        cache_dir: 缓存目录
        expected_config: 期望的配置（hdkn_end_year, decay_factor等）
        auto_rebuild: 是否自动重建（如果无效）
    
    Returns:
        (is_valid, reason)
    """
    manifest = load_manifest(cache_dir)
    if manifest is None:
        return False, "manifest不存在"
    
    # 检查关键配置
    cached_config = manifest.get('config', {})
    
    if cached_config.get('hdkn_end_year') != expected_config.get('hdkn_end_year'):
        reason = f"hdkn_end_year不匹配: 缓存={cached_config.get('hdkn_end_year')}, 期望={expected_config.get('hdkn_end_year')}"
        if auto_rebuild:
            logger.warning(f"缓存过期: {reason}，将自动重建")
        return False, reason
    
    if cached_config.get('decay_factor') != expected_config.get('decay_factor'):
        reason = f"decay_factor不匹配: 缓存={cached_config.get('decay_factor')}, 期望={expected_config.get('decay_factor')}"
        if auto_rebuild:
            logger.warning(f"缓存过期: {reason}，将自动重建")
        return False, reason
    
    return True, None

def create_manifest_entry(
    cache_key: str,
    file_path: Path,
    schema: Dict[str, Any],
    config: Dict[str, Any],
    graph_stats: Dict[str, Any]
) -> Dict[str, Any]:
    """创建manifest条目"""
    return {
        'cache_key': cache_key,
        'file_path': str(file_path),
        'created_at': datetime.now().isoformat(),
        'schema': schema,
        'config': config,
        'graph_stats': graph_stats
    }
