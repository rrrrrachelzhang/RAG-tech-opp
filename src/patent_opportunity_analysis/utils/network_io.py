"""
网络保存/加载工具模块

提供统一的DKN网络保存和加载接口，支持gzip压缩和metadata管理。
"""
import pickle
import gzip
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from loguru import logger
from ..utils.dkn_wrapper import DKNNetwork


def compute_file_hash(file_path: Path) -> str:
    """计算文件的SHA256哈希值
    
    Args:
        file_path: 文件路径
        
    Returns:
        16位短哈希值
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()[:16]


def compute_data_hash(data: Any) -> str:
    """计算数据的哈希值（用于配置等）
    
    Args:
        data: 可序列化的数据
        
    Returns:
        16位短哈希值
    """
    data_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(data_str.encode()).hexdigest()[:16]


def save_dkn(dkn: DKNNetwork, output_path: Path) -> None:
    """保存DKN网络到文件（gzip压缩）
    
    Args:
        dkn: DKNNetwork对象
        output_path: 输出文件路径（会自动添加.gz扩展名）
    """
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 添加.gz扩展名
    if not output_path.name.endswith('.gz'):
        output_path = output_path.parent / f"{output_path.name}.gz"
    
    logger.info(f"保存{dkn.kind}网络到: {output_path}")
    
    # 保存网络对象
    with gzip.open(output_path, 'wb') as f:
        pickle.dump(dkn, f, protocol=5)
    
    logger.success(f"✅ {dkn.kind}网络已保存: {output_path}")


def load_dkn(input_path: Path) -> DKNNetwork:
    """从文件加载DKN网络
    
    Args:
        input_path: 输入文件路径（支持.gz扩展名）
        
    Returns:
        DKNNetwork对象
        
    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 加载的对象不是DKNNetwork
    """
    # 自动添加.gz扩展名（如果不存在）
    if not input_path.name.endswith('.gz'):
        gz_path = input_path.parent / f"{input_path.name}.gz"
        if gz_path.exists():
            input_path = gz_path
        else:
            # 尝试直接加载（未压缩）
            if input_path.exists():
                logger.warning(f"加载未压缩的网络文件: {input_path}")
                with open(input_path, 'rb') as f:
                    dkn = pickle.load(f)
            else:
                raise FileNotFoundError(f"网络文件不存在: {input_path}")
    else:
        if not input_path.exists():
            raise FileNotFoundError(f"网络文件不存在: {input_path}")
    
    logger.info(f"加载网络文件: {input_path}")
    
    # 加载网络对象
    if input_path.name.endswith('.gz'):
        with gzip.open(input_path, 'rb') as f:
            dkn = pickle.load(f)
    else:
        with open(input_path, 'rb') as f:
            dkn = pickle.load(f)
    
    # 验证类型
    if not isinstance(dkn, DKNNetwork):
        raise ValueError(f"加载的对象不是DKNNetwork: {type(dkn)}")
    
    # 验证不变量
    dkn.assert_invariants()
    
    logger.success(f"✅ 成功加载{dkn.kind}网络: {dkn}")
    return dkn


def create_networks_metadata(
    hdkn: DKNNetwork,
    pdkn: DKNNetwork,
    input_data_hash: str,
    config_hash: str,
    hist_end_year: int,
    max_year: int,
    run_id: str,
    patents_count: int,
    input_data_rows: int = None
) -> Dict[str, Any]:
    """创建网络构建步骤的metadata
    
    Args:
        hdkn: HDKN网络对象
        pdkn: PDKN网络对象
        input_data_hash: 输入数据哈希
        config_hash: 配置哈希
        hist_end_year: 历史截止年份
        max_year: 最大年份
        run_id: 运行ID
        patents_count: 专利数量（实际加载的专利数）
        input_data_rows: 输入数据行数（CSV文件行数，用于测试模式校验）
        
    Returns:
        metadata字典
    """
    metadata = {
        "step_name": "01_networks",
        "created_at": datetime.now().isoformat(),
        "run_id": run_id,
        "input_data_hash": input_data_hash,
        "config_hash": config_hash,
        "hist_end_year": hist_end_year,
        "max_year": max_year,
        "graph_stats": {
            "hdkn": {
                "nodes": hdkn.number_of_nodes(),
                "edges": hdkn.number_of_edges(),
                "ref_year": hdkn.ref_year,
                "kind": hdkn.kind
            },
            "pdkn": {
                "nodes": pdkn.number_of_nodes(),
                "edges": pdkn.number_of_edges(),
                "ref_year": pdkn.ref_year,
                "kind": pdkn.kind
            }
        },
        "patents_count": patents_count,
        "invariants": {
            "hdkn_ref_year_equals_hist_end_year": hdkn.ref_year == hist_end_year,
            "pdkn_ref_year_equals_max_year": pdkn.ref_year == max_year
        }
    }
    
    # 添加输入数据行数（用于测试模式校验）
    if input_data_rows is not None:
        metadata["input_data_rows"] = input_data_rows
    
    return metadata


def save_metadata(metadata: Dict[str, Any], output_path: Path) -> None:
    """保存metadata到JSON文件
    
    Args:
        metadata: metadata字典
        output_path: 输出文件路径
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.debug(f"Metadata已保存: {output_path}")


def load_metadata(input_path: Path) -> Dict[str, Any]:
    """加载metadata从JSON文件
    
    Args:
        input_path: 输入文件路径
        
    Returns:
        metadata字典
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def validate_metadata_consistency(
    current_metadata: Dict[str, Any],
    expected_metadata: Dict[str, Any],
    check_fields: list = None
) -> tuple[bool, str]:
    """验证metadata一致性
    
    Args:
        current_metadata: 当前metadata
        expected_metadata: 期望的metadata
        check_fields: 要检查的字段列表（None表示检查所有关键字段）
        
    Returns:
        (是否一致, 错误消息)
    """
    if check_fields is None:
        check_fields = ['hist_end_year', 'max_year', 'input_data_hash', 'config_hash']
    
    for field in check_fields:
        if field in expected_metadata and field in current_metadata:
            expected_val = expected_metadata[field]
            # 期望值为 None 表示“不关心该字段”，与已有产物一致即可（用于 max_year 等由数据推导的字段）
            if expected_val is None:
                continue
            if current_metadata[field] != expected_val:
                return False, f"字段 {field} 不匹配: 当前={current_metadata[field]}, 期望={expected_metadata[field]}"
    
    return True, ""
