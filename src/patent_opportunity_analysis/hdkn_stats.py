# src/patent_opportunity_analysis/hdkn_stats.py
"""
HDKN图级统计一次性预计算模块

实现严格复现版：所有特征严格从HDKN计算，避免未来信息泄漏
"""

import pandas as pd
import networkx as nx
import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from loguru import logger
from tqdm import tqdm

from . import config as _config
from .utils.paths import MODELS_CACHE_DIR
from .utils.cache_io import save_table, load_table, table_exists
from .utils.cache_manifest import (
    load_manifest, save_manifest, check_cache_validity,
    update_manifest_entry, create_manifest_entry, compute_data_hash
)
from .feature_extraction import (
    compute_eigen_centrality_power_iteration,
    get_hdkn_constraint_map,
    _get_graph_fingerprint
)

DECAY_FACTOR = getattr(_config, 'DECAY_FACTOR', 0.9)
HIST_END_YEAR = getattr(_config, 'HIST_END_YEAR', 2022)
NODE_STRENGTH_MODE = getattr(_config, 'NODE_STRENGTH_MODE', 'weighted_degree')


def build_or_load_hdkn_stats(
    HDKN,
    config: Optional[Dict[str, Any]] = None,
    force_rebuild: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    HDKN图级统计一次性预计算（并缓存）
    
    Args:
        HDKN: HDKN对象（DKNNetwork或nx.Graph）
        config: 配置字典（hdkn_end_year, decay_factor等）
        force_rebuild: 是否强制重建（忽略缓存）
    
    Returns:
        (node_stats_df, edge_stats_df, auxiliary_dicts)
        - node_stats_df: 节点统计DataFrame（index=node）
        - edge_stats_df: 边统计DataFrame（index=(u,v)）
        - auxiliary_dicts: 辅助字典（如p90阈值等）
    """
    # 获取HDKN图对象
    if hasattr(HDKN, 'assert_kind'):
        HDKN.assert_kind("HDKN")
        hdkn_graph = HDKN.graph
        hist_end_year = HDKN.hist_end_year
    else:
        hdkn_graph = HDKN
        hist_end_year = config.get('hdkn_end_year', HIST_END_YEAR) if config else HIST_END_YEAR
    
    # 合并配置
    if config is None:
        config = {}
    skip_betweenness = config.get('skip_betweenness', False)
    skip_eigen = config.get('skip_eigen', False)
    skip_constraint = config.get('skip_constraint', False)
    decay_ref_year = config.get('decay_ref_year', hist_end_year)  # 衰减参考年，默认 hist_end_year（兼容旧逻辑）
    final_config = {
        'hdkn_end_year': hist_end_year,
        'decay_ref_year': decay_ref_year,
        'decay_factor': config.get('decay_factor', DECAY_FACTOR),
        'node_strength_mode': config.get('node_strength_mode', NODE_STRENGTH_MODE),
        'skip_betweenness': skip_betweenness,
        'skip_eigen': skip_eigen,
        'skip_constraint': skip_constraint,
    }
    
    # 生成缓存键（decay_ref_year 区分观测年，skip_* 影响 node 缓存内容）
    # 使用传入的 hist_end_year，避免与模块级 HIST_END_YEAR 不一致导致缓存键错误
    graph_hash = _compute_graph_hash(hdkn_graph, hist_end_year)
    cache_key_base = f"hdkn_stats_{graph_hash}_{hist_end_year}_ref{decay_ref_year}_{final_config['decay_factor']}_sb{int(skip_betweenness)}_se{int(skip_eigen)}_sc{int(skip_constraint)}"
    
    node_cache_path = MODELS_CACHE_DIR / f"{cache_key_base}_nodes"
    edge_cache_path = MODELS_CACHE_DIR / f"{cache_key_base}_edges"
    
    # 检查缓存有效性
    if not force_rebuild:
        is_valid, reason = check_cache_validity(
            MODELS_CACHE_DIR,
            final_config,
            auto_rebuild=True
        )
        
        if is_valid and table_exists(node_cache_path) and table_exists(edge_cache_path):
            try:
                logger.info("📊 从缓存加载HDKN统计...")
                node_stats_df, node_meta = load_table(node_cache_path)
                edge_stats_df, edge_meta = load_table(edge_cache_path)

                # 兼容旧缓存：若缺少列则补算（仅 betweenness）或填默认值
                if 'betweenness' not in node_stats_df.columns:
                    if not skip_betweenness:
                        logger.info("   旧缓存缺少 betweenness，补算...")
                        n_nodes = hdkn_graph.number_of_nodes()
                        k_default = getattr(_config, 'BETWEENNESS_K_SAMPLE', 50)
                        k_sample = min(k_default, n_nodes) if n_nodes > 2000 else None
                        bc_dict = nx.betweenness_centrality(
                            hdkn_graph, k=k_sample, weight="weight",
                            normalized=True, seed=42
                        )
                        node_stats_df['betweenness'] = node_stats_df.index.map(
                            lambda n: bc_dict.get(n, 0.0) if not np.isnan(bc_dict.get(n, np.nan)) else 0.0
                        )
                    else:
                        node_stats_df['betweenness'] = 0.0
                if 'eigen' not in node_stats_df.columns:
                    node_stats_df['eigen'] = 0.0
                if 'constraint' not in node_stats_df.columns:
                    node_stats_df['constraint'] = 1.0

                # 构建辅助字典
                auxiliary_dicts = {
                    'p90_year_n': node_stats_df['year_min_node'].quantile(0.9),
                    'p90_year_e': edge_stats_df['year_min_edge'].quantile(0.9),
                }
                
                logger.success(f"✅ 缓存加载成功: {len(node_stats_df)}个节点, {len(edge_stats_df)}条边")
                return node_stats_df, edge_stats_df, auxiliary_dicts
            except Exception as e:
                logger.warning(f"加载缓存失败: {e}，将重新计算")
    
    # 计算统计
    logger.info("📊 计算HDKN图级统计（首次计算，可能需要几分钟）...")
    logger.info(f"   图规模: {hdkn_graph.number_of_nodes()}节点, {hdkn_graph.number_of_edges()}边")
    
    # 1. 节点统计（排序保证可复现）
    logger.info("   计算节点统计...")
    node_data = []
    for node in tqdm(sorted(hdkn_graph.nodes()), desc="   节点", leave=False):
        data = hdkn_graph.nodes[node]
        node_data.append({
            'node': node,
            'year_min_node': data.get('year_min', hist_end_year),
            'strength': data.get('strength', 0.0),
            'degree': hdkn_graph.degree(node),
            'patents_count_node': len(data.get('patents', set())),
        })
    
    node_stats_df = pd.DataFrame(node_data).set_index('node')
    
    # 2. 计算 Eigen centrality（全图，仅当需要时）
    if skip_eigen:
        node_stats_df['eigen'] = 0.0
        logger.info("   跳过 Eigen centrality（未选用该特征）")
    else:
        logger.info("   计算Eigen centrality...")
        try:
            if hdkn_graph.number_of_nodes() > 1000:
                eigen_dict = compute_eigen_centrality_power_iteration(hdkn_graph, weight="weight")
            else:
                if nx.is_connected(hdkn_graph):
                    eigen_dict = nx.eigenvector_centrality_numpy(hdkn_graph, weight="weight", max_iter=1000)
                else:
                    largest_cc = max(nx.connected_components(hdkn_graph), key=len)
                    if len(largest_cc) > 1:
                        hdkn_sub = hdkn_graph.subgraph(largest_cc).copy()
                        eigen_dict = nx.eigenvector_centrality_numpy(hdkn_sub, weight="weight", max_iter=1000)
                        eigen_dict = {n: eigen_dict.get(n, 0.0) for n in hdkn_graph.nodes()}
                    else:
                        eigen_dict = {n: 0.0 for n in hdkn_graph.nodes()}
            
            node_stats_df['eigen'] = node_stats_df.index.map(lambda n: eigen_dict.get(n, 0.0))
            logger.info("   ✅ Eigen centrality计算完成")
        except Exception as e:
            logger.error(f"计算Eigen centrality失败: {e}")
            node_stats_df['eigen'] = 0.0
    
    # 3. 计算 Constraint（全图，仅当需要时）
    if skip_constraint:
        node_stats_df['constraint'] = 1.0
        logger.info("   跳过 Constraint（未选用该特征）")
    else:
        logger.info("   计算Constraint...")
        try:
            constraint_dict = get_hdkn_constraint_map(
                HDKN,
                nodes=None,
                weight="weight",
                use_cache=True,
                hist_end_year=hist_end_year
            )
            # 处理NaN值
            node_stats_df['constraint'] = node_stats_df.index.map(
                lambda n: constraint_dict.get(n, 1.0) if not np.isnan(constraint_dict.get(n, np.nan)) else 1.0
            )
            logger.info("   ✅ Constraint计算完成")
        except Exception as e:
            logger.error(f"计算Constraint失败: {e}")
            node_stats_df['constraint'] = 1.0

    # 3.5 计算 Betweenness Centrality（中介中心度，仅当需要时）
    # 算法：Brandes 最短路径中介中心性，weight 用于加权最短路径（NetworkX 将 weight 视为距离，
    # 路径成本=边权之和；本图 weight=时间衰减强度，高权=强连接，作为距离时强连接路径成本更高）
    if skip_betweenness:
        node_stats_df['betweenness'] = 0.0
        logger.info("   跳过 Betweenness Centrality（未选用该特征）")
    else:
        logger.info("   计算 Betweenness Centrality...")
        try:
            n_nodes = hdkn_graph.number_of_nodes()
            k_default = getattr(_config, 'BETWEENNESS_K_SAMPLE', 50)
            k_sample = min(k_default, n_nodes) if n_nodes > 2000 else None
            if k_sample is not None:
                logger.info(f"   大图（{n_nodes} 节点），使用 k={k_sample} 采样近似")
            # 使用单线程实现以保证可复现性（parallel backend 可能导致多次运行 LL 不一致）
            bc_kw = dict(k=k_sample, weight="weight", normalized=True, seed=42)
            bc_dict = nx.betweenness_centrality(hdkn_graph, **bc_kw)
            node_stats_df['betweenness'] = node_stats_df.index.map(
                lambda n: bc_dict.get(n, 0.0) if not np.isnan(bc_dict.get(n, np.nan)) else 0.0
            )
            logger.info("   ✅ Betweenness Centrality 计算完成")
        except Exception as e:
            logger.error(f"计算 Betweenness Centrality 失败: {e}")
            node_stats_df['betweenness'] = 0.0

    # 4. 边统计
    logger.info("   计算边统计...")
    edge_data = []
    for u, v, data in tqdm(hdkn_graph.edges(data=True), desc="   边", leave=False):
        edge_key = tuple(sorted([u, v]))  # 规范化无向图边
        edge_data.append({
            'u': edge_key[0],
            'v': edge_key[1],
            'year_min_edge': data.get('year_min', hist_end_year),
            'weight': data.get('weight', 0.0),
            'patents_count_edge': len(data.get('patents', set())),
        })
    
    edge_stats_df = pd.DataFrame(edge_data)
    edge_stats_df['edge_key'] = [
        tuple(sorted([row['u'], row['v']]))
        for row in tqdm(edge_data, desc="   构建边索引", leave=False)
    ]
    edge_stats_df = edge_stats_df.set_index('edge_key')
    
    # 5. 构建辅助字典
    auxiliary_dicts = {
        'p90_year_n': float(node_stats_df['year_min_node'].quantile(0.9)),
        'p90_year_e': float(edge_stats_df['year_min_edge'].quantile(0.9)),
    }
    
    # 6. 保存缓存
    logger.info("   保存缓存...")
    try:
        node_metadata = {
            'schema': {
                'index': 'node',
                'columns': list(node_stats_df.columns),
                'dtypes': {col: str(dtype) for col, dtype in node_stats_df.dtypes.items()}
            },
            'config': final_config,
            'graph_stats': {
                'n_nodes': hdkn_graph.number_of_nodes(),
                'n_edges': hdkn_graph.number_of_edges(),
            }
        }
        
        edge_metadata = {
            'schema': {
                'index': 'edge_key',
                'columns': list(edge_stats_df.columns),
                'dtypes': {col: str(dtype) for col, dtype in edge_stats_df.dtypes.items()}
            },
            'config': final_config,
            'graph_stats': {
                'n_nodes': hdkn_graph.number_of_nodes(),
                'n_edges': hdkn_graph.number_of_edges(),
            }
        }
        
        save_table(node_stats_df, node_cache_path, node_metadata)
        save_table(edge_stats_df, edge_cache_path, edge_metadata)
        
        # 更新manifest
        update_manifest_entry(
            MODELS_CACHE_DIR,
            f"{cache_key_base}_nodes",
            create_manifest_entry(
                f"{cache_key_base}_nodes",
                node_cache_path,
                node_metadata['schema'],
                final_config,
                node_metadata['graph_stats']
            )
        )
        update_manifest_entry(
            MODELS_CACHE_DIR,
            f"{cache_key_base}_edges",
            create_manifest_entry(
                f"{cache_key_base}_edges",
                edge_cache_path,
                edge_metadata['schema'],
                final_config,
                edge_metadata['graph_stats']
            )
        )
        
        # 保存全局manifest配置
        manifest = load_manifest(MODELS_CACHE_DIR) or {
            'version': '1.0',
            'created_at': pd.Timestamp.now().isoformat(),
            'cache_files': {}
        }
        manifest['config'] = final_config
        save_manifest(MODELS_CACHE_DIR, manifest)
        
        logger.success("✅ 缓存保存完成")
    except Exception as e:
        logger.warning(f"保存缓存失败: {e}")
    
    logger.success(f"✅ HDKN统计计算完成: {len(node_stats_df)}个节点, {len(edge_stats_df)}条边")
    return node_stats_df, edge_stats_df, auxiliary_dicts


def _compute_graph_hash(graph: nx.Graph, hist_end_year: Optional[int] = None) -> str:
    """计算图的哈希值（用于缓存键）
    
    Args:
        graph: HDKN 图对象
        hist_end_year: 历史截止年，用于指纹一致性。None 时使用模块级 HIST_END_YEAR。
    """
    _hist = hist_end_year if hist_end_year is not None else HIST_END_YEAR
    try:
        return _get_graph_fingerprint(graph, _hist, "weight")[:16]
    except Exception:
        # 回退实现
        import hashlib
        nodes_sorted = tuple(sorted(graph.nodes()))
        edges_sorted = tuple(sorted(graph.edges()))
        hash_data = (
            graph.number_of_nodes(),
            graph.number_of_edges(),
            nodes_sorted[:100] if len(nodes_sorted) > 100 else nodes_sorted,
            edges_sorted[:100] if len(edges_sorted) > 100 else edges_sorted,
        )
        hash_str = str(hash_data)
        return hashlib.md5(hash_str.encode('utf-8')).hexdigest()[:16]
