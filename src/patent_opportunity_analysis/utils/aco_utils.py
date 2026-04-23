# src/utils/aco_utils.py
"""
ACO算法工具函数
包括参数加载、可视化等
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from loguru import logger

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_aco_config(config_path: Optional[Path] = None) -> Dict:
    """
    加载ACO配置文件
    
    Args:
        config_path: 配置文件路径，如果为None则使用默认路径
    
    Returns:
        配置字典
    """
    if config_path is None:
        from .paths import ACO_CONFIG_FILE
        config_path = ACO_CONFIG_FILE
    
    if not config_path.exists():
        logger.warning(f"ACO配置文件不存在: {config_path}，使用默认配置")
        return get_default_aco_config()
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"成功加载ACO配置: {config_path}")
        return config
    except Exception as e:
        logger.error(f"加载ACO配置失败: {e}，使用默认配置")
        return get_default_aco_config()

def get_diverse_top_k(
    all_solutions: List[Dict],
    top_k: int = 10,
    max_overlap_ratio: float = 0.5,
) -> List[Dict]:
    """
    从所有解中筛选出多样性强的 Top-K 技术族群（基于 Jaccard 相似度/重叠率过滤）

    Args:
        all_solutions: 列表，元素格式为 {'nodes': set(...) 或 list(...), 'score': float}
        top_k: 返回的机会数量
        max_overlap_ratio: 允许的最大重叠率（例如 0.5 表示最多允许一半节点相同）

    Returns:
        多样性筛选后的解列表，格式与输入一致（nodes 保持原类型）
    """
    if not all_solutions:
        return []

    # 先按得分从高到低排序
    sorted_solutions = sorted(all_solutions, key=lambda x: x["score"], reverse=True)

    diverse_top_k = []

    for sol in sorted_solutions:
        if len(diverse_top_k) >= top_k:
            break

        nodes = sol.get("nodes")
        if nodes is None:
            continue
        nodes_set = set(nodes) if not isinstance(nodes, set) else nodes
        if not nodes_set:
            continue

        # 检查当前解与已选入 diverse_top_k 的解是否高度重叠
        is_too_similar = False
        for selected_sol in diverse_top_k:
            selected_nodes = selected_sol.get("nodes")
            selected_set = set(selected_nodes) if not isinstance(selected_nodes, set) else selected_nodes
            overlap_count = len(nodes_set.intersection(selected_set))
            if overlap_count / len(nodes_set) > max_overlap_ratio:
                is_too_similar = True
                break

        if not is_too_similar:
            diverse_top_k.append(sol)

    return diverse_top_k


def get_default_aco_config() -> Dict:
    """获取默认ACO配置"""
    return {
        'num_ants': 300,
        'num_generations': 200,
        'top_k_per_gen': 50,
        'pheromone': {
            'init': 1.0,
            'alpha': 3,
            'evaporation': 0.95,
            'tau_min': 0.01,
            'tau_max': 4.0
        },
        'heuristic': {
            'beta': 4
        },
        'subnetwork_size': 15,
        'top_k_opportunities': 10,
        'max_overlap_ratio': 0.5
    }

def plot_aco_convergence(
    best_scores_per_gen: List[float],
    avg_scores_per_gen: List[float],
    output_path: Path
):
    """
    绘制ACO收敛曲线
    
    Args:
        best_scores_per_gen: 每代最优得分
        avg_scores_per_gen: 每代平均得分
        output_path: 输出路径
    """
    plt.figure(figsize=(12, 6))
    generations = range(1, len(best_scores_per_gen) + 1)
    
    plt.plot(generations, best_scores_per_gen, 'b-', linewidth=2.5, label='最优得分', marker='o', markersize=4)
    if avg_scores_per_gen:
        plt.plot(generations, avg_scores_per_gen, 'r--', linewidth=2, label='平均得分', alpha=0.8, marker='s', markersize=3)
    
    plt.xlabel('迭代代数', fontsize=13, fontweight='bold')
    plt.ylabel('得分', fontsize=13, fontweight='bold')
    plt.title('ACO算法收敛曲线', fontsize=15, fontweight='bold', pad=15)
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # 添加统计信息
    if best_scores_per_gen:
        max_score = max(best_scores_per_gen)
        final_score = best_scores_per_gen[-1]
        improvement = ((final_score - best_scores_per_gen[0]) / best_scores_per_gen[0] * 100) if best_scores_per_gen[0] > 0 else 0
        plt.text(0.02, 0.98, f'最高得分: {max_score:.4f}\n最终得分: {final_score:.4f}\n提升: {improvement:.1f}%',
                transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # 保存到reports目录
    if output_path.parent.name != 'reports':
        reports_dir = output_path.parent.parent / "reports" if output_path.parent.name == "models" else output_path.parent / "reports"
        reports_dir.mkdir(exist_ok=True)
        output_path = reports_dir / "aco_progress.png"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.success(f"ACO收敛曲线已保存到: {output_path}")

def visualize_subnetwork(
    PDKN: nx.Graph,
    nodes: List[str],
    output_path: Path,
    title: str = "技术机会子网络"
):
    """
    可视化子网络
    
    Args:
        PDKN: 完整DKN图（可以是 DKNNetwork 或 nx.Graph）
        nodes: 子网络节点列表
        output_path: 输出路径
        title: 图标题
    """
    # 提取子图（处理 DKNNetwork 和普通 Graph）
    try:
        if hasattr(PDKN, 'assert_kind'):
            subg_result = PDKN.subgraph(nodes)
            if hasattr(subg_result, 'assert_kind'):
                subg = subg_result.graph.copy()
            elif hasattr(subg_result, 'copy'):
                subg = subg_result.copy()
            else:
                subg = subg_result
        else:
            subg = PDKN.subgraph(nodes).copy()
        
        if not hasattr(subg, 'number_of_nodes'):
            logger.warning(f"子图提取失败，subg 类型: {type(subg)}, PDKN 类型: {type(PDKN)}")
            return
    except Exception as e:
        logger.warning(f"提取子图时出错: {e}, PDKN 类型: {type(PDKN)}")
        return
    
    if subg.number_of_nodes() == 0:
        logger.warning("子网络为空，无法可视化")
        return
    
    # 若子图无边（诱导子图可能不连通），按 ACO 路径顺序添加边以展示探索顺序
    if subg.number_of_edges() == 0 and len(nodes) >= 2:
        for i in range(len(nodes) - 1):
            if not subg.has_edge(nodes[i], nodes[i + 1]):
                subg.add_edge(nodes[i], nodes[i + 1], weight=1.0)
        logger.info(f"子图原无边，已按路径顺序添加 {len(nodes)-1} 条边")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_facecolor('#f8fafc')
    fig.patch.set_facecolor('#f8fafc')
    
    # 使用 spring 布局，k 稍大以分散节点
    pos = nx.spring_layout(subg, k=1.2, iterations=80, seed=42)
    
    # 用 matplotlib 直接画边
    edges = list(subg.edges())
    if edges:
        # 边过多时用 MST 减少重叠，否则画全部
        if len(edges) > 50:
            try:
                mst = nx.minimum_spanning_tree(subg, weight='weight')
                edges = list(mst.edges())
            except Exception:
                pass
        edge_weights = [subg[u][v].get('weight', 0.1) for u, v in edges]
        w_min, w_max = min(edge_weights), max(edge_weights)
        w_range = (w_max - w_min) or 1
        for (u, v), w in zip(edges, edge_weights):
            lw = max(2.0, 2.0 + 3.0 * (w - w_min) / w_range)
            ax.plot(
                [pos[u][0], pos[v][0]],
                [pos[u][1], pos[v][1]],
                color='#2563eb',
                linewidth=lw,
                alpha=0.9,
                zorder=0
            )
    
    # 节点大小：strength 可能很大(数千)，需归一化到合理范围，避免遮挡边
    strengths = [subg.nodes[n].get('strength', 1) for n in subg.nodes()]
    s_min, s_max = min(strengths), max(strengths)
    s_range = (s_max - s_min) or 1
    node_sizes = [
        int(400 + 600 * (s - s_min) / s_range)
        for s in strengths
    ]
    nx.draw_networkx_nodes(
        subg, pos,
        node_size=node_sizes,
        node_color='#ffffff',
        edgecolors='#1e293b',
        linewidths=1.5,
        alpha=0.95,
        ax=ax
    )
    
    # 绘制标签：加大字号、深色
    nx.draw_networkx_labels(
        subg, pos,
        font_size=10,
        font_weight='bold',
        font_color='#0f172a',
        ax=ax
    )
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20, color='#0f172a')
    ax.axis('off')
    plt.tight_layout()
    
    # 保存到reports目录
    if output_path.parent.name != 'reports':
        reports_dir = output_path.parent.parent / "reports" if output_path.parent.name == "models" else output_path.parent / "reports"
        reports_dir.mkdir(exist_ok=True)
        # 从原路径提取文件名
        filename = output_path.name
        output_path = reports_dir / filename
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.success(f"子网络可视化已保存到: {output_path}")
