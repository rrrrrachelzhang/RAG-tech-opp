#!/usr/bin/env python3
"""
绘制技术机会子网络图（论文 Fig.6 风格）

运行:
    python scripts/plot_opportunity_network.py [--ranks 1 2 3] [--input PATH]
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import networkx as nx
import numpy as np

from src.patent_opportunity_analysis.utils.paths import RAG_ENRICHED_JSON

COLOR_NEW       = "#D32F2F"
COLOR_MARGINAL  = "#EF6C00"
COLOR_MARGINAL_FILL = "#FFB74D"
COLOR_CONV_NODE = "#FFFFFF"
COLOR_NODE_EDGE = "#424242"
COLOR_CONV_EDGE = "#2E7D32"
COLOR_NEW_EDGE  = "#1565C0"
COLOR_SPE_EDGE  = "#1565C0"


def _node_radius(strength: float, min_s: float, max_s: float) -> float:
    if max_s <= min_s:
        return 900
    ratio = (math.log1p(strength) - math.log1p(min_s)) / (math.log1p(max_s) - math.log1p(min_s))
    return 500 + ratio * 1800


def _edge_width(con_e: float, max_con_e: float) -> float:
    if max_con_e <= 0:
        return 1.0
    ratio = math.log1p(con_e) / math.log1p(max_con_e)
    return 0.8 + ratio * 4.0


def _layout(G: nx.Graph, node_data: Dict, seed: int) -> Dict:
    """
    混合布局：kamada_kawai 布局后，将低度 new/marginal 节点拉近主簇。
    """
    try:
        pos = nx.kamada_kawai_layout(G)
    except Exception:
        pos = nx.spring_layout(G, seed=seed, k=2.0/math.sqrt(max(G.number_of_nodes(), 1)), iterations=100)

    conv_nodes = [n for n in G.nodes() if node_data.get(n, {}).get("type") == "conventional"]
    special_nodes = [n for n in G.nodes() if node_data.get(n, {}).get("type") in ("new", "marginal")]

    if not conv_nodes or not special_nodes:
        return pos

    conv_positions = np.array([pos[n] for n in conv_nodes])
    center = conv_positions.mean(axis=0)
    dists = np.linalg.norm(conv_positions - center, axis=1)
    conv_radius = np.percentile(dists, 80)

    groups: List[List[str]] = []
    assigned: Set[str] = set()
    for sn in special_nodes:
        if sn in assigned:
            continue
        group = [sn]
        assigned.add(sn)
        for sn2 in special_nodes:
            if sn2 not in assigned and G.has_edge(sn, sn2):
                group.append(sn2)
                assigned.add(sn2)
        groups.append(group)

    placed: Set[str] = set()
    all_special = set()
    for g in groups:
        all_special.update(g)

    for _pass in range(3):
        for gi, group in enumerate(groups):
            unplaced = [sn for sn in group if sn not in placed]
            if not unplaced:
                continue

            for sn in unplaced:
                anchors = []
                for nb in G.neighbors(sn):
                    if nb in conv_nodes or nb in placed:
                        anchors.append(pos[nb])

                if anchors:
                    anchor = np.mean(anchors, axis=0)
                    vec = anchor - center
                    norm = np.linalg.norm(vec)
                    if norm < 1e-6:
                        vec = np.array([1.0, 0.0])
                        norm = 1.0
                    offset_dist = 0.09
                    candidate = anchor + (vec / norm) * offset_dist

                    for cn in conv_nodes:
                        d = np.linalg.norm(candidate - np.array(pos[cn]))
                        if d < 0.10:
                            push_vec = candidate - np.array(pos[cn])
                            push_norm = np.linalg.norm(push_vec)
                            if push_norm < 1e-8:
                                push_vec = vec / norm
                            else:
                                push_vec = push_vec / push_norm
                            candidate = np.array(pos[cn]) + push_vec * 0.11

                    pos[sn] = candidate
                    placed.add(sn)
                elif _pass == 2:
                    base_angle = math.pi * 0.6 + gi * 0.8
                    pos[sn] = center + np.array([conv_radius * 0.5 * math.cos(base_angle),
                                                  conv_radius * 0.5 * math.sin(base_angle)])
                    placed.add(sn)

    special_list = list(all_special)
    for _repel in range(5):
        moved = False
        for i, sn_a in enumerate(special_list):
            for sn_b in special_list[i+1:]:
                d = np.linalg.norm(np.array(pos[sn_a]) - np.array(pos[sn_b]))
                if d < 0.10:
                    mid = (np.array(pos[sn_a]) + np.array(pos[sn_b])) / 2
                    vec = np.array(pos[sn_a]) - np.array(pos[sn_b])
                    if np.linalg.norm(vec) < 1e-8:
                        perp = np.array([0.05, 0.05])
                    else:
                        perp = vec / np.linalg.norm(vec)
                    push = perp * 0.06
                    pos[sn_a] = mid + push
                    pos[sn_b] = mid - push
                    moved = True
        if not moved:
            break

    return pos


def draw_opportunity_network(
    opp: Dict[str, Any],
    ax: plt.Axes,
    title: str = "",
    seed: int = 42,
) -> None:
    G = nx.Graph()
    node_data: Dict[str, Dict] = {}
    for n in opp["nodes"]:
        stem = n["stem"]
        G.add_node(stem)
        node_data[stem] = n

    edge_data: Dict[Tuple[str, str], Dict] = {}
    for e in opp["edges"]:
        u, v = e["node_pair"]
        if G.has_node(u) and G.has_node(v):
            G.add_edge(u, v)
            key = tuple(sorted([u, v]))
            edge_data[key] = e

    strengths = [nd["strength"] for nd in node_data.values()]
    min_s, max_s = min(strengths), max(strengths)

    conv_con_es = [
        ed.get("con_e", 0) for ed in edge_data.values()
        if ed["type"] == "conventional" and ed.get("con_e")
    ]
    max_con_e = max(conv_con_es) if conv_con_es else 1.0

    pos = _layout(G, node_data, seed)

    # ── 绘制边 ───────────────────────────────────────────────
    for (u, v), ed in edge_data.items():
        etype = ed["type"]
        xu, yu = pos[u]
        xv, yv = pos[v]

        if etype == "conventional":
            con_e = ed.get("con_e", 0)
            if con_e and con_e >= 5.0:
                lw = _edge_width(con_e, max_con_e)
                ax.plot([xu, xv], [yu, yv], color=COLOR_CONV_EDGE, linewidth=lw,
                        alpha=0.55, solid_capstyle="round", zorder=1)
                mx, my = (xu+xv)/2, (yu+yv)/2
                label = f"{con_e:.0f}" if con_e >= 10 else f"{con_e:.1f}"
                ax.annotate(
                    label, (mx, my), fontsize=5.5, color="#1B5E20", fontweight="bold",
                    ha="center", va="center",
                    bbox=dict(boxstyle="square,pad=0.12", fc="white", ec=COLOR_CONV_EDGE,
                              alpha=0.85, linewidth=0.5),
                    zorder=6,
                )
            else:
                ax.plot([xu, xv], [yu, yv], color=COLOR_CONV_EDGE, linewidth=1.0,
                        alpha=0.40, solid_capstyle="round", zorder=1)

        elif etype == "new":
            ax.plot([xu, xv], [yu, yv], color=COLOR_NEW_EDGE, linewidth=1.0,
                    linestyle=(0, (4, 3)), alpha=0.5, zorder=1)
            yr = ed.get("year_e")
            if yr:
                mx, my = (xu+xv)/2, (yu+yv)/2
                ax.annotate(
                    str(yr), (mx, my), fontsize=5, color=COLOR_NEW_EDGE,
                    ha="center", va="center", fontstyle="italic",
                    bbox=dict(boxstyle="round,pad=0.12", fc="#E3F2FD", ec="none", alpha=0.85),
                    zorder=5,
                )

        elif etype == "special":
            ax.plot([xu, xv], [yu, yv], color=COLOR_SPE_EDGE, linewidth=1.0,
                    linestyle="-", alpha=0.45, zorder=1)
            mx, my = (xu+xv)/2, (yu+yv)/2
            ax.plot(mx, my, marker="^", color=COLOR_SPE_EDGE, markersize=7, zorder=7)
            yr = ed.get("year_e")
            if yr:
                ax.annotate(
                    str(yr), (mx, my), fontsize=5.5, color=COLOR_SPE_EDGE,
                    xytext=(5, 5), textcoords="offset points",
                    fontstyle="italic", fontweight="bold", zorder=6,
                )

    # ── 绘制节点（全部普通圆圈）──────────────────────────────
    conv_positions = np.array([pos[n] for n in G.nodes()
                               if node_data[n]["type"] == "conventional"])
    center = conv_positions.mean(axis=0) if len(conv_positions) else np.array([0.0, 0.0])

    for stem, nd in node_data.items():
        ntype = nd["type"]
        x, y = pos[stem]
        sz = _node_radius(nd["strength"], min_s, max_s)

        ax.scatter(x, y, s=sz, marker="o", c=COLOR_CONV_NODE,
                   edgecolors=COLOR_NODE_EDGE, linewidths=1.3, zorder=10)
        ax.annotate(
            stem, (x, y), fontsize=7.5, ha="center", va="center",
            fontweight="bold", color="#212121", zorder=15,
        )

        if ntype in ("new", "marginal"):
            vec = np.array([x, y]) - center
            angle = math.atan2(vec[1], vec[0])
            dx = 20 * math.cos(angle)
            dy = 20 * math.sin(angle)
            ha = "left" if dx > 2 else "right" if dx < -2 else "center"

            if ntype == "new":
                marker_char = "★"
                tag_color = COLOR_NEW
            else:
                marker_char = "◆"
                tag_color = COLOR_MARGINAL

            eigen_val = nd.get("eigen", 0)
            yr = nd.get("year_first", "")
            parts = [marker_char]
            if eigen_val is not None:
                parts.append(f"{eigen_val:.6f}")
            if yr:
                parts.append(str(yr))
            tag_text = "  ".join(parts)

            ax.annotate(
                tag_text, (x, y), fontsize=6, color=tag_color,
                xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=tag_color,
                          alpha=0.9, linewidth=0.6),
                arrowprops=dict(arrowstyle="-", color=tag_color, lw=0.6, alpha=0.5),
                zorder=16,
            )

    # ── 图例 ──────────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(facecolor="white", edgecolor=COLOR_NEW, linewidth=1.2,
                       label="★  new node"),
        mpatches.Patch(facecolor="white", edgecolor=COLOR_MARGINAL, linewidth=1.2,
                       label="◆  marginal node"),
        mlines.Line2D([], [], marker="o", color="w", markerfacecolor=COLOR_CONV_NODE,
                       markersize=9, markeredgecolor=COLOR_NODE_EDGE, label="conventional node"),
        mlines.Line2D([], [], color=COLOR_NEW_EDGE, linewidth=1.2, linestyle="--", label="new edge"),
        mlines.Line2D([], [], color=COLOR_SPE_EDGE, linewidth=1.0, marker="^",
                       markersize=6, label="special edge"),
        mlines.Line2D([], [], color=COLOR_CONV_EDGE, linewidth=4, alpha=0.7,
                       label="conventional edge"),
    ]
    leg = ax.legend(handles=legend_elements, loc="lower right", fontsize=7,
                    framealpha=0.95, edgecolor="#9E9E9E", fancybox=True,
                    borderpad=0.8, labelspacing=0.6)
    leg.get_frame().set_linewidth(0.8)

    z = opp.get("z_score", 0)
    feats = opp.get("feature_scores", {})
    feat_str = "   ".join(f"{k} = {v:.4f}" for k, v in feats.items())
    ax.set_title(f"{title}\nZ = {z:.4f}     {feat_str}",
                 fontsize=10, fontweight="bold", pad=12)
    ax.set_axis_off()
    ax.margins(0.08)


def main():
    parser = argparse.ArgumentParser(description="绘制技术机会子网络图")
    parser.add_argument("--input", type=Path,
                        default=RAG_ENRICHED_JSON)
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        all_opps = json.load(f)

    opp_map = {o["opportunity_rank"]: o for o in all_opps}

    out_dir = args.output_dir or Path("outputs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    for rank in args.ranks:
        if rank not in opp_map:
            print(f"Rank {rank} 不存在，跳过")
            continue
        opp = opp_map[rank]

        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        fig.patch.set_facecolor("white")

        draw_opportunity_network(
            opp, ax,
            title=f"Opportunity {rank}",
            seed=args.seed + rank,
        )

        fig.tight_layout(pad=1.5)
        out_path = out_dir / f"opportunity_{rank}_network.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Rank {rank} -> {out_path}")


if __name__ == "__main__":
    main()
