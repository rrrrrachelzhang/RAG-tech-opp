# src/patent_opportunity_analysis/config.py

from .utils.paths import (
    PROJECT_ROOT, DATA_DIR, DATA_RAW_DIR, CONFIG_DIR, OUTPUTS_DIR,
    MODELS_DIR, LOGS_DIR, CACHE_DIR, REPORTS_DIR,
    RAW_PATENT_FILE, RAW_PATENT_TEST_FILE, ACO_CONFIG_FILE
)

# DKN 参数
DECAY_FACTOR = 0.9      # α 衰减因子（论文 Section 4.2: "setting the decay factor to 0.9"）
HIST_END_YEAR = 2022    # HDKN 截止年份（历史数据划分点）
# 注意：回归训练样本使用所有 <= HIST_END_YEAR 的专利（即HDKN中的专利）
SUBNETWORK_SIZE = 15    # ACO 搜索的子网节点数 N

# 特征计算模式配置（严格复现版 vs 工程版）
# 严格复现版：完全符合 Ren & Zhao (2021) 原文定义
# 工程版：简化实现，提高计算效率

# New_n/New_e 阈值模式
# - "quantile_90": 使用 HDKN 中 Year_n/Year_e 的 90% 分位数（严格版，默认）
# - "hist_end_year": 使用 HIST_END_YEAR 作为固定阈值（工程版）
NOVELTY_THRESHOLD_MODE = "quantile_90"

# Min_pn 计算模式
# - "setcover_greedy": 贪心 set cover 算法（严格版，默认）
# - "union": 节点和边专利集合的并集大小（工程版，近似）
MIN_PN_MODE = "setcover_greedy"

# 中心性计算数据源
# - "HDKN": 使用 HDKN 计算中心性（严格版，默认）
# - "PDKN": 使用 PDKN 计算中心性（工程版，用于对比实验）
CENTRALITY_SOURCE = "HDKN"

# 节点 strength 计算模式
# - "weighted_degree": 加权度 = sum(incident edge weights)
# - "time_decay": 时间衰减权重之和 = sum(α^(T-year))（与边权完全平行的映射逻辑，默认）
NODE_STRENGTH_MODE = "time_decay"

# 默认参与回归的特征（不含 Betweenness、Avg_Weight，二者不计算以节省时间）
DEFAULT_SELECTED_FEATURES = [
    "New_n", "New_e", "Min_pn", "Con_n", "Con_e", "Eigen", "Constraint",
]

# Betweenness 计算优化（大图 k 采样，越小越快、精度越低）
BETWEENNESS_K_SAMPLE = 50   # 大图(>2000节点)时采样节点数，50 约 1 分钟/次，可调大提高精度

# Constraint 计算优化（性能相关）
# - SKIP_CONSTRAINT_FOR_LARGE_GRAPH: 对于大图（>LARGE_GRAPH_THRESHOLD）跳过 Constraint 计算
#   默认 False（启用计算），设置为 True 可跳过大图计算以加速
SKIP_CONSTRAINT_FOR_LARGE_GRAPH = False  # 默认 False，启用 Constraint 计算
LARGE_GRAPH_THRESHOLD = 5000  # 节点数阈值（超过此值且 SKIP_CONSTRAINT_FOR_LARGE_GRAPH=True 时跳过）
ALLOW_CONSTRAINT_FALLBACK = True  # 默认 True，允许回退到旧实现；测试模式设为 False 禁止回退

# ACO 参数（论文中的设置为参考）:contentReference[oaicite:1]{index=1}
ACO_NUM_ANTS = 300
ACO_NUM_GENERATIONS = 200
ACO_TOP_K_PER_GEN = 50
ACO_PHEROMONE_INIT = 1.0
ACO_PHEROMONE_ALPHA = 3      # pheromone factor
ACO_HEURISTIC_BETA = 4       # heuristic factor
ACO_RHO = 0.95               # pheromone retention rate (ρ=0.95 means 5% evaporates)
ACO_TAU_MIN = 0.01
ACO_TAU_MAX = 4.0
TOP_K_OPPORTUNITIES = 10     # 最终取前 K 个机会
