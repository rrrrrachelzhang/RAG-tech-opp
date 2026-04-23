"""
项目路径统一管理模块

集中定义所有路径常量，避免硬编码路径散落各处。
"""
from pathlib import Path

# 项目根目录（相对于此文件向上三级：utils -> patent_opportunity_analysis -> src -> code）
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# 数据目录
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_INTERIM_DIR = DATA_DIR / "interim"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

# 配置文件目录
CONFIG_DIR = PROJECT_ROOT / "configs"

# 输出目录
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
MODELS_CACHE_DIR = MODELS_DIR / "cache"  # HDKN统计缓存目录
LOGS_DIR = OUTPUTS_DIR / "logs"
CACHE_DIR = OUTPUTS_DIR / "cache"
NLP_CACHE_DIR = CACHE_DIR / "nlp"
MATPLOTLIB_CACHE_DIR = CACHE_DIR / "matplotlib"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# 常用文件路径
RAW_PATENT_FILE = DATA_RAW_DIR / "patents.csv"
RAW_PATENT_TEST_FILE = DATA_RAW_DIR / "patents_test.csv"
ACO_CONFIG_FILE = CONFIG_DIR / "aco_config.yaml"
RAG_CONFIG_FILE = CONFIG_DIR / "rag_config.yaml"

# RAG 数据目录（合并后的候选子网）
RAG_DATA_DIR = DATA_PROCESSED_DIR / "rag"
RAG_ENRICHED_JSON = RAG_DATA_DIR / "aco_merged_top30_enriched.json"
RAG_CANDIDATES_JSON = RAG_DATA_DIR / "aco_merged_top30_candidates.json"
RUN_MERGED_RAG_DIRNAME = "03_merged_rag"

# 运行目录（分步式Pipeline）
RUNS_DIR = OUTPUTS_DIR / "runs"


def get_run_merged_rag_dir(run_id: str) -> Path:
    """返回 run 内合并 RAG 数据目录。"""
    return RUNS_DIR / run_id / RUN_MERGED_RAG_DIRNAME


def get_run_merged_rag_enriched_json(run_id: str) -> Path:
    """返回 run 内合并后的富化子网 JSON 路径。"""
    return get_run_merged_rag_dir(run_id) / "aco_merged_top30_enriched.json"


def get_run_merged_rag_candidates_json(run_id: str) -> Path:
    """返回 run 内合并后的候选列表 JSON 路径。"""
    return get_run_merged_rag_dir(run_id) / "aco_merged_top30_candidates.json"


def get_run_merged_rag_summary_json(run_id: str) -> Path:
    """返回 run 内合并摘要 JSON 路径。"""
    return get_run_merged_rag_dir(run_id) / "merge_summary.json"

# 确保必要的目录存在
for dir_path in [DATA_RAW_DIR, DATA_INTERIM_DIR, DATA_PROCESSED_DIR, 
                 CONFIG_DIR, MODELS_DIR, MODELS_CACHE_DIR, LOGS_DIR, CACHE_DIR,
                 NLP_CACHE_DIR, MATPLOTLIB_CACHE_DIR, REPORTS_DIR, RUNS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)
