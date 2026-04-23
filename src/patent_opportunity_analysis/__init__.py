"""
patent_opportunity_analysis package
专利技术机会分析系统主包
"""

import os
from pathlib import Path

# Ensure matplotlib and font caches are writable inside the project (CLI environment blocks home dir writes)
from .utils.paths import PROJECT_ROOT, CACHE_DIR
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
(CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)

# 导入所有模块（供外部使用）
from . import config
from . import nlp_utils
from . import patent_graph
from . import dkn_builder
from . import feature_extraction
from . import regression_model
from . import aco_search
from . import aco_to_rag
from . import pipeline

__all__ = [
    'config', 'nlp_utils', 'patent_graph', 'dkn_builder',
    'feature_extraction', 'regression_model', 'aco_search',
    'aco_to_rag', 'pipeline'
]
