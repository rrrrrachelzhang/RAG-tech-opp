"""
兼容层：历史代码若导入 utils.logger，转发到统一 logging_config。
"""

from pathlib import Path
from typing import Optional

from .logging_config import setup_project_logging


def setup_logger(
    log_dir: Optional[Path] = None,
    log_level: str = "INFO",
    rotation: str = "10 MB",
    retention: str = "7 days",
):
    """兼容旧入口，内部统一转发到 setup_project_logging。"""
    return setup_project_logging(
        log_dir=log_dir,
        log_level=log_level,
        rotation=rotation,
        retention=retention,
    )
