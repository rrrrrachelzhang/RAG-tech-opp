#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试回归脚本 - 使用小数据量验证功能
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.patent_opportunity_analysis.utils.logging_config import setup_project_logging
setup_project_logging(log_level="INFO")

from loguru import logger

sys.path.insert(0, str(project_root / "scripts"))
from scripts.step2_hdkn_regression import step2_hdkn_regression


def test_with_limited_data():
    """使用限制数据量测试 Step2 回归流程"""
    logger.info("测试模式：使用限制数据量")

    from src.patent_opportunity_analysis.utils.run_utils import generate_run_id, ensure_run_dirs
    run_id = generate_run_id(prefix="test_reg")
    ensure_run_dirs(run_id)

    try:
        step2_hdkn_regression(run_id=run_id)
        logger.success("测试成功")
    except Exception as e:
        logger.exception(f"测试失败: {e}")
        raise


if __name__ == "__main__":
    try:
        test_with_limited_data()
    except KeyboardInterrupt:
        logger.warning("用户中断")
    except Exception as e:
        logger.exception(f"错误: {e}")
        sys.exit(1)
