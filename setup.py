"""
专利技术机会分析系统 - 安装脚本
"""

from setuptools import setup, find_packages
from pathlib import Path

# 读取README
readme_file = Path(__file__).parent / "readme.md"
long_description = readme_file.read_text(encoding='utf-8') if readme_file.exists() else ""

# 读取requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = [
        line.strip()
        for line in requirements_file.read_text(encoding='utf-8').splitlines()
        if line.strip() and not line.startswith('#')
    ]

setup(
    name="patent-opportunity-analysis",
    version="1.0.0",
    description="基于专利数据的知识网络构建与技术机会发现系统",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/patent-analysis",
    packages=find_packages(where="."),
    package_dir={"": "."},
    python_requires=">=3.8",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    entry_points={
        "console_scripts": [
            "patent-analysis=src.patent_opportunity_analysis.pipeline:main",
            "dkn-run-pipeline=src.patent_opportunity_analysis.pipeline:main",
            "dkn-run-all=scripts.run_all:main",
        ],
    },
    # 包含非Python文件（如果需要）
    include_package_data=True,
    # 项目URLs
    project_urls={
        "Bug Reports": "https://github.com/yourusername/patent-analysis/issues",
        "Source": "https://github.com/yourusername/patent-analysis",
    },
)

