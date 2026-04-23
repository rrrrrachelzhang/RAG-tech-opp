# src/patent_opportunity_analysis/pipeline.py

from typing import List, Dict, Optional
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from .utils.errors import (
    DataLoadingError, FeatureExtractionError,
)

from loguru import logger

from . import config as _config
from . import patent_graph as _patent_graph
from . import feature_extraction as _feature_extraction
from . import feature_registry as _feature_registry
extract_title_subnetwork = _feature_extraction.extract_title_subnetwork
from . import hdkn_stats as _hdkn_stats
from . import nlp_utils as _nlp_utils
from .utils.paths import RAW_PATENT_FILE as _RAW_PATENT_FILE

RAW_PATENT_FILE = _RAW_PATENT_FILE
HIST_END_YEAR = _config.HIST_END_YEAR

PatentRecord = _patent_graph.PatentRecord
build_or_load_hdkn_stats = _hdkn_stats.build_or_load_hdkn_stats
compute_features_for_subnetwork = _feature_registry.compute_features_for_subnetwork
FEATURE_REGISTRY = _feature_registry.FEATURE_REGISTRY
NLPProcessor = _nlp_utils.NLPProcessor

# 常见 2 字母国家/地区代码（用于区分 "Name,CC" 个人格式 vs "Org,City,State,CC" 机构地址）
_COUNTRY_CODES = frozenset({
    'IN', 'US', 'KR', 'CN', 'JP', 'EP', 'WO', 'SG', 'IL', 'GB', 'DE', 'FR',
    'CA', 'AU', 'RU', 'TW', 'HK', 'IT', 'ES', 'NL', 'SE', 'CH', 'AT', 'BE',
})

# 机构识别关键词（含中韩日常见机构后缀）
_ORG_KEYWORDS = [
    'corp', 'ltd', 'inc', 'co.', 'company', 'corporation', 'limited',
    'university', 'institute', 'lab', 'laboratory', 'academy',
    'research', 'center', 'centre', 'foundation', 'association',
    'group', 'holding', 'enterprises', 'technologies', 'systems',
    'international', 'national', 'federal', 'government', 'agency',
    'ministry', 'department', 'bureau', 'office', 'committee',
    'council', 'board', 'commission', 'authority', 'service',
    'pte', 'gmbh', 'ag',  # 新加坡/德瑞公司
    '有限公司', '股份有限公司', '주식회사',  # 中韩
]


def _classify_assignee_org_vs_personal(assignee_str: str) -> int:
    """
    判断专利权人为组织(1)或个人(0)。

    规则：
    1. 含机构关键词 → 组织(1)
    2. 含逗号但结构为「姓名,国家代码」且无机构词 → 个人(0)
       （如 "Prasanthi B,IN | Anil Kumar G,IN"）
    3. 含逗号且为机构地址格式（多段如 City,State,Zip） → 组织(1)
    4. 无机构词且无逗号 → 个人(0)
    """
    s = str(assignee_str or '').strip()
    if not s:
        return 0
    lower = s.lower()
    if any(kw in lower for kw in _ORG_KEYWORDS):
        return 1
    if ',' not in s:
        return 0
    segments = [seg.strip() for seg in s.split('|') if seg.strip()]
    if not segments:
        return 0
    for seg in segments:
        parts = [p.strip() for p in seg.split(',')]
        if len(parts) < 2:
            continue
        last = parts[-1].upper()
        if last not in _COUNTRY_CODES or len(last) != 2:
            return 1
        if len(parts) > 2:
            return 1
    return 0


def load_patents_from_csv(
    file_path: Path,
    limit: int = None,
    smart_select: bool = True,
    min_app_year: int = None,
) -> List[PatentRecord]:
    """从CSV文件加载专利数据

    Args:
        file_path: CSV文件路径
        limit: 限制数量
        smart_select: 是否智能选择（优先选择有引用数据的历史专利）
        min_app_year: 最小申请年份，过滤掉 app_year < min_app_year 的专利（None 表示不过滤）
    """
    try:
        logger.info(f"正在加载专利数据: {file_path}")

        # 先读取所有数据进行智能选择
        # Windows兼容：优先utf-8-sig（处理BOM），fallback到utf-8/gb18030
        try:
            full_df = pd.read_csv(file_path, encoding='utf-8-sig')
        except UnicodeDecodeError:
            try:
                full_df = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                # 最后尝试gb18030（中文Windows常见编码）
                full_df = pd.read_csv(file_path, encoding='gb18030')
        
        # 清理列名（去除空格、全角空格、隐藏字符）
        full_df.columns = full_df.columns.str.strip().str.replace('\u3000', ' ', regex=False)  # 全角空格
        full_df.columns = full_df.columns.str.replace('\ufeff', '', regex=False)  # BOM字符

        # 过滤 app_year < min_app_year 的专利
        if min_app_year is not None and 'app_year' in full_df.columns:
            before = len(full_df)
            full_df = full_df[full_df['app_year'] >= min_app_year]
            logger.info(f"过滤 app_year < {min_app_year}，保留 {len(full_df)} 条（去除 {before - len(full_df)} 条）")

        if smart_select and limit:
            # 智能选择策略：优先选择有引用数据的历史专利
            hist_end_year = 2022

            # 1. 选择历史专利（可以进入HDKN）
            hist_patents = full_df[full_df['app_year'] <= hist_end_year]

            # 2. 在历史专利中，优先选择有引用数据的
            cited_hist = hist_patents[(hist_patents['forward_cites'] > 0) | (hist_patents['backward_cites'] > 0)]
            uncited_hist = hist_patents[(hist_patents['forward_cites'] == 0) & (hist_patents['backward_cites'] == 0)]

            # 3. 补充未来专利（进入PDKN）
            future_patents = full_df[full_df['app_year'] > hist_end_year]

            # 组合选择
            selected_df = pd.concat([
                cited_hist,    # 有引用数据的历史专利
                uncited_hist,  # 无引用数据的历史专利
                future_patents # 未来专利
            ])

            # 限制数量
            df = selected_df.head(limit)
            logger.info(f"智能选择 {len(df)} 条专利数据（优先历史专利和有引用数据）")
        else:
            df = full_df.head(limit) if limit else full_df
            logger.info(f"成功读取 {len(df)} 行数据")
        
        patents = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="解析专利数据"):
            try:
                # 处理IPC分类号
                ipc_str = str(row.get('ipc_classes', ''))
                ipc_classes = [x.strip() for x in ipc_str.split('|') if x.strip()] if ipc_str and ipc_str != '-' else []
                
                # 处理被引用数量
                forward_cites = row.get('forward_cites', 0)
                if pd.isna(forward_cites) or forward_cites == '-':
                    forward_cites = 0
                else:
                    forward_cites = int(forward_cites)
                
                # 处理引用专利数
                bc_raw = row.get('backward_cites', 0)
                if isinstance(bc_raw, (int, float)) and not pd.isna(bc_raw):
                    backward_cites = int(bc_raw)
                else:
                    bc_str = str(bc_raw).strip()
                    if '|' in bc_str:
                        backward_cites = bc_str.count('|') + 1
                    elif bc_str.isdigit():
                        backward_cites = int(bc_str)
                    elif bc_str in ('', '-', 'nan', 'None'):
                        backward_cites = 0
                    else:
                        try:
                            backward_cites = int(float(bc_str))
                        except (ValueError, TypeError):
                            backward_cites = 0
                
                # 处理专利权人字段（用于Assignee控制变量）；缺失时标记为 "0"
                assignee = str(row.get('assignee', '')).strip()
                if not assignee:
                    assignee = "0"
                
                patent = PatentRecord(
                    patent_id=str(row.get('patent_id', '')),
                    title=str(row.get('title', '')),
                    abstract=str(row.get('abstract', '')),
                    app_year=int(row.get('app_year', 0)),
                    forward_cites=forward_cites,
                    backward_cites=backward_cites,
                    assignee_type='',  # 保留字段以兼容
                    ipc_classes=ipc_classes,
                    assignee=assignee  # 读取assignee字段
                )
                patents.append(patent)
            except Exception as e:
                logger.warning(f"解析专利行时出错: {e}, 跳过该行")
                continue
        
        logger.success(f"成功加载 {len(patents)} 条专利")
        return patents
    except Exception as e:
        logger.error(f"加载专利数据失败: {e}")
        raise DataLoadingError(f"无法加载专利数据: {e}") from e

def extract_features_for_regression(
    HDKN,
    PDKN,
    patents: List[PatentRecord],
    selected_features: Optional[List[str]] = None,
    hist_end_year: Optional[int] = None,
    target_patents: Optional[List[PatentRecord]] = None,
    decay_factor: Optional[float] = None,
    force_rebuild_hdkn_stats: bool = False,
) -> List[Dict]:
    """
    为回归模型提取特征
    
    【核心原则】回归样本与 HDKN 不重合：使用 app_year = hist_end_year + 1 的专利，
    基于 HDKN（历史网络，app_year <= hist_end_year）提取特征，预测其被引用数。
    避免训练数据泄露，确保「用历史知识结构预测未来专利表现」。
    
    【网络使用矩阵】
    - HDKN: 包含 app_year <= hist_end_year 的专利
    - 回归样本: app_year == hist_end_year + 1（不在 HDKN 中）
    - 特征提取: 对每个样本专利，独立构建其子图 G_current，与 HDKN 对比计算 New_n/Con_e 等
    
    Args:
        HDKN: HDKN网络对象（DKNNetwork 或 nx.Graph）
        PDKN: PDKN网络对象（DKNNetwork 或 nx.Graph，当前未使用但保留以备将来扩展）
        patents: 专利列表（用于 IPC 索引构建等）
        selected_features: 要计算的特征列表（默认全部）
        hist_end_year: HDKN 截止年份。如果为 None，
            优先从 HDKN.hist_end_year 获取，否则使用 config.HIST_END_YEAR。
        target_patents: 指定要提取特征的专利子集。如果为 None，
            默认选取 app_year == hist_end_year + 1 的专利（与 HDKN 不重合）。
        decay_factor: 时间衰减因子 α。若提供，将覆盖 config.DECAY_FACTOR，
            用于 HDKN 统计缓存的 decay_factor 配置（α 选择实验时使用）。
        force_rebuild_hdkn_stats: 是否强制重建 HDKN 统计缓存（忽略磁盘缓存）。
            α 选择实验时，对每个 α 需设为 True 以基于新权重重新计算。
    """
    try:
        # 断言：确保传入的是 HDKN
        if hasattr(HDKN, 'assert_kind'):
            HDKN.assert_kind("HDKN")

        # ---- 解析实际的 hist_end_year ----
        if hist_end_year is None:
            if hasattr(HDKN, 'hist_end_year'):
                hist_end_year = HDKN.hist_end_year
            else:
                hist_end_year = HIST_END_YEAR
        logger.info(f"hist_end_year={hist_end_year}（用于 IPC 索引和缓存校验）")
        
        nlp = NLPProcessor()
        features = []
        
        # ---- 选择回归样本（论文: app_year = t+1, 或 t+1 和 t+2）----
        # 论文 Section 3.2(3): "if the number of patents in year t+1 is inadequate
        # for regression analysis, we consider adding the patents in year t+2"
        MIN_REGRESSION_SAMPLES = 50
        target_year_1 = hist_end_year + 1
        target_year_2 = hist_end_year + 2
        if target_patents is not None:
            sample_patents = target_patents
            logger.info(f"使用调用方指定的 target_patents（{len(sample_patents)} 条）")
        else:
            sample_patents = [p for p in patents if p.app_year == target_year_1]
            if len(sample_patents) < MIN_REGRESSION_SAMPLES:
                extra = [p for p in patents if p.app_year == target_year_2]
                if extra:
                    logger.info(
                        f"app_year={target_year_1} 仅 {len(sample_patents)} 条，"
                        f"合并 app_year={target_year_2} 的 {len(extra)} 条（论文 Section 3.2(3)）"
                    )
                    sample_patents.extend(extra)
            if not sample_patents:
                raise FeatureExtractionError(
                    f"没有找到 app_year={target_year_1} 或 {target_year_2} 的专利"
                    f"（hist_end_year={hist_end_year}）。"
                )
            logger.info(f"回归样本: {len(sample_patents)} 条（与 HDKN 不重合）")
        
        logger.info(f"为 {len(sample_patents)} 条专利提取特征...")
        logger.info(f"使用HDKN（历史网络，{HDKN.number_of_nodes()}节点）提取特征，样本基于 HDKN 做对比")
        
        # 获取底层的 graph 对象（如果 HDKN 是 DKNNetwork）
        hdkn_graph = HDKN.graph if hasattr(HDKN, 'graph') else HDKN
        
        # 提前确定 selected_features，用于决定是否计算耗时特征（Betweenness/Eigen/Constraint）
        # 默认不计算 Betweenness、Avg_Weight（见 config.DEFAULT_SELECTED_FEATURES）
        _feature_registry._register_feature_functions()
        if selected_features is None:
            _selected = getattr(_config, 'DEFAULT_SELECTED_FEATURES', None) or list(FEATURE_REGISTRY.keys())
        else:
            invalid = [f for f in selected_features if f not in FEATURE_REGISTRY]
            if invalid:
                raise ValueError(f"无效的特征名: {invalid}. 可用: {list(FEATURE_REGISTRY.keys())}")
            _selected = selected_features
        _need_betweenness = "Betweenness" in _selected
        _need_eigen = "Eigen" in _selected
        _need_constraint = "Constraint" in _selected
        
        # 构建或加载HDKN统计缓存（全局预计算，性能关键）
        # 衰减参考年 = 网络快照截断年（hist_end_year），与原文定义一致
        ref_year = hist_end_year
        logger.info("📊 构建/加载HDKN统计缓存（一次性预计算所有图级特征）...")
        try:
            _decay = decay_factor if decay_factor is not None else getattr(_config, 'DECAY_FACTOR', 0.9)
            config_dict = {
                'hdkn_end_year': hist_end_year,
                'decay_ref_year': ref_year,  # 衰减参考年=网络快照截断年，用于缓存键区分
                'decay_factor': _decay,
                'node_strength_mode': getattr(_config, 'NODE_STRENGTH_MODE', 'weighted_degree'),
                'skip_betweenness': not _need_betweenness,
                'skip_eigen': not _need_eigen,
                'skip_constraint': not _need_constraint,
            }
            node_stats_df, edge_stats_df, auxiliary_dicts = build_or_load_hdkn_stats(
                HDKN,
                config=config_dict,
                force_rebuild=force_rebuild_hdkn_stats
            )
            logger.success("✅ HDKN统计缓存加载/构建完成")
        except Exception as e:
            logger.error(f"构建/加载HDKN统计缓存失败: {e}")
            raise FeatureExtractionError(f"无法构建/加载HDKN统计缓存: {e}") from e
        
        selected_features = _selected
        logger.info(f"将计算以下特征: {selected_features}")
        
        # 预计算IPC索引（论文: "in the last 5 years of the HDKN's time period"）
        total_pat_start_year = hist_end_year - 4  # 最后 5 年
        hist_patents_5yr = [
            p for p in patents
            if total_pat_start_year <= p.app_year <= hist_end_year
        ]
        ipc_index = {}  # ipc_prefix -> set(patent_ids)
        for hist_patent in hist_patents_5yr:
            if hist_patent.ipc_classes:
                for ipc in hist_patent.ipc_classes:
                    if ipc and len(ipc) >= 4:
                        prefix = ipc[:4]
                        if prefix not in ipc_index:
                            ipc_index[prefix] = set()
                        ipc_index[prefix].add(hist_patent.patent_id)
        logger.debug(
            f"IPC索引构建完成: {len(ipc_index)} 个IPC前缀 "
            f"（{total_pat_start_year}-{hist_end_year} 年，{len(hist_patents_5yr)} 条专利）"
        )
        
        hdkn_cache = {
            'hdkn_graph': hdkn_graph,
            'hist_end_year': hist_end_year,
            'node_stats_df': node_stats_df,
            'edge_stats_df': edge_stats_df,
            'p90_year_n': auxiliary_dicts['p90_year_n'],
            'p90_year_e': auxiliary_dicts['p90_year_e'],
            'min_pn_mode': getattr(_config, 'MIN_PN_MODE', 'greedy'),
            '_strength_dict': node_stats_df['strength'].to_dict(),
            '_eigen_dict': node_stats_df['eigen'].to_dict(),
            '_constraint_dict': node_stats_df['constraint'].to_dict(),
            '_betweenness_dict': node_stats_df['betweenness'].to_dict(),
            '_year_min_node_dict': node_stats_df['year_min_node'].to_dict(),
            '_weight_dict': edge_stats_df['weight'].to_dict(),
            '_year_min_edge_dict': edge_stats_df['year_min_edge'].to_dict(),
            '_node_patents': {
                node: hdkn_graph.nodes[node].get("patents", set())
                for node in hdkn_graph.nodes()
            },
            '_edge_patents': {
                tuple(sorted([u, v])): data.get("patents", set())
                for u, v, data in hdkn_graph.edges(data=True)
            },
        }
        logger.debug("hdkn_cache dict 查找表构建完成")

        for patent in tqdm(sample_patents, desc="提取特征"):
            try:
                # 论文 Ren & Zhao (2021) Section 3.2(4):
                # 仅用标题文本 → NLP 解析 → 词干化 → 在 HDKN 中查找节点
                # → 取 induced subgraph（S_i）→ 基于 HDKN 全局属性计算特征
                G_current, title_stems, title_pairs = extract_title_subnetwork(
                    patent_title=patent.title or "",
                    hdkn=hdkn_graph,
                    nlp_processor=nlp,
                    patent_id=patent.patent_id,
                )

                if G_current.number_of_nodes() == 0:
                    continue

                subnetwork_features = compute_features_for_subnetwork(
                    G_current=G_current,
                    hdkn_cache=hdkn_cache,
                    selected_features=selected_features,
                    current_patent_id=patent.patent_id,
                    current_year=patent.app_year,
                    extra_ctx={
                        "title_stems": title_stems,
                        "title_pairs": title_pairs,
                    },
                )
                
                # 提取特征值（兼容旧代码结构）
                new_n = subnetwork_features.get("New_n", 0)
                new_e = subnetwork_features.get("New_e", 0)
                min_pn = subnetwork_features.get("Min_pn", 0)
                con_n = subnetwork_features.get("Con_n", 0.0)
                con_e = subnetwork_features.get("Con_e", 0.0)
                eigen = subnetwork_features.get("Eigen", 0.0)
                constraint_feature = subnetwork_features.get("Constraint", 1.0)
                betweenness_feature = subnetwork_features.get("Betweenness", 0.0)
                avg_weight_feature = subnetwork_features.get("Avg_Weight", 0.0)
                
                # 计算控制变量（仅用于回归模型，不用于ACO评估）
                back_cite = patent.backward_cites

                # Assignee: 从专利权人字段判断组织(1) vs 个人(0)
                assignee_value = _classify_assignee_org_vs_personal(
                    getattr(patent, 'assignee', '') or ''
                )

                # Total_pat: HDKN中相同IPC类别的专利总数（不包括当前专利）
                # 控制变量：衡量技术领域的竞争程度
                # 使用预构建的IPC索引优化性能
                total_pat = 0
                if patent.ipc_classes:
                    # 收集当前专利的IPC前缀（取前4位，如B25J等）
                    current_ipc_prefixes = set()
                    for ipc in patent.ipc_classes:
                        if ipc and len(ipc) >= 4:
                            current_ipc_prefixes.add(ipc[:4])
                    
                    if current_ipc_prefixes:
                        # 使用预构建的IPC索引快速查找匹配的专利
                        matching_patents = set()
                        for prefix in current_ipc_prefixes:
                            if prefix in ipc_index:
                                matching_patents.update(ipc_index[prefix])
                        # 排除当前专利本身
                        matching_patents.discard(patent.patent_id)
                        total_pat = len(matching_patents)

                features.append({
                    'patent_id': patent.patent_id,
                    'New_n': new_n,
                    'New_e': new_e,
                    'Min_pn': min_pn,
                    'Con_n': con_n,
                    'Con_e': con_e,
                    'Eigen': eigen,
                    'Constraint': constraint_feature,
                    'Betweenness': betweenness_feature,
                    'Avg_Weight': avg_weight_feature,
                    'Cited': patent.forward_cites,
                    'Back_cite': back_cite,  # 控制变量
                    'Assignee': assignee_value,  # 控制变量：组织(1) vs 个人(0)
                    'Total_pat': total_pat  # 控制变量：相同IPC类别的专利数
                })
            except Exception as e:
                logger.warning(f"提取专利 {patent.patent_id} 特征时出错: {e}, 跳过")
                continue
        
        logger.success(f"成功提取 {len(features)} 条专利的特征")
        
        # 检查特征数量
        if len(features) == 0:
            logger.error("⚠️  警告：没有成功提取任何特征！可能所有专利的子图都为空或处理失败")
            raise FeatureExtractionError("特征提取失败：没有成功提取任何特征")
        
        # 检查特征字段
        if features:
            sample_feature = features[0]
            missing_keys = [key for key in ['New_n', 'Min_pn', 'Con_e', 'Eigen', 'Cited'] if key not in sample_feature]
            if missing_keys:
                logger.error(f"⚠️  警告：特征字典缺少字段: {missing_keys}")
                logger.error(f"   样本特征字段: {list(sample_feature.keys())}")
                raise FeatureExtractionError(f"特征提取失败：缺少必需字段 {missing_keys}")
        
        return features
    except Exception as e:
        logger.error(f"特征提取失败: {e}")
        raise FeatureExtractionError(f"特征提取过程出错: {e}") from e

def run_full_pipeline(
    data_file: Path = None,
    limit: int = None,
    output_dir: Path = None
) -> Dict:
    """
    兼容旧入口，转发到分步编排器。
    
    Args:
        data_file: 数据文件路径，如果为None则使用配置中的路径
        limit: 限制加载的专利数量
        output_dir: 旧参数，已不再支持
    
    Returns:
        分步编排器的结果字典
    """
    if output_dir is not None:
        raise ValueError(
            "run_full_pipeline 不再支持自定义 output_dir。"
            "当前统一写入 outputs/runs/<run_id>/...；如需控制目录，请直接调用分步脚本。"
        )

    from scripts.run_all import run_all_pipeline

    logger.warning(
        "run_full_pipeline 已降级为兼容入口，内部转发到 scripts.run_all.run_all_pipeline。"
    )
    return run_all_pipeline(
        patents_csv=data_file,
        limit=limit,
        resume=True,
    )

def main():
    """
    命令行入口函数
    用于 setup.py 的 entry_points 配置
    
    Returns:
        int: 退出码（0表示成功，1表示失败）
    """
    import argparse
    from .utils.logging_config import setup_project_logging
    from scripts.run_all import run_all_pipeline
    
    # 初始化日志系统
    setup_project_logging()
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="专利技术机会分析系统（兼容入口，内部转发到分步编排器）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  patent-analysis                    # 使用默认配置运行完整分步流程
  patent-analysis --limit 1000       # 处理1000条数据
  patent-analysis --run-id exp_001   # 指定 run_id
        """
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='限制处理的专利数量'
    )
    parser.add_argument(
        '--data-file',
        type=str,
        default=None,
        help='数据文件路径（兼容旧参数，映射到 patents_csv）'
    )
    parser.add_argument(
        '--run-id',
        type=str,
        default=None,
        help='运行 ID'
    )
    parser.add_argument(
        '--hist-end-year',
        type=int,
        default=None,
        help='历史截止年份'
    )
    parser.add_argument(
        '--max-year',
        type=int,
        default=None,
        help='最大年份'
    )
    parser.add_argument(
        '--skip-step4',
        action='store_true',
        help='跳过 Step4'
    )
    
    args = parser.parse_args()
    
    data_file = Path(args.data_file) if args.data_file else None
    try:
        results = run_all_pipeline(
            patents_csv=data_file,
            limit=args.limit,
            run_id=args.run_id,
            hist_end_year=args.hist_end_year,
            max_year=args.max_year,
            skip_step4=args.skip_step4,
        )
        logger.success(f"\n流程执行完成，运行目录: {results['run_dir']}")
        return 0
    except Exception:
        logger.exception("流程执行失败")
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
