# src/utils/report_generator.py
"""
报告生成工具
生成PDF报告、JSON输出等
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
from loguru import logger
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import platform
import os

# 注册中文字体
def _register_chinese_fonts():
    """注册中文字体，支持中文显示"""
    try:
        system = platform.system()
        
        # 尝试注册常见的中文字体
        font_paths = []
        
        if system == "Darwin":  # macOS
            # macOS系统字体路径（按优先级排序）
            font_paths = [
                "/System/Library/Fonts/Supplemental/Songti.ttc",  # 宋体（新版本，优先）
                "/System/Library/Fonts/STSong.ttc",  # 宋体
                "/System/Library/Fonts/STHeiti Light.ttc",  # 黑体
                "/Library/Fonts/Arial Unicode.ttf",  # Arial Unicode（如果安装）
            ]
        elif system == "Windows":
            # Windows系统字体路径
            font_paths = [
                "C:/Windows/Fonts/simsun.ttc",  # 宋体
                "C:/Windows/Fonts/simhei.ttf",  # 黑体
                "C:/Windows/Fonts/msyh.ttc",  # 微软雅黑
            ]
        elif system == "Linux":
            # Linux系统字体路径
            font_paths = [
                "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",  # 文泉驿微米黑
                "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",  # 文泉驿正黑
                "/usr/share/fonts/truetype/arphic/uming.ttc",  # AR PL UMing
            ]
        
        # 尝试注册字体
        chinese_font_registered = False
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    # 对于TTC文件，可能需要指定字体索引（0通常是第一个字体）
                    # 注册字体（使用不同的名称）
                    pdfmetrics.registerFont(TTFont('ChineseFont', font_path))
                    # 对于粗体，尝试使用同一个字体文件（TTC可能包含多个字体）
                    try:
                        pdfmetrics.registerFont(TTFont('ChineseFontBold', font_path))
                    except:
                        # 如果注册粗体失败，使用普通字体作为粗体
                        pdfmetrics.registerFont(TTFont('ChineseFontBold', font_path))
                    chinese_font_registered = True
                    logger.info(f"✅ 成功注册中文字体: {font_path}")
                    break
                except Exception as e:
                    logger.debug(f"注册字体失败 {font_path}: {e}")
                    continue
        
        # 如果系统字体都不可用，尝试使用reportlab内置的字体
        if not chinese_font_registered:
            try:
                # 尝试使用reportlab的CJK字体（如果安装了reportlab的CJK支持）
                from reportlab.pdfbase.cidfonts import UnicodeCIDFont
                pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
                chinese_font_registered = True
                logger.info("使用reportlab内置CJK字体")
            except ImportError:
                logger.warning("无法注册中文字体，中文可能显示为方块")
                # 使用Helvetica作为回退（中文会显示为方块）
                return False
        
        return chinese_font_registered
    except Exception as e:
        logger.warning(f"注册中文字体时出错: {e}")
        return False

# 在模块加载时注册字体
_CHINESE_FONT_AVAILABLE = _register_chinese_fonts()

def generate_json_output(
    results: Dict,
    output_path: Path
) -> Path:
    """
    生成结构化JSON输出
    
    Args:
        results: 包含所有结果的字典
        output_path: 输出路径
    
    Returns:
        输出文件路径
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 准备JSON数据
    json_data = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'version': '1.0'
        },
        'summary': {
            'num_patents': len(results.get('patents', [])),
            'num_features': len(results.get('features', [])),
            'num_opportunities': len(results.get('opportunities', []))
        },
        'dkn_stats': {
            'hdkn': {
                'nodes': results.get('HDKN', {}).number_of_nodes() if hasattr(results.get('HDKN'), 'number_of_nodes') else 0,
                'edges': results.get('HDKN', {}).number_of_edges() if hasattr(results.get('HDKN'), 'number_of_edges') else 0
            },
            'pdkn': {
                'nodes': results.get('PDKN', {}).number_of_nodes() if hasattr(results.get('PDKN'), 'number_of_nodes') else 0,
                'edges': results.get('PDKN', {}).number_of_edges() if hasattr(results.get('PDKN'), 'number_of_edges') else 0
            }
        },
        'opportunities': []
    }
    
    # 处理机会数据（包含边和特征）
    PDKN = results.get('PDKN')
    for i, opp in enumerate(results.get('opportunities', []), 1):
        nodes = opp.get('nodes', [])
        
        # 提取子网络的边
        edges = []
        if PDKN and hasattr(PDKN, 'subgraph'):
            try:
                # 处理 DKNNetwork 对象：subgraph() 返回 DKNNetwork，需要访问 .graph 属性
                subg_result = PDKN.subgraph(nodes)
                if hasattr(subg_result, 'graph'):
                    # DKNNetwork 对象
                    subg = subg_result.graph
                else:
                    # 普通 nx.Graph 对象
                    subg = subg_result.copy() if hasattr(subg_result, 'copy') else subg_result
                
                for u, v, data in subg.edges(data=True):
                    edges.append({
                        'source': u,
                        'target': v,
                        'weight': data.get('weight', 0.0),
                        'patents': list(data.get('patents', set()))[:10]  # 限制数量
                    })
            except Exception as e:
                logger.warning(f"提取机会{i}的边时出错: {e}")
        
        # 提取节点特征
        node_features = []
        if PDKN:
            # 获取节点集合（处理 DKNNetwork 和普通 Graph）
            if hasattr(PDKN, 'graph'):
                # DKNNetwork 对象
                pdkn_nodes = set(PDKN.graph.nodes())
            else:
                # 普通 nx.Graph 对象
                pdkn_nodes = set(PDKN.nodes())
            
            for node in nodes:
                if node in pdkn_nodes:
                    # 获取节点数据
                    if hasattr(PDKN, 'graph'):
                        node_data = PDKN.graph.nodes[node]
                    else:
                        node_data = PDKN.nodes[node]
                    node_features.append({
                        'node': node,
                        'strength': node_data.get('strength', 0.0),
                        'patents': list(node_data.get('patents', set()))[:10]
                    })
        
        json_data['opportunities'].append({
            'rank': i,
            'nodes': nodes,
            'edges': edges,
            'node_features': node_features,
            'score': opp.get('score', 0.0),
            'size': opp.get('size', 0),
            'num_edges': len(edges)
        })
    
    # 处理特征数据（只保存前10条作为示例）
    if 'features' in results:
        json_data['sample_features'] = results['features'][:10]
    
    # 保存JSON到models目录
    models_dir = output_path.parent if output_path.name.endswith('.json') else output_path
    if models_dir.name != 'models':
        models_dir = models_dir / 'models'
    models_dir.mkdir(exist_ok=True)
    
    json_output_path = models_dir / "opportunities.json"
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    logger.success(f"JSON输出已保存到: {json_output_path}")
    return json_output_path

def generate_pdf_report(
    results: Dict,
    output_path: Path,
    regression_report_path: Optional[Path] = None,
    diagnostics_plot_path: Optional[Path] = None,
    convergence_plot_path: Optional[Path] = None
) -> Path:
    """
    生成PDF报告（保存到reports目录）
    
    Args:
        results: 包含所有结果的字典
        output_path: 输出路径
        regression_report_path: 回归报告路径
        diagnostics_plot_path: 回归诊断图路径
        convergence_plot_path: ACO收敛曲线路径
    
    Returns:
        输出文件路径
    """
    # 保存到reports目录
    if output_path.parent.name != 'reports':
        reports_dir = output_path.parent.parent / "reports" if output_path.parent.name == "models" else output_path.parent / "reports"
        reports_dir.mkdir(exist_ok=True)
        output_path = reports_dir / "final_report.pdf"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    doc = SimpleDocTemplate(str(output_path), pagesize=A4)
    story = []
    styles = getSampleStyleSheet()
    
    # 确定使用的字体
    chinese_font_name = 'ChineseFont' if _CHINESE_FONT_AVAILABLE else 'Helvetica'
    chinese_font_bold = 'ChineseFontBold' if _CHINESE_FONT_AVAILABLE else 'Helvetica-Bold'
    
    # 标题样式（使用中文字体）
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontName=chinese_font_name,
        fontSize=24,
        textColor=colors.HexColor('#1a1a1a'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontName=chinese_font_name,
        fontSize=16,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    # 正文样式（使用中文字体）
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontName=chinese_font_name,
        fontSize=10
    )
    
    # 标题
    story.append(Paragraph("专利技术机会分析报告", title_style))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
    story.append(Spacer(1, 0.3*inch))
    
    # 1. 执行摘要
    story.append(Paragraph("1. 执行摘要", heading_style))
    summary_data = [
        ['指标', '数值'],
        ['专利总数', str(len(results.get('patents', [])))],
        ['提取特征数', str(len(results.get('features', [])))],
        ['技术机会数', str(len(results.get('opportunities', [])))],
    ]
    
    if hasattr(results.get('HDKN'), 'number_of_nodes'):
        summary_data.append(['HDKN节点数', str(results['HDKN'].number_of_nodes())])
        summary_data.append(['HDKN边数', str(results['HDKN'].number_of_edges())])
        summary_data.append(['PDKN节点数', str(results['PDKN'].number_of_nodes())])
        summary_data.append(['PDKN边数', str(results['PDKN'].number_of_edges())])
    
    summary_table = Table(summary_data, colWidths=[3*inch, 3*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), chinese_font_bold),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('FONTNAME', (0, 1), (-1, -1), chinese_font_name),  # 表格内容使用中文字体
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 0.3*inch))
    
    # 2. 回归模型分析
    if regression_report_path and regression_report_path.exists():
        story.append(Paragraph("2. 回归模型分析", heading_style))
        story.append(Paragraph("回归模型已训练完成，详细分析请参考回归分析报告。", normal_style))
        
        if diagnostics_plot_path and diagnostics_plot_path.exists():
            try:
                img = Image(str(diagnostics_plot_path), width=5*inch, height=3.5*inch)
                story.append(img)
                story.append(Spacer(1, 0.2*inch))
            except Exception as e:
                logger.warning(f"无法添加诊断图到PDF: {e}")
        story.append(Spacer(1, 0.2*inch))
    
    # 3. 技术机会
    story.append(Paragraph("3. Top技术机会", heading_style))
    opportunities = results.get('opportunities', [])
    if opportunities:
        opp_data = [['排名', '节点数', '得分', '节点列表']]
        for i, opp in enumerate(opportunities[:10], 1):
            nodes_str = ', '.join(opp.get('nodes', [])[:5])
            if len(opp.get('nodes', [])) > 5:
                nodes_str += '...'
            opp_data.append([
                str(i),
                str(opp.get('size', 0)),
                f"{opp.get('score', 0.0):.4f}",
                nodes_str
            ])
        
        opp_table = Table(opp_data, colWidths=[0.8*inch, 0.8*inch, 1*inch, 3.4*inch])
        opp_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), chinese_font_bold),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTNAME', (0, 1), (-1, -1), chinese_font_name),  # 表格内容使用中文字体
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP')
        ]))
        story.append(opp_table)
    else:
        story.append(Paragraph("未找到技术机会。", normal_style))
    
    story.append(Spacer(1, 0.3*inch))
    
    # 4. ACO算法收敛
    if convergence_plot_path and convergence_plot_path.exists():
        story.append(Paragraph("4. ACO算法收敛曲线", heading_style))
        try:
            img = Image(str(convergence_plot_path), width=5*inch, height=3*inch)
            story.append(img)
        except Exception as e:
            logger.warning(f"无法添加收敛曲线到PDF: {e}")
    
    # 构建PDF
    doc.build(story)
    logger.success(f"PDF报告已保存到: {output_path}")
    return output_path

