# src/nlp_utils.py

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import spacy
from pathlib import Path
from nltk.stem import PorterStemmer

# 导入缓存工具
from .utils import cache as _cache_utils
get_nlp_cache = _cache_utils.get_nlp_cache

# 全局 Porter Stemmer 实例（论文 Section 3.2(2): "all nodes (words) are converted into their stems"）
_porter = PorterStemmer()

@dataclass
class DependencyEdge:
    head: str
    dependent: str
    relation: str

class NLPProcessor:
    # 句子级语言检测阈值：PROPN+X 占比超过此值判定为非英语句
    _NON_ENGLISH_SENT_THRESHOLD = 0.40
    # 全大写句子的大写字母占比阈值（超过则跳过语言检测，可能是英文标题）
    _UPPER_SKIP_RATIO = 0.80

    # 明确的非英语功能词集合（德/法/西/葡），用于句级语言检测的 override。
    # 仅收录在英文专利正文中不会单独出现的词；≥2 匹配即判定为非英语句。
    _NON_EN_FUNCTION_WORDS = frozenset({
        # 德语
        'ein', 'eine', 'einer', 'eines', 'einem', 'einen',
        'und', 'ist', 'sind', 'wird', 'werden', 'wurde', 'wurden',
        'auf', 'aus', 'von', 'zur', 'zum', 'vom',
        'nach', 'durch', 'oder', 'aber', 'wenn', 'dass', 'weil',
        'wobei', 'sowie', 'nicht', 'auch', 'nur', 'noch',
        'diese', 'dieser', 'dieses', 'diesem', 'diesen',
        'dabei', 'jeder', 'keine', 'kein', 'wenigstens',
        'der', 'den', 'dem', 'des', 'das',
        # 西班牙语
        'los', 'las', 'del', 'una',
        'que', 'por', 'como', 'con',
        'siendo', 'cuando', 'donde', 'cual',
        'desde', 'hasta', 'entre', 'sobre',
        # 法语
        'dans', 'avec', 'sont', 'ont',
        'cette', 'ces', 'leurs', 'nous',
    })
    _NON_EN_MIN_HITS = 2

    def __init__(self, model_name: str = "en_core_web_sm", use_cache: bool = True, cache_dir: Optional[Path] = None):
        """
        初始化NLP处理器
        
        Args:
            model_name: spaCy模型名称
            use_cache: 是否使用缓存
            cache_dir: 缓存目录
        """
        self.nlp = spacy.load(model_name, disable=["ner"])
        # 专利领域法律套话/泛化动词（论文 Section 3.2(2)：仅移除 stop words）
        # 注意：system/device/unit/method 等技术词不能作为停用词，
        # 论文 Table 8/10 中它们是技术机会的核心节点
        _custom_stop = {
            "claim", "provide", "invention", "include", "accord",
            "comprise", "realize", "relate", "base", "combine",
            "finish", "use", "obtain", "perform",
            "wherein", "thereof", "say",
            "establish", "improve", "construct",
            "disclose", "determine", "reduce",
        }
        self.stopwords = self.nlp.Defaults.stop_words | _custom_stop
        self.use_cache = use_cache
        self.cache = get_nlp_cache(cache_dir) if use_cache else None

    def nlp_cached(self, text: str, patent_id: str = None):
        """
        带缓存的 NLP 处理（用于特征提取，避免重复处理相同标题）
        
        注意：spaCy Doc 对象不能直接 pickle，所以只使用内存缓存（不写入文件）
        如果缓存出错，自动回退到直接调用 nlp()
        
        Args:
            text: 要处理的文本
            patent_id: 专利ID，用于缓存键（如果提供）
        
        Returns:
            spaCy Doc 对象
        """
        try:
            # 只使用内存缓存（spaCy Doc 对象不能 pickle）
            if self.use_cache and self.cache:
                # 直接访问内存缓存（跳过文件缓存）
                key = self.cache._get_key(text, patent_id)
                if key in self.cache.memory_cache:
                    cached_doc = self.cache.memory_cache[key]
                    # 验证缓存的对象是 spaCy Doc（检查是否有 'text' 属性和可迭代的 tokens）
                    # 如果缓存返回了错误类型（如 DependencyEdge 列表），忽略缓存
                    if (hasattr(cached_doc, 'text') and 
                        hasattr(cached_doc, '__iter__') and 
                        len(cached_doc) > 0 and
                        hasattr(cached_doc[0], 'is_punct')):
                        return cached_doc
                    # 缓存中存储了错误类型，删除它并重新计算
                    del self.cache.memory_cache[key]
        except Exception:
            # 缓存访问失败，继续执行 NLP 处理
            pass
        
        # 执行 NLP 处理
        doc = self.nlp(text)
        
        # 存入内存缓存（不写入文件，因为 Doc 对象不能 pickle）
        try:
            if self.use_cache and self.cache:
                key = self.cache._get_key(text, patent_id)
                # 只存入内存缓存
                if len(self.cache.memory_cache) >= self.cache.max_size:
                    # 简单的FIFO策略：删除最旧的条目
                    oldest_key = next(iter(self.cache.memory_cache))
                    del self.cache.memory_cache[oldest_key]
                self.cache.memory_cache[key] = doc
        except Exception:
            # 缓存存储失败，不影响返回结果
            pass
        
        return doc
    
    @staticmethod
    def _has_non_ascii_alpha(text: str) -> bool:
        """检查文本是否包含非 ASCII 字母（ä, ß, í, ñ, CJK 等），数字和标点不受影响。"""
        return any(c.isalpha() and ord(c) > 127 for c in text)

    # spaCy 仅对识别的英语功能词赋予这些 POS 标签，非英文词不会获得
    _EN_FUNCTION_POS = frozenset({'DET', 'ADP', 'CCONJ', 'AUX', 'PRON', 'PART', 'SCONJ'})

    def _is_likely_non_english_sentence(self, sent) -> bool:
        """
        句子级语言检测，三层判定：
        1. 非英语功能词 override：句中含 ≥_NON_EN_MIN_HITS 个已知非英语功能词
           → 直接判定为非英语（即使 spaCy 误标了英语 POS 也能捕获）
        2. 英语功能词安全阀：句中含英语功能词 POS → 判定为英语
        3. PROPN+X 占比阈值：超过阈值 → 判定为非英语
        """
        non_punct = [t for t in sent if not t.is_punct and not t.is_space]
        if len(non_punct) < 3:
            return False

        # 第 1 层: 非英语功能词 override（最高优先级）
        non_en_hits = sum(
            1 for t in non_punct
            if t.text.lower() in self._NON_EN_FUNCTION_WORDS
        )
        if non_en_hits >= self._NON_EN_MIN_HITS:
            return True

        # 第 2 层: 英语功能词安全阀
        en_func_count = sum(1 for t in non_punct if t.pos_ in self._EN_FUNCTION_POS)
        if en_func_count >= 1:
            return False

        # 第 3 层: 全大写 + 纯 ASCII → 英文标题
        text = sent.text
        alpha_chars = [c for c in text if c.isalpha()]
        if alpha_chars:
            upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
            if upper_ratio > self._UPPER_SKIP_RATIO and text.isascii():
                return False

        # 第 4 层: PROPN+X 占比
        propn_x = sum(1 for t in non_punct if t.pos_ in ('PROPN', 'X'))
        return propn_x / len(non_punct) > self._NON_ENGLISH_SENT_THRESHOLD

    def normalize_token(self, token) -> str:
        """
        论文 Section 3.2(2): "all nodes (words) are converted into their stems"
        使用 Porter Stemmer 进行词干化（非 lemmatization），与论文一致。
        对于包含 "-" 的词汇，将各部分分别词干化后拼回。
        含非 ASCII 字母的 token 返回空串（语言核验）。
        """
        if token.is_punct:
            return ""

        token_text = token.text

        if self._has_non_ascii_alpha(token_text):
            return ""

        if "-" in token_text:
            parts = token_text.split("-")
            stemmed_parts = []
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                if self._has_non_ascii_alpha(part):
                    return ""
                stemmed = _porter.stem(part)
                if stemmed:
                    stemmed_parts.append(stemmed)
            return "-".join(stemmed_parts) if stemmed_parts else ""

        return _porter.stem(token_text.lower().strip())
    
    def is_valid_token(self, token) -> bool:
        """
        检查token是否有效（非标点符号、非停用词、是字母或字母数字混合）
        标点符号等同于停用词，会被完全过滤。
        允许：缩写词（VLA, CNN）、字母数字混合（sim2real, word2vec）、
        含连字符的技术词（machine-learning, GPT-3, state-of-the-art）。
        
        Args:
            token: spaCy token对象
        
        Returns:
            是否是有效token
        """
        if token.is_punct:
            return False
        token_text = token.text

        # 非 ASCII 字母检测：拒绝 ä, ß, í, ñ 等（保留纯 ASCII 缩写如 105b, wlm）
        if self._has_non_ascii_alpha(token_text):
            return False

        # 允许字母词（is_alpha）或字母数字混合词（但不全是数字）
        if token.is_alpha or (token_text.isalnum() and not token_text.isdigit()):
            valid = True
        # 含连字符的词：各段需为字母数字，且至少有一段含字母（保留 GPT-3，过滤 12-34）
        elif "-" in token_text:
            parts = [p for p in token_text.split("-") if p.strip()]
            valid = (
                len(parts) >= 1
                and all(p.isalnum() for p in parts)
                and any(p.isalpha() for p in parts)
            )
        else:
            valid = False
        if not valid:
            return False
        # 过滤停用词（含自定义停用词）
        if token.is_stop:
            return False
        tok_lower = token.text.lower()
        tok_stem = _porter.stem(tok_lower)
        if tok_lower in self.stopwords or tok_stem in self.stopwords:
            return False
        return True
    
    def _is_valid_token(self, token) -> bool:
        """
        内部方法，保持向后兼容
        """
        return self.is_valid_token(token)

    def _bridge_edges(self, edges: List[DependencyEdge], doc) -> List[DependencyEdge]:
        """
        论文 Section 3.2(2) 补边机制：
        "The stop words and their corresponding nodes are then removed from the
        knowledge network, meaning that nodes that are neighbors of the stop words
        are now directly connected with each other."

        对原始依存树中被停用词隔开的有效节点对进行桥接：
        如果一个停用词/标点的 head 或 children 中有两个或以上有效节点，
        则这些有效节点两两互连。
        """
        bridged_edges = list(edges)
        existing_pairs: Set[Tuple[str, str]] = set()
        for e in edges:
            existing_pairs.add(tuple(sorted([e.head, e.dependent])))

        for sent in doc.sents:
            token_to_node = self._merge_hyphenated_tokens(sent)
            for token in sent:
                # 找到停用词/无效token（即将被删除的节点）
                if self._is_valid_token(token):
                    continue

                # 收集该停用词的邻居中所有有效节点
                neighbor_nodes: Set[str] = set()
                # head 方向
                if token.head != token and self._is_valid_token(token.head):
                    node_name = token_to_node.get(token.head)
                    if node_name is None:
                        node_name = self.normalize_token(token.head)
                    if node_name:
                        neighbor_nodes.add(node_name)
                # children 方向
                for child in token.children:
                    if self._is_valid_token(child):
                        node_name = token_to_node.get(child)
                        if node_name is None:
                            node_name = self.normalize_token(child)
                        if node_name:
                            neighbor_nodes.add(node_name)

                # 邻居两两互连
                neighbor_list = list(neighbor_nodes)
                for i in range(len(neighbor_list)):
                    for j in range(i + 1, len(neighbor_list)):
                        pair = tuple(sorted([neighbor_list[i], neighbor_list[j]]))
                        if pair not in existing_pairs:
                            bridged_edges.append(
                                DependencyEdge(
                                    head=neighbor_list[i],
                                    dependent=neighbor_list[j],
                                    relation="bridged",
                                )
                            )
                            existing_pairs.add(pair)

        return bridged_edges

    def _get_ancestor_descendant_path_distance(
        self, ancestor_tok, descendant_tok, max_hops: int = 3
    ) -> Optional[int]:
        """
        计算依存树中从 descendant 沿 head 链向上追溯到 ancestor 的路径长度。
        仅当 ancestor 是 descendant 的祖先时返回有效距离，否则返回 None。
        路径可经过停用词/标点，但只沿有向边（head -> dependent）计数。

        Args:
            ancestor_tok: 祖先 token
            descendant_tok: 后代 token
            max_hops: 最大允许跳数

        Returns:
            路径长度（边数），若超过 max_hops 或非祖先关系则返回 None
        """
        if ancestor_tok == descendant_tok:
            return 0
        dist = 0
        current = descendant_tok
        while current != ancestor_tok:
            if current.head == current:
                return None
            current = current.head
            dist += 1
            if dist > max_hops:
                return None
        return dist

    def _check_bridge_candidate(
        self,
        node1: str,
        node2: str,
        tokens1: List,
        tokens2: List,
        max_hops: int = 3,
    ) -> Optional[Tuple[str, str]]:
        """
        检查 (node1, node2) 是否满足 DAG 约束的桥接条件。
        若存在祖先-后代关系且路径 <= max_hops，返回 (head_node, dependent_node)；
        否则返回 None。边的方向始终为 Ancestor(Head) -> Descendant(Dependent)。
        """
        for t1 in tokens1:
            for t2 in tokens2:
                if t1 == t2:
                    continue
                # t1 是 t2 的祖先？
                dist = self._get_ancestor_descendant_path_distance(t1, t2, max_hops=max_hops)
                if dist is not None:
                    return (node1, node2)
                # t2 是 t1 的祖先？
                dist = self._get_ancestor_descendant_path_distance(t2, t1, max_hops=max_hops)
                if dist is not None:
                    return (node2, node1)
        return None

    def _merge_compound_tokens(self, sent, token_to_node: dict) -> dict:
        """
        合并 compound 关系的 token（如 machine learning -> machin-learn），
        各部分使用 Porter Stemmer 词干化后以 "-" 连接。
        """
        for token in sent:
            if token.dep_ != "compound":
                continue
            if not self._is_valid_token(token) or not self._is_valid_token(token.head):
                continue
            dep_stem = _porter.stem(token.text.lower().strip())
            head_stem = _porter.stem(token.head.text.lower().strip())
            if not dep_stem or not head_stem:
                continue
            merged = f"{dep_stem}-{head_stem}"
            token_to_node[token] = merged
            token_to_node[token.head] = merged
        return token_to_node

    def _merge_hyphenated_tokens(self, sent) -> dict:
        """
        合并句子中通过连字符连接的token，返回token到合并后节点名的映射。
        同时处理 compound 关系（如 machine learning -> machine-learning）。
        
        Args:
            sent: spaCy句子对象
        
        Returns:
            token到合并后节点名的映射字典
        """
        token_to_node = {}
        tokens = list(sent)
        i = 0
        processed = set()  # 记录已处理的token索引
        
        while i < len(tokens):
            if i in processed:
                i += 1
                continue
                
            token = tokens[i]
            
            # 检查是否是连字符连接的词的一部分
            # 模式: word1 - word2 或 word1 - word2 - word3
            if (i + 2 < len(tokens) and 
                tokens[i + 1].text == "-" and 
                self._is_valid_token(token) and 
                self._is_valid_token(tokens[i + 2])):
                
                # 收集连字符连接的词部分
                parts = [token]
                parts.append(tokens[i + 2])  # 添加连字符后的第一个词
                j = i + 3  # 跳过当前token、连字符和第一个词
                processed.add(i)
                processed.add(i + 1)  # 标记连字符为已处理
                processed.add(i + 2)  # 标记第一个词为已处理
                
                # 继续收集后续通过连字符连接的部分
                while j < len(tokens):
                    if tokens[j].text == "-" and j + 1 < len(tokens) and self._is_valid_token(tokens[j + 1]):
                        processed.add(j)  # 标记连字符
                        j += 1
                        if j < len(tokens) and self._is_valid_token(tokens[j]):
                            parts.append(tokens[j])
                            processed.add(j)
                            j += 1
                        else:
                            break
                    else:
                        break
                
                # 如果有多个部分，合并它们（使用 Porter Stemmer）
                if len(parts) > 1:
                    stemmed_parts = []
                    for part_token in parts:
                        part_stem = _porter.stem(part_token.text.lower().strip())
                        if part_stem:
                            stemmed_parts.append(part_stem)

                    if stemmed_parts:
                        merged_node = "-".join(stemmed_parts)
                        # 将所有部分映射到合并后的节点
                        for part_token in parts:
                            token_to_node[part_token] = merged_node
                        i = j  # 跳到下一个未处理的token
                        continue
            
            # 普通token，正常处理
            if self._is_valid_token(token) and token not in token_to_node:
                normalized = self.normalize_token(token)
                if normalized:
                    token_to_node[token] = normalized
            
            i += 1
        
        # 合并 compound 关系（如 machine learning -> machine-learning）
        token_to_node = self._merge_compound_tokens(sent, token_to_node)
        
        return token_to_node

    def extract_dependencies(self, text: str, patent_id: str = None) -> List[DependencyEdge]:
        """
        对一段文本（标题+摘要）做句法分析，抽取 (head, dep)。
        使用缓存避免重复处理相同文本。
        实现补边机制：去除停用词后，连接被停用词分隔的节点。
        
        Args:
            text: 要处理的文本
            patent_id: 专利ID，用于缓存键（如果提供）
        """
        # 检查缓存（优先使用patent_id）
        if self.use_cache and self.cache:
            cached_result = self.cache.get(text, patent_id)
            if cached_result is not None:
                return cached_result
        
        # 执行NLP处理
        doc = self.nlp(text)
        edges: List[DependencyEdge] = []
        for sent in doc.sents:
            # 句子级语言检测：跳过非英语句
            if self._is_likely_non_english_sentence(sent):
                continue

            # 先合并连字符词，获取token到节点的映射
            token_to_node = self._merge_hyphenated_tokens(sent)
            
            for token in sent:
                # 跳过ROOT节点
                if token.dep_ == "ROOT":
                    continue
                
                # 过滤标点符号（在spaCy处理之后）- 标点符号等同于停用词
                if not self._is_valid_token(token):
                    continue
                # 过滤head是标点符号的情况（标点符号等同于停用词）
                if not self._is_valid_token(token.head):
                    continue
                
                # 使用映射获取节点名（如果存在），否则使用normalize_token
                head = token_to_node.get(token.head)
                dep = token_to_node.get(token)
                
                # 如果映射中不存在，使用normalize_token（向后兼容）
                if head is None:
                    head = self.normalize_token(token.head)
                if dep is None:
                    dep = self.normalize_token(token)
                
                # 如果normalize_token返回空字符串（标点符号），跳过
                if not head or not dep:
                    continue
                # 再次检查停用词（虽然_is_valid_token已经检查过，但这里确保一致性）
                if head in self.stopwords or dep in self.stopwords:
                    continue
                
                # 如果head和dep相同（可能是连字符词内部的关系），跳过
                if head == dep:
                    continue
                
                edges.append(DependencyEdge(head=head, dependent=dep, relation=token.dep_))
        
        # 补边：连接被停用词分隔的节点
        edges = self._bridge_edges(edges, doc)
        
        # 存入缓存（优先使用patent_id）
        if self.use_cache and self.cache:
            self.cache.set(text, edges, patent_id)
        
        return edges
