"""
ChatHumanScore Benchmark

Measures the "human-like" quality of conversational AI responses in casual chat scenarios.
Evaluates language naturalness, emotional alignment, diversity, context cohesion, and human signals.

评估标准详细说明 (Evaluation Criteria):

1. nat_score (自然度分数, 0-1): 评估回复语言的自然性和口语化程度
   - 基础分数: 0.5
   - 加分项:
     * 中文标点符号 [。！？] (+0.1 per pattern)
     * 语气词 [啊呀吧呢] (+0.1 per pattern)  
     * 感叹词/语气词 [嗯嗯哦哈] (+0.1 per pattern)
   - 减分项:
     * 过于正式的表达 (-0.1 per pattern):
       - "根据您的|按照您的"
       - "非常感谢您|感谢您的" 
       - "我将为您|我会为您"
   - 语法错误惩罚: 超过5%错误率时扣分

2. aff_score (情感对齐分数, 0-1): 评估回复与用户情感的匹配程度
   - 基于情感词典进行简单情感分析
   - 正面词汇: ['好', '棒', '赞', '开心', '高兴']
   - 负面词汇: ['不好', '糟糕', '难过', '生气', '失望']
   - 评分逻辑:
     * 用户和回复情感方向一致: 0.9
     * 用户和回复情感方向相反: 0.3
     * 双方都中性: 0.7
     * 一方中性: 0.6

3. div_score (多样性分数, 0-1): 评估回复的词汇丰富度和表达多样性
   - 计算公式: 0.6 × (独特词数/总词数) + 0.4 × (独特双词组合数/总双词组合数)
   - 词汇重复率惩罚: 超过30%重复率时扣分
   - 最小词数要求: 少于3个词时给予较低分数(0.3)

4. ctx_score (上下文连贯性分数, 0-1): 评估回复与对话上下文的相关性
   - 使用句子嵌入模型计算语义相似度
   - 提取最近一次用户发言作为上下文
   - 计算用户发言和AI回复的余弦相似度
   - 相似度映射: (cos_sim + 1) / 2 转换到0-1区间
   - 默认分数: 0.7 (当无法计算时)

5. hum_score (人类信号强度分数, 0-1): 评估回复中人类化特征的丰富程度
   - 基础分数: 0.3
   - 人类化特征检测:
     * 填充词 (+0.1 per word, 最多+0.3): ['呃', '嗯', '那个', '就是', '其实']
     * 表情符号 (+0.1 per emoji, 最多+0.2): Unicode表情符号
     * 随意表达 (+0.1 per pattern):
       - 笑声: '哈哈|嘿嘿|呵呵'
       - 语气词: '吧[，。！？]', '呀[，。！？]'
     * 个人观点表达 (+0.1 per phrase, 最多+0.2): ['我觉得', '我想', '我感觉', '个人认为']

综合评分计算:
- 总分 = (nat_score × 0.25 + aff_score × 0.20 + div_score × 0.15 + ctx_score × 0.20 + hum_score × 0.20) × 10
- 取值范围: 0-10分
- 人工审核阈值: 总分 < 5.0 或任一维度分数 < 0.2 时需要人工审核

惩罚机制:
- 语法错误惩罚: 错误率超过5%时对nat_score扣分
- 重复惩罚: 词汇重复率超过30%时对div_score扣分  
- 长度惩罚: 回复过短(<2字符)或过长(>1000字符)时扣分
- 情感惩罚: 情感不匹配时对aff_score扣分

"""

import re
import math
import asyncio
from typing import Any, Dict, List, Tuple, Optional
from collections import Counter
import json

from .base import BaseBenchmark, BenchmarkResult
from ..types import JobContext

try:
    import language_tool_python
    LANGUAGE_TOOL_AVAILABLE = True
except ImportError:
    LANGUAGE_TOOL_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    import torch
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class ChatHumanScore(BaseBenchmark):
    """Evaluates human-like quality in casual conversations"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._lang_tool = None
        self._sentence_model = None
        self._openai_client = None
        
        # Initialize tools lazily
        self._init_tools()
    
    def _init_tools(self):
        """Initialize NLP tools"""
        if LANGUAGE_TOOL_AVAILABLE and self.config.get("enable_grammar_check", True):
            try:
                self._lang_tool = language_tool_python.LanguageTool('zh-CN')
            except Exception:
                pass
        
        if SENTENCE_TRANSFORMERS_AVAILABLE and self.config.get("enable_semantic_analysis", True):
            try:
                model_name = self.config.get("sentence_model", "all-MiniLM-L6-v2")
                self._sentence_model = SentenceTransformer(model_name)
            except Exception:
                pass
        
        if OPENAI_AVAILABLE and self.config.get("enable_gpt_judge", True):
            api_key = self.config.get("openai_api_key")
            base_url = self.config.get("openai_base_url", "https://api.openai.com/v1")
            if api_key:
                try:
                    self._openai_client = openai.OpenAI(api_key=api_key, base_url=base_url)
                except Exception:
                    pass
    
    @property
    def _get_name(self) -> str:
        return "chathumanscore"
    
    def required_artifacts(self) -> List[str]:
        return ["conversation_text", "conversation_json"]
    
    def compute(self, job_ctx: JobContext) -> BenchmarkResult:
        """
        Compute ChatHumanScore metrics
        
        Metrics computed:
        - nat_score: Naturalness (0-1)
        - aff_score: Affective Alignment (0-1)  
        - div_score: Diversity (0-1)
        - ctx_score: Context Cohesion (0-1)
        - hum_score: Human Signal Strength (0-1)
        - score: Overall human-like score (0-10)
        - needs_human: Whether manual review is needed
        """
        if not self.validate_artifacts(job_ctx):
            return BenchmarkResult(
                benchmark_name=self.name,
                metrics=self._get_default_metrics()
            )
        
        conv_json = job_ctx.artifacts["conversation_json"]
        conv_text = job_ctx.artifacts["conversation_text"]
        
        # Extract assistant response(s)
        assistant_responses = self._extract_assistant_responses(conv_json)
        if not assistant_responses:
            return BenchmarkResult(
                benchmark_name=self.name,
                metrics=self._get_default_metrics()
            )
        
        # Combine responses for analysis
        response_text = " ".join(assistant_responses)
        
        # Basic filtering and penalty calculation
        penalties = self._basic_filter(response_text)
        
        # Compute individual metrics
        nat_score = self._compute_naturalness(response_text)
        aff_score = self._compute_affective_alignment(conv_json, assistant_responses)
        div_score = self._compute_diversity(assistant_responses)
        ctx_score = self._compute_context_cohesion(conv_json, assistant_responses)
        hum_score = self._compute_human_signals(response_text)
        
        # Apply penalties
        final_scores = {
            'nat_score': max(0.0, nat_score - penalties.get('grammar_penalty', 0)),
            'aff_score': max(0.0, aff_score - penalties.get('sentiment_penalty', 0)),
            'div_score': max(0.0, div_score - penalties.get('repetition_penalty', 0)),
            'ctx_score': max(0.0, ctx_score - penalties.get('context_penalty', 0)),
            'hum_score': max(0.0, hum_score - penalties.get('format_penalty', 0))
        }
        
        # Calculate weighted overall score (0-10 scale)
        weights = self.config.get('score_weights', {
            'naturalness': 0.25,
            'affective_alignment': 0.20,
            'diversity': 0.15,
            'context_cohesion': 0.20,
            'human_signal': 0.20
        })
        
        overall_score = (
            final_scores['nat_score'] * weights['naturalness'] +
            final_scores['aff_score'] * weights['affective_alignment'] +
            final_scores['div_score'] * weights['diversity'] +
            final_scores['ctx_score'] * weights['context_cohesion'] +
            final_scores['hum_score'] * weights['human_signal']
        ) * 10
        
        # Determine if human review is needed
        needs_human = (
            overall_score < self.config.get('human_review_threshold', 5.0) or
            any(score < 0.2 for score in final_scores.values())
        )
        
        metrics = {
            **final_scores,
            'score': round(overall_score, 2),
            'needs_human': needs_human,
            'penalty_total': sum(penalties.values())
        }
        
        metadata = {
            'conversation_id': job_ctx.conv_id,
            'response_count': len(assistant_responses),
            'response_length': len(response_text),
            'penalties': penalties,
            'weights_used': weights
        }
        
        return BenchmarkResult(
            benchmark_name=self.name,
            metrics=metrics,
            metadata=metadata
        )
    
    def _get_default_metrics(self) -> Dict[str, Any]:
        """Return default metrics for failed cases"""
        return {
            'nat_score': 0.0,
            'aff_score': 0.0,
            'div_score': 0.0,
            'ctx_score': 0.0,
            'hum_score': 0.0,
            'score': 0.0,
            'needs_human': True,
            'penalty_total': 1.0
        }
    
    def _extract_assistant_responses(self, conv_json: List[Dict]) -> List[str]:
        """Extract assistant responses from conversation"""
        responses = []
        for turn in conv_json:
            if turn.get('role') == 'assistant' and turn.get('content'):
                responses.append(turn['content'].strip())
        return responses
    
    def _basic_filter(self, text: str) -> Dict[str, float]:
        """Apply basic filtering and return penalties"""
        penalties = {}
        
        # Grammar error penalty
        if self._lang_tool and LANGUAGE_TOOL_AVAILABLE:
            try:
                errors = self._lang_tool.check(text)
                error_rate = len(errors) / max(len(text.split()), 1)
                max_error_rate = self.config.get('max_grammar_error_rate', 0.05)
                if error_rate > max_error_rate:
                    penalties['grammar_penalty'] = min(0.3, error_rate - max_error_rate)
            except Exception:
                pass
        
        # Repetition penalty
        words = text.split()
        if len(words) > 5:
            word_counts = Counter(words)
            total_words = len(words)
            repeated_words = sum(count - 1 for count in word_counts.values() if count > 1)
            repeat_ratio = repeated_words / total_words
            
            max_repeat_ratio = self.config.get('max_repeat_ratio', 0.30)
            if repeat_ratio > max_repeat_ratio:
                penalties['repetition_penalty'] = min(0.4, repeat_ratio - max_repeat_ratio)
        
        # Length penalties
        if len(text.strip()) < 2:
            penalties['format_penalty'] = 0.5
        elif len(text) > 1000:  # Too verbose
            penalties['format_penalty'] = 0.1
        
        return penalties
    
    def _compute_naturalness(self, text: str) -> float:
        """Compute naturalness score (0-1)"""
        if not text.strip():
            return 0.0
        
        score = 0.5  # Base score
        
        # Check for natural language patterns
        natural_patterns = self.config.get('natural_patterns', [
            r'[。！？]',  # Chinese punctuation
            r'[啊呀吧呢]',  # Particles
            r'[嗯嗯哦哈]',  # Interjections
        ])
        
        for pattern in natural_patterns:
            if re.search(pattern, text):
                score += 0.1
        
        # Penalty for overly formal language
        formal_patterns = self.config.get('formal_patterns', [
            r'根据您的|按照您的',
            r'非常感谢您|感谢您的',
            r'我将为您|我会为您'
        ])
        
        for pattern in formal_patterns:
            if re.search(pattern, text):
                score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def _compute_affective_alignment(self, conv_json: List[Dict], responses: List[str]) -> float:
        """Compute emotional alignment score (0-1)"""
        if not responses:
            return 0.0
        
        # Simple sentiment analysis based on keywords
        positive_words = self.config.get('positive_words', ['好', '棒', '赞', '开心', '高兴'])
        negative_words = self.config.get('negative_words', ['不好', '糟糕', '难过', '生气', '失望'])
        
        # Get user sentiment from last user message
        user_sentiment = 0
        for turn in reversed(conv_json):
            if turn.get('role') == 'user' and turn.get('content'):
                user_text = turn['content']
                pos_count = sum(1 for word in positive_words if word in user_text)
                neg_count = sum(1 for word in negative_words if word in user_text)
                user_sentiment = pos_count - neg_count
                break
        
        # Check response sentiment
        response_text = " ".join(responses)
        pos_count = sum(1 for word in positive_words if word in response_text)
        neg_count = sum(1 for word in negative_words if word in response_text)
        response_sentiment = pos_count - neg_count
        
        # Calculate alignment
        if user_sentiment == 0 and response_sentiment == 0:
            return 0.7  # Neutral alignment
        elif user_sentiment * response_sentiment > 0:
            return 0.9  # Same direction
        elif user_sentiment * response_sentiment < 0:
            return 0.3  # Opposite direction
        else:
            return 0.6  # One neutral
    
    def _compute_diversity(self, responses: List[str]) -> float:
        """Compute diversity score (0-1)"""
        if not responses:
            return 0.0
        
        combined_text = " ".join(responses)
        words = combined_text.split()
        
        if len(words) < 3:
            return 0.3
        
        # Calculate distinct-n ratios
        unique_words = len(set(words))
        total_words = len(words)
        
        # Bigram diversity
        bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
        unique_bigrams = len(set(bigrams))
        total_bigrams = max(len(bigrams), 1)
        
        diversity_score = 0.6 * (unique_words / total_words) + 0.4 * (unique_bigrams / total_bigrams)
        
        return min(1.0, diversity_score)
    
    def _compute_context_cohesion(self, conv_json: List[Dict], responses: List[str]) -> float:
        """Compute context cohesion score (0-1)"""
        if not responses or not self._sentence_model:
            return 0.7  # Default neutral score
        
        try:
            # Get last user message for context
            user_context = ""
            for turn in reversed(conv_json):
                if turn.get('role') == 'user' and turn.get('content'):
                    user_context = turn['content']
                    break
            
            if not user_context:
                return 0.7
            
            # Calculate semantic similarity
            response_text = " ".join(responses)
            embeddings = self._sentence_model.encode([user_context, response_text])
            
            # Cosine similarity
            cos_sim = torch.nn.functional.cosine_similarity(
                torch.tensor(embeddings[0]).unsqueeze(0),
                torch.tensor(embeddings[1]).unsqueeze(0)
            ).item()
            
            # Map similarity to score
            return max(0.0, min(1.0, (cos_sim + 1) / 2))
            
        except Exception:
            return 0.7
    
    def _compute_human_signals(self, text: str) -> float:
        """Compute human signal strength (0-1)"""
        if not text.strip():
            return 0.0
        
        score = 0.3  # Base score
        
        # Filler words (natural hesitation markers)
        filler_words = self.config.get('filler_words', ['呃', '嗯', '那个', '就是', '其实'])
        filler_count = sum(1 for word in filler_words if word in text)
        score += min(0.3, filler_count * 0.1)
        
        # Emoji usage
        emoji_pattern = r'[😀-🙏🌀-🗿🚀-🛿⚠-⚡]'
        emoji_count = len(re.findall(emoji_pattern, text))
        score += min(0.2, emoji_count * 0.1)
        
        # Casual expressions
        casual_patterns = self.config.get('casual_patterns', [
            r'哈哈|嘿嘿|呵呵',
            r'吧[，。！？]',
            r'呀[，。！？]',
        ])
        
        for pattern in casual_patterns:
            if re.search(pattern, text):
                score += 0.1
        
        # Personal references
        personal_patterns = ['我觉得', '我想', '我感觉', '个人认为']
        personal_count = sum(1 for pattern in personal_patterns if pattern in text)
        score += min(0.2, personal_count * 0.1)
        
        return min(1.0, score)
    
    def get_description(self) -> str:
        return "Evaluates human-like quality in conversational responses across 5 dimensions"
    
    def get_config_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "enable_grammar_check": {
                    "type": "boolean",
                    "description": "Enable grammar checking",
                    "default": True
                },
                "enable_semantic_analysis": {
                    "type": "boolean", 
                    "description": "Enable semantic similarity analysis",
                    "default": True
                },
                "enable_gpt_judge": {
                    "type": "boolean",
                    "description": "Enable GPT-based judging",
                    "default": False
                },
                "openai_api_key": {
                    "type": "string",
                    "description": "OpenAI API key for GPT judge"
                },
                "openai_base_url": {
                    "type": "string",
                    "description": "OpenAI base URL",
                    "default": "https://api.openai.com/v1"
                },
                "sentence_model": {
                    "type": "string",
                    "description": "Sentence transformer model name",
                    "default": "all-MiniLM-L6-v2"
                },
                "max_grammar_error_rate": {
                    "type": "number",
                    "description": "Maximum acceptable grammar error rate",
                    "default": 0.05,
                    "minimum": 0
                },
                "max_repeat_ratio": {
                    "type": "number",
                    "description": "Maximum acceptable word repetition ratio",
                    "default": 0.30,
                    "minimum": 0
                },
                "human_review_threshold": {
                    "type": "number",
                    "description": "Score threshold below which human review is needed",
                    "default": 5.0,
                    "minimum": 0,
                    "maximum": 10
                },
                "score_weights": {
                    "type": "object",
                    "description": "Weights for each dimension",
                    "properties": {
                        "naturalness": {"type": "number", "default": 0.25},
                        "affective_alignment": {"type": "number", "default": 0.20},
                        "diversity": {"type": "number", "default": 0.15},
                        "context_cohesion": {"type": "number", "default": 0.20},
                        "human_signal": {"type": "number", "default": 0.20}
                    }
                },
                "filler_words": {
                    "type": "array",
                    "description": "List of filler words that indicate natural speech",
                    "items": {"type": "string"},
                    "default": ["呃", "嗯", "那个", "就是", "其实"]
                },
                "positive_words": {
                    "type": "array",
                    "description": "List of positive sentiment words",
                    "items": {"type": "string"},
                    "default": ["好", "棒", "赞", "开心", "高兴"]
                },
                "negative_words": {
                    "type": "array",
                    "description": "List of negative sentiment words", 
                    "items": {"type": "string"},
                    "default": ["不好", "糟糕", "难过", "生气", "失望"]
                }
            },
            "additionalProperties": False
        } 