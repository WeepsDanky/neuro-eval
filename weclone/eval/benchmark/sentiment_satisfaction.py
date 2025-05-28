"""
Sentiment Satisfaction Benchmark

Measures user satisfaction through sentiment analysis of the conversation.
Evaluates the emotional tone and satisfaction level based on conversation content.
"""

from typing import Any, Dict, List
from .base import BaseBenchmark, BenchmarkResult
from ..types import JobContext


class SentimentSatisfactionBenchmark(BaseBenchmark):
    """Measures user satisfaction through sentiment analysis"""
    
    @property
    def _get_name(self) -> str:
        return "sentiment_satisfaction"
    
    def required_artifacts(self) -> List[str]:
        return ["conversation_text"]
    
    def compute(self, job_ctx: JobContext) -> BenchmarkResult:
        """
        Compute sentiment satisfaction metrics
        
        Metrics computed:
        - post_chat_rating: Overall satisfaction rating (1-5 scale)
        - sentiment_score: Normalized sentiment score (-1 to 1)
        """
        if not self.validate_artifacts(job_ctx):
            return BenchmarkResult(
                benchmark_name=self.name,
                metrics={"post_chat_rating": 3.0, "sentiment_score": 0.0}
            )
        
        conversation_text = job_ctx.artifacts.get("conversation_text", "")
        
        # Configuration for sentiment words
        positive_words = self.config.get("positive_words", ["好", "棒", "满意", "喜欢", "不错", "很好", "开心", "高兴", "优秀", "完美"])
        negative_words = self.config.get("negative_words", ["差", "糟糕", "不满", "讨厌", "不好", "失望", "糟", "烂", "垃圾", "恶心"])
        
        # Simple sentiment analysis based on keyword counting
        pos_count = sum(word in conversation_text for word in positive_words)
        neg_count = sum(word in conversation_text for word in negative_words)
        
        # Calculate normalized sentiment score
        total_sentiment_words = pos_count + neg_count
        if total_sentiment_words == 0:
            sentiment_score = 0.0
        else:
            sentiment_score = (pos_count - neg_count) / total_sentiment_words
        
        # Calculate satisfaction rating on 1-5 scale
        base_rating = self.config.get("base_rating", 3.0)
        rating_sensitivity = self.config.get("rating_sensitivity", 0.5)
        
        if pos_count > neg_count:
            rating = min(5.0, base_rating + (pos_count - neg_count) * rating_sensitivity)
        else:
            rating = max(1.0, base_rating - (neg_count - pos_count) * rating_sensitivity)
        
        metrics = {
            "post_chat_rating": round(rating, 2),
            "sentiment_score": round(sentiment_score, 3)
        }
        
        metadata = {
            "positive_word_count": pos_count,
            "negative_word_count": neg_count,
            "total_sentiment_words": total_sentiment_words,
            "conversation_length": len(conversation_text),
            "base_rating": base_rating,
            "rating_sensitivity": rating_sensitivity
        }
        
        return BenchmarkResult(
            benchmark_name=self.name,
            metrics=metrics,
            metadata=metadata
        )
    
    def get_description(self) -> str:
        return "Analyzes conversation sentiment to measure user satisfaction levels"
    
    def get_config_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "positive_words": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of positive sentiment words to detect",
                    "default": ["好", "棒", "满意", "喜欢", "不错", "很好"]
                },
                "negative_words": {
                    "type": "array", 
                    "items": {"type": "string"},
                    "description": "List of negative sentiment words to detect",
                    "default": ["差", "糟糕", "不满", "讨厌", "不好", "失望"]
                },
                "base_rating": {
                    "type": "number",
                    "description": "Baseline satisfaction rating",
                    "default": 3.0,
                    "minimum": 1.0,
                    "maximum": 5.0
                },
                "rating_sensitivity": {
                    "type": "number",
                    "description": "How much each sentiment word affects the rating",
                    "default": 0.5,
                    "minimum": 0.1,
                    "maximum": 2.0
                }
            },
            "additionalProperties": False
        } 