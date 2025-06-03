"""
ChatHumanScore Benchmark

Measures the "human-like" quality of conversational AI responses in casual chat scenarios.
Uses GPT-Judge for intelligent evaluation across four key dimensions.

评估标准详细说明 (Evaluation Criteria):

**评估方法: GPT-Judge**
使用大语言模型作为评委，对AI回复进行智能化、多维度评估。

**四个核心评估维度:**

1. **nat_score (自然度分数, 0-1)**: 评估回复语言的自然性和口语化程度
   - 评估要素:
     * 语气词使用: 啊、呀、吧、呢、嗯、哦、哈等
     * 表情符号和emoji的恰当使用
     * 口头语、网络梗、流行语的自然融入
     * 标点符号的口语化使用（如连续感叹号、省略号等）
     * 避免过于正式或官方的表达方式
   - GPT-Judge会根据语言的流畅度、亲切感、口语化程度综合评分

2. **bio_score (情感对齐分数, 0-1)**: 评估回复与用户情感的匹配程度
   - 评估要素:
     * 情绪识别: 准确捕捉用户的情感状态
     * 情感回应: 回复情感与用户情感的协调性
     * 共情能力: 是否能理解和回应用户的感受
     * 情绪感染力: 能否通过回复影响用户情绪
   - GPT-Judge会分析对话上下文，判断情感匹配的恰当性

3. **psycho_score (人格一致性分数, 0-1)**: 评估回复是否符合一致的性格特征
   - 评估要素:
     * 语言风格一致性: 措辞、表达习惯的连贯性
     * 行为模式: 反应方式、处事态度的一致性
     * 价值观表达: 观点、立场的稳定性
     * 个性化特征: 独特的语言标识和个人特色
   - GPT-Judge会检查AI是否展现出连贯、可信的"人格"

4. **social_score (社交适应性分数, 0-1)**: 评估回复在社交语境中的恰当性
   - 评估要素:
     * 话题相关性: 回复与对话主题的关联度
     * 社交礼仪: 礼貌用语、社交规范的遵循
     * 场合适宜性: 语言风格与对话场景的匹配
     * 互动自然度: 对话的流畅性和参与感
     * 社交边界: 对隐私、敏感话题的适当处理

**评分机制:**
- GPT-Judge输出JSON格式结果，包含各维度分数(0-1)和详细reasoning
- 综合评分: (nat_score × 0.25 + bio_score × 0.25 + psycho_score × 0.25 + social_score × 0.25) × 10
- 取值范围: 0-10分

**技术特性:**
- 语义理解: 能够理解复杂的语境、隐含意思、讽刺等高级语言现象
- 灵活适应: 自动适应新梗、新表达、新的社交语境，无需手动更新词典
- 推理能力: 能够进行多轮对话的逻辑推理和性格一致性判断
- 可解释性: 提供详细的评分理由和改进建议，便于模型优化

**配置要求:**
- 必须配置有效的OpenAI API密钥 (openai_api_key)
- 建议使用GPT-4或GPT-4-turbo模型以获得最佳评估效果
- 支持自定义评分权重和评估prompt风格

"""

import json
from typing import Any, Dict, List, Optional

from .base import BaseBenchmark, BenchmarkResult
from .prompt import format_judge_prompt # 导入格式化prompt
from ..types import JobContext
import openai


class ChatHumanScore(BaseBenchmark):
    """Evaluates human-like quality in casual conversations using GPT-Judge"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._openai_client = None
        
        # Initialize OpenAI client
        self._init_openai_client()
    
    def _init_openai_client(self):
        """Initialize OpenAI client"""
        if self.config.get("enable_gpt_judge", True):
            api_key = self.config.get("openai_api_key")
            base_url = self.config.get("openai_base_url", "https://api.openai.com/v1")
            
            if not api_key:
                print("Warning: No openai_api_key configured for ChatHumanScore benchmark")
                return
                
            try:
                self._openai_client = openai.OpenAI(
                    api_key=api_key, 
                    base_url=base_url
                )
                print(f"Initialized OpenAI client with base_url: {base_url}")
            except Exception as e:
                print(f"Failed to initialize OpenAI client: {e}")
                self._openai_client = None
    
    @property
    def _get_name(self) -> str:
        return "chathumanscore"
    
    def required_artifacts(self) -> List[str]:
        return ["conversation_text", "conversation_json"]
    
    def compute(self, job_ctx: JobContext) -> BenchmarkResult:
        """
        Compute ChatHumanScore metrics using GPT-Judge
        
        Metrics computed:
        - nat_score: Naturalness (0-1)
        - bio_score: Emotional Alignment (0-1)  
        - psycho_score: Personality Consistency (0-1)
        - social_score: Social Context Appropriateness (0-1)
        - score: Overall human-like score (0-10)
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
        
        # Evaluate using GPT-Judge
        if not self._openai_client:
            raise RuntimeError(
                "OpenAI client not available. Please configure 'openai_api_key' and 'openai_base_url' "
                "in the benchmarks.chathumanscore section of your config file. "
                f"Current config: api_key={'configured' if self.config.get('openai_api_key') else 'missing'}, "
                f"base_url={self.config.get('openai_base_url', 'default')}"
            )
        
        try:
            gpt_metrics = self._gpt_judge_evaluate(conv_json, assistant_responses)
            if gpt_metrics:
                # 将 raw_judge_output 添加到 metadata 中
                metadata = gpt_metrics['metadata'].copy()
                if 'raw_judge_output' in gpt_metrics:
                    metadata['raw_judge_output'] = gpt_metrics['raw_judge_output']
                
                return BenchmarkResult(
                    benchmark_name=self.name,
                    metrics=gpt_metrics['metrics'],
                    metadata=metadata
                )
            else:
                raise RuntimeError("GPT-Judge evaluation returned empty result")
                
        except Exception as e:
            print(f"GPT-Judge evaluation failed: {e}")
            return BenchmarkResult(
                benchmark_name=self.name,
                metrics=self._get_default_metrics()
            )
    
    def _gpt_judge_evaluate(self, conv_json: List[Dict], assistant_responses: List[str]) -> Optional[Dict[str, Any]]:
        """Evaluate using GPT-Judge"""
        if not self._openai_client:
            return None
        
        # Prepare conversation context
        conversation_context = self._format_conversation_for_judge(conv_json)
        assistant_text = " ".join(assistant_responses)
        
        # Create evaluation prompt
        prompt = self._create_judge_prompt(conversation_context, assistant_text)
        
        try:
            response = self._openai_client.chat.completions.create(
                model=self.config.get("judge_model", "gpt-4o-mini"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=1000
            )
            
            result_text = response.choices[0].message.content
            parsed_result = self._parse_judge_response(result_text, conv_json, assistant_responses)
            
            # Add raw GPT judge output to the result for debugging
            if parsed_result:
                parsed_result['raw_judge_output'] = {
                    'prompt': prompt,
                    'response': result_text,
                    'model': self.config.get("judge_model", "gpt-4o-mini"),
                    'conversation_context': conversation_context,
                    'assistant_text': assistant_text
                }
            
            return parsed_result
            
        except Exception as e:
            print(f"GPT-Judge API call failed: {e}")
            return None
    
    def _create_judge_prompt(self, conversation_context: str, assistant_response: str) -> str:
        """Create evaluation prompt for GPT-Judge"""
        # Get prompt style from config (default, strict, lenient)
        prompt_style = self.config.get("prompt_style", "default")
        return format_judge_prompt(conversation_context, assistant_response, prompt_style)

    def _format_conversation_for_judge(self, conv_json: List[Dict]) -> str:
        """Format conversation for GPT judge"""
        formatted_lines = []
        for turn in conv_json:
            role = turn.get('role', 'unknown')
            content = turn.get('content', '').strip()
            if content:
                if role == 'user':
                    formatted_lines.append(f"用户: {content}")
                elif role == 'assistant':
                    formatted_lines.append(f"AI助手: {content}")
        
        return "\n".join(formatted_lines)
    
    def _parse_judge_response(self, response_text: str, conv_json: List[Dict], assistant_responses: List[str]) -> Dict[str, Any]:
        """Parse GPT-Judge response and extract metrics"""
        try:
            # Extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response_text[json_start:json_end]
            judge_result = json.loads(json_str)
            
            # Extract scores
            nat_score = float(judge_result.get('nat_score', 0.0))
            bio_score = float(judge_result.get('bio_score', 0.0))
            psycho_score = float(judge_result.get('psycho_score', 0.0))
            social_score = float(judge_result.get('social_score', 0.0))
            
            # Calculate weighted overall score (0-10 scale)
            weights = self.config.get('score_weights', {
                'naturalness': 0.25,
                'bio_alignment': 0.25,
                'personality': 0.25,
                'social_context': 0.25
            })
            
            overall_score = (
                nat_score * weights['naturalness'] +
                bio_score * weights['bio_alignment'] +
                psycho_score * weights['personality'] +
                social_score * weights['social_context']
            ) * 10
            
            metrics = {
                'nat_score': round(nat_score, 2),
                'bio_score': round(bio_score, 2),
                'psycho_score': round(psycho_score, 2),
                'social_score': round(social_score, 2),
                'score': round(overall_score, 2),
                'penalty_total': 0.0  # No penalties in GPT-Judge mode
            }
            
            metadata = {
                'evaluation_method': 'gpt_judge',
                'judge_model': self.config.get("judge_model", "gpt-4o-mini"),
                'response_count': len(assistant_responses),
                'response_length': len(" ".join(assistant_responses)),
                'reasoning': {
                    'nat_reasoning': judge_result.get('nat_reasoning', ''),
                    'bio_reasoning': judge_result.get('bio_reasoning', ''),
                    'psycho_reasoning': judge_result.get('psycho_reasoning', ''),
                    'social_reasoning': judge_result.get('social_reasoning', ''),
                    'overall_reasoning': judge_result.get('overall_reasoning', '')
                },
                'weights_used': weights,
                'raw_response_text': response_text  # 保存原始响应文本用于调试
            }
            
            return {'metrics': metrics, 'metadata': metadata}
            
        except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
            print(f"Failed to parse GPT-Judge response: {e}")
            print(f"Response text: {response_text}")
            return None
    
    def _get_default_metrics(self) -> Dict[str, Any]:
        """Return default metrics for failed cases"""
        return {
            'nat_score': 0.0,
            'bio_score': 0.0,
            'psycho_score': 0.0,
            'social_score': 0.0,
            'score': 0.0,
            'penalty_total': 1.0
        }
    
    def _extract_assistant_responses(self, conv_json: List[Dict]) -> List[str]:
        """Extract assistant responses from conversation"""
        responses = []
        for turn in conv_json:
            if turn.get('role') == 'assistant' and turn.get('content'):
                responses.append(turn['content'].strip())
        return responses
    
    def get_description(self) -> str:
        return "ChatHumanScore Benchmark"
    
    def get_config_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "enable_gpt_judge": {
                    "type": "boolean",
                    "description": "Enable GPT-based judging (primary method)",
                    "default": True
                },
                "judge_model": {
                    "type": "string",
                    "description": "GPT model to use for judging",
                    "default": "gpt-4o-mini"
                },
                "prompt_style": {
                    "type": "string", 
                    "description": "Evaluation prompt style",
                    "enum": ["default", "strict", "lenient"],
                    "default": "default"
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
                "score_weights": {
                    "type": "object",
                    "description": "Weights for each dimension",
                    "properties": {
                        "naturalness": {"type": "number", "default": 0.25},
                        "bio_alignment": {"type": "number", "default": 0.25},
                        "personality": {"type": "number", "default": 0.25},
                        "social_context": {"type": "number", "default": 0.25}
                    }
                }
            },
            "additionalProperties": False
        } 