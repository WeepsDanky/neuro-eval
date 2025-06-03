"""
Prompt templates for ChatHumanScore evaluation

Contains all prompt templates used by the ChatHumanScore benchmark for GPT-Judge evaluation.
"""

# GPT-Judge evaluation prompt template
CHATHUMANSCORE_JUDGE_PROMPT = """你是一个专业的对话质量评估专家。请评估AI助手回复的"人性化"程度，从以下四个维度进行打分：

对话上下文：
{conversation_context}

AI助手回复：
{assistant_response}

请从以下四个维度评估这个回复（每个维度0-1分，保留两位小数）：

1. **nat_score (自然度)**: 语言的自然性和口语化程度
   - 考虑因素：语气词使用（啊、呀、吧、呢、嗯、哦、哈等）、emoji表情、口头语、网络梗、标点符号的自然使用
   - 避免过于正式或官方的表达

2. **bio_score (情感对齐)**: 与用户情感的匹配程度  
   - 考虑因素：识别用户情绪，回复是否情感恰当、共情能力、情绪感染力
   - 能否准确捕捉并回应用户的情感状态

3. **psycho_score (人格一致性)**: 行为和语言是否符合设定的性格特征
   - 考虑因素：语言风格一致性、行为模式、价值观表达、个性化特征
   - 是否展现出连贯的"人格"特征

4. **social_score (社交适应性)**: 回复是否符合社交语境和场合
   - 考虑因素：话题相关性、社交礼仪、场合适宜性、互动自然度
   - 是否懂得"察言观色"和社交边界

请严格按照以下JSON格式输出评估结果：

```json
{{
    "nat_score": 0.XX,
    "nat_reasoning": "自然度评分理由（50字以内）",
    "bio_score": 0.XX,
    "bio_reasoning": "情感对齐评分理由（50字以内）",
    "psycho_score": 0.XX,
    "psycho_reasoning": "人格一致性评分理由（50字以内）",
    "social_score": 0.XX,
    "social_reasoning": "社交适应性评分理由（50字以内）",
    "overall_reasoning": "整体评价和建议（100字以内）"
}}
```

注意：
- 分数请严格控制在0.00-1.00范围内
- 理由要简洁明确，突出关键评判要素
- 避免过度宽松或严苛的评分标准"""

# Alternative prompt for different evaluation styles (can be extended)
CHATHUMANSCORE_JUDGE_PROMPT_STRICT = """你是一个严格的对话质量评估专家。请以高标准评估AI助手回复的"人性化"程度：

对话上下文：
{conversation_context}

AI助手回复：
{assistant_response}

评估标准（每个维度0-1分，采用严格评分标准）：

1. **nat_score (自然度)**: 必须达到真人聊天水平
2. **bio_score (情感对齐)**: 必须精准识别并恰当回应情感
3. **psycho_score (人格一致性)**: 必须保持明显且一致的个性特征
4. **social_score (社交适应性)**: 必须完美适配社交场景

输出格式同标准模板。"""

CHATHUMANSCORE_JUDGE_PROMPT_LENIENT = """你是一个宽松的对话质量评估专家。请以包容的态度评估AI助手回复：

对话上下文：
{conversation_context}

AI助手回复：
{assistant_response}

评估重点（每个维度0-1分，重视潜力和改进空间）：

1. **nat_score (自然度)**: 关注语言的亲切感和易懂性
2. **bio_score (情感对齐)**: 关注基本的情感理解能力
3. **psycho_score (人格一致性)**: 关注基础的风格连贯性
4. **social_score (社交适应性)**: 关注基本的社交礼貌

输出格式同标准模板。"""


def get_judge_prompt(style: str = "default") -> str:
    """
    Get judge prompt template by style
    
    Args:
        style: Prompt style ("default", "strict", "lenient")
        
    Returns:
        Prompt template string
    """
    prompt_map = {
        "default": CHATHUMANSCORE_JUDGE_PROMPT,
        "strict": CHATHUMANSCORE_JUDGE_PROMPT_STRICT,
        "lenient": CHATHUMANSCORE_JUDGE_PROMPT_LENIENT
    }
    
    return prompt_map.get(style, CHATHUMANSCORE_JUDGE_PROMPT)


def format_judge_prompt(conversation_context: str, assistant_response: str, style: str = "default") -> str:
    """
    Format judge prompt with conversation data
    
    Args:
        conversation_context: Formatted conversation context
        assistant_response: Assistant response to evaluate
        style: Prompt style ("default", "strict", "lenient")
        
    Returns:
        Formatted prompt ready for GPT judge
    """
    template = get_judge_prompt(style)
    return template.format(
        conversation_context=conversation_context,
        assistant_response=assistant_response
    ) 