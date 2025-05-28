import json
import openai
from openai import OpenAI  # 导入 OpenAI 类

from tqdm import tqdm
from typing import List, Dict, cast # 导入 cast
from openai.types.chat import ChatCompletionMessageParam # 导入消息参数类型

from weclone.utils.config import load_config

config = load_config("web_demo")

config = {
    "default_prompt": config["default_system"],
    "model": "gpt-3.5-turbo",
    "history_len": 15,
}

config = type("Config", (object,), config)()

# 初始化 OpenAI 客户端
client = OpenAI(
    api_key="""sk-test""",
    base_url="http://127.0.0.1:8005/v1"
)


def handler_text(content: str, history: list, config):
    messages = [{"role": "system", "content": f"{config.default_prompt}"}]
    for item in history:
        messages.append(item)
    messages.append({"role": "user", "content": content})
    history.append({"role": "user", "content": content})
    try:
        # 使用新的 API 调用方式
        # 将 messages 转换为正确的类型
        typed_messages = cast(List[ChatCompletionMessageParam], messages)
        response = client.chat.completions.create(
            model=config.model,
            messages=typed_messages, # 传递转换后的列表
            max_tokens=50
        )
    except openai.APIError as e:
        history.pop()
        return "AI接口出错,请重试\n" + str(e)

    resp = str(response.choices[0].message.content) # type: ignore
    resp = resp.replace("\n ", "")
    history.append({"role": "assistant", "content": resp})
    return resp


def main():
    test_list = json.loads(open("dataset/test_data.json", "r", encoding="utf-8").read())["questions"]
    res = []
    
    for questions in tqdm(test_list, desc=" Testing..."):
        history = []
        
        # Add all user questions to history first (for recording purposes)
        for q in questions:
            history.append({"role": "user", "content": q})
        
        # Get one response for all the questions
        if questions:  # Make sure there are questions
            # Merge all user questions into one message for API call
            merged_question = "\n".join(questions)
            
            # Create messages for API call (strict u/a alternating format)
            messages = [
                {"role": "system", "content": f"{config.default_prompt}"},
                {"role": "user", "content": merged_question}
            ]
            
            try:
                # Call API with merged question
                typed_messages = cast(List[ChatCompletionMessageParam], messages)
                response = client.chat.completions.create(
                    model=config.model,
                    messages=typed_messages,
                    max_tokens=50
                )
                
                resp = str(response.choices[0].message.content)
                resp = resp.replace("\n ", "")
                history.append({"role": "assistant", "content": resp})
                
            except openai.APIError as e:
                history.append({"role": "assistant", "content": f"AI接口出错,请重试\n{str(e)}"})
        
        res.append(history)

    with open("test_result-my.txt", "w", encoding="utf-8") as res_file:
        for r in res:
            for i in r:
                res_file.write(f"{i['role']}: {i['content']}\n")
            res_file.write("\n")


if __name__ == "__main__":
    main()
