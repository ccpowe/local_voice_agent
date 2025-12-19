"""
langchain 多模态图片输入
"""

import base64
from pathlib import Path

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from local_voice_agent.settings import Settings


def _build_chat_model(settings: Settings) -> ChatOpenAI:
    if settings.base_url:
        return ChatOpenAI(
            model_name=settings.model_name,
            openai_api_key=settings.api_key,
            openai_api_base=settings.base_url,
        )
    return ChatOpenAI(model_name=settings.model_name, openai_api_key=settings.api_key)


model = _build_chat_model(Settings())


# 读取本地图片文件
def image_to_base64(image_path: str | Path) -> str:
    """将图片文件转换为 base64 字符串"""
    return base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")


# 使用图片文件
# repo_root = Path(__file__).resolve().parents[1]
image_path = "data/images/截图 2025-12-17 10-28-34.png"
image_base64 = image_to_base64(image_path)

message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "这张图片中有什么？"},
        {
            "type": "image",
            "base64": image_base64,
            "mime_type": "image/png",
        },
    ],
}

agent = create_agent(model, tools=[], system_prompt="根据图片内容回答问题")
result = agent.invoke({"messages": [message]})
print(result["messages"][-1].content)
