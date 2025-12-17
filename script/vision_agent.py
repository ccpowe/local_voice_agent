"""
langchain 多模态图片输入
"""

import base64
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

load_dotenv()
model = ChatOpenAI(
    model=os.getenv("MODEL_NAME"),  # type:ignore
    api_key=os.getenv("API_KEY"),  # type:ignore
    base_url=os.getenv("BASE_URL"),  # type:ignore
)


# 读取本地图片文件
def image_to_base64(image_path: str) -> str:
    """将图片文件转换为 base64 字符串"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


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
