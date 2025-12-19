这个文档记录我在开发过程中的心得体会

### 声明ChatopenAI 通过ty类型检测
```python
return ChatOpenAI(
    model_name=model_name,
    openai_api_key=SecretStr(api_key),#use openai_api_base openai_api_key 和api_key..一样只是内部pydantic的别名
    openai_api_base=base_url,
)
```
### 使用vibe的python开发 Agent.md
- 让gemini3pro 给优化了，提升one shot 和异步指南

## 使用pydantic_settings 替代load_dotenv
- 编写setting.py 集中管理env环境变量,
- 类型丢失：os.getenv 永远返回 str 或 None。你需要手动转 int、bool
- 拼写错误：如果写错了字符串 key (os.getenv("DB_HST"))，程序可能会静默失败。
