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

### 使用pydantic_settings 替代load_dotenv
- 编写setting.py 集中管理env环境变量,
- 类型丢失：os.getenv 永远返回 str 或 None。你需要手动转 int、bool
- 拼写错误：如果写错了字符串 key (os.getenv("DB_HST"))，程序可能会静默失败。

### 项目结构
- uv init 扁平结构 用在一个独立运行的项目，不需要被发布到 PyPI，也不会被其他项目作为依赖安装。
  - 典型场景：Web 服务（Django/FastAPI）、数据分析脚本、爬虫、自动化工具。
- uv init project-name --package 定位：中间组件。这是一段可复用的代码，设计目的是为了被打包（Build）成 .whl 文件，然后发布让别人安装。
  - 典型场景：开发一个 SDK、一个公共函数库、一个 FastAPI 的插件。
