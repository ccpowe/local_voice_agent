# python312.rule
# 强制执行现代 Python 3.12+ 最佳实践的规则。
# 适用于项目中的所有 Python 文件 (*.py)。
#
# 涵盖的准则：
# - 语法：Match-case, 海象运算符 (:=), 泛型集合 (list/dict), 联合类型 (|)。
# - 结构：拒绝嵌套 (Never Nester), 结构化并发 (Asyncio TaskGroup)。
# - 类型：强静态类型, 禁止行内忽略, Pydantic v2 原生验证。
# - 配置：使用 Pydantic Settings 进行强类型配置管理。
# - 工具：Pathlib, uv 包管理。

description: "现代 Python 3.12+ 编码最佳实践、异步并发规范及风格指南。"
files: "**/*.py"

guidelines:
  - title: "Match-Case 语法"
    description: >
      当适用模式匹配时，优先使用 match-case 结构，而不是传统的 if/elif/else 链。
      这能使代码更清晰、更简洁且更易于维护。

  - title: "海象运算符 (Walrus Operator)"
    description: >
      在可以将赋值和条件测试结合的地方，利用海象运算符 (:=) 来精简代码。
      应当审慎地使用它，仅在能提高可读性并减少冗余时使用。

  - title: "拒绝嵌套 (Never Nester)"
    description: >
      旨在通过避免深层嵌套来保持代码扁平化。使用提前返回 (early returns) 和卫语句 (guard clauses)。
      
      Example:
      ```python
      # BAD
      def process_data(data):
          if data:
              if data.is_valid:
                  save(data)
      
      # GOOD
      def process_data(data):
          if not data or not data.is_valid:
              return
          save(data)
      ```

  - title: "现代类型提示"
    description: >
      采用现代类型提示，使用内置泛型（如 list 和 dict）以及用于联合类型的管道符 (|)（例如 int | None）。
      避免使用 typing 模块中旧的、已弃用的构造，如 Optional, Union, Dict, List。

  - title: "现代异步 (结构化并发)"
    description: >
      在 Python 3.11+ 中编写异步代码时，必须遵循结构化并发 (Structured Concurrency) 原则：
      - 优先使用 `asyncio.TaskGroup` 管理并发任务，而不是 `asyncio.gather`。TaskGroup 能确保异常正确传播并取消同组任务。
      - 使用 `asyncio.run()` 作为程序入口。
      - 避免使用 `add_done_callback`，应使用 `await` 或上下文管理器。
      
      Example:
      ```python
      # GOOD
      async def main():
          async with asyncio.TaskGroup() as tg:
              tg.create_task(task1())
              tg.create_task(task2())
      # 当 task1 失败时，task2 会被自动取消，且异常会被正确抛出。
      ```

  - title: "Pydantic 优先解析"
    description: >
      优先使用 Pydantic v2 的原生验证 (`model_validate`, `field_validator`, `computed_field`)。
      避免手动编写转换逻辑或自定义 `from_sdk` 方法。让 Pydantic 处理所有的数据清洗工作。

  - title: "类型安全配置 (Type-Safe Settings)"
    description: >
      使用 `pydantic-settings` 管理环境变量，严禁使用 `os.getenv` 或 `python-dotenv`。
      通过定义继承自 `BaseSettings` 的类来自动处理加载、类型转换和默认值。
      
      Example:
      ```python
      # BAD
      import os
      db_port = int(os.getenv("DB_PORT", 5432))
      
      # GOOD
      from pydantic_settings import BaseSettings, SettingsConfigDict
      
      class Settings(BaseSettings):
          db_port: int = 5432
          api_key: str
          model_config = SettingsConfigDict(env_file=".env")
          
      settings = Settings()
      ```

  - title: "Pydantic 区分联合 (Discriminated Unions)"
    description: >
      在处理多态模型时，严禁通过继承覆写字段的方式来区分类型（这违反 LSP 原则）。
      必须使用 `Annotated[Union[A, B], Field(discriminator='type')]` 的组合方式。

      Example:
      ```python
      # BAD (Do NOT do this)
      class BaseEvent(BaseModel):
          type: Literal["base"]
      class MessageEvent(BaseEvent):
          type: Literal["message"]  # 错误：子类缩小了父类字段类型，导致类型检查失败
      
      # GOOD (Do this)
      class EventMixin(BaseModel):
          timestamp: float
      
      class MessageEvent(EventMixin):
          type: Literal["message"] = "message"
          content: str
      
      class ErrorEvent(EventMixin):
          type: Literal["error"] = "error"
          code: int
      
      # 使用联合类型 + 区分符
      Event = Annotated[MessageEvent | ErrorEvent, Field(discriminator="type")]
      ```

  - title: "使用 Pathlib 进行文件操作"
    description: >
      完全禁止使用 `os.path.join`, `os.getcwd` 等旧式 API。
      必须使用 `pathlib.Path` 进行所有文件路径操作（如 `Path.cwd() / "data"`, `path.read_text()`）。

  - title: "声明式与极简主义代码"
    description: >
      编写声明式代码。如果逻辑很长，请将其提取为具有描述性名称的私有辅助函数。
      代码应尽量扁平，逻辑流应像文章一样从上到下阅读。

  - title: "异常文档"
    description: >
      仅记录显式引发 (raise) 的异常。不要记录显而易见的内置异常（如 TypeError）。
      这保持了文档的信噪比。

  - title: "现代枚举 (Enum) 用法"
    description: >
      使用 `StrEnum` (Python 3.11+) 进行字符串枚举。
      使用 `auto()` 自动赋值。
      成员名称必须大写。

  - title: "禁止行内忽略 (No Inline Ignores)"
    description: >
      禁止使用 `# type: ignore` 或 `# noqa`。
      如果遇到类型错误，请修复代码结构（使用 `isinstance` 防护、`cast` 或重构类型定义）。

  - title: "所有命令使用 uv"
    description: >
      严禁使用 `pip` 或直接调用 `python`。
      - 安装依赖: `uv add <package>` (例如 `uv add pydantic-settings`)
      - 运行脚本: `uv run script.py`
      - 运行工具: `uv run pytest`
