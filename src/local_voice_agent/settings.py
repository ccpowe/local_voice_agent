from __future__ import annotations

from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    项目配置（环境变量）集中管理。

    为什么要这么做：
    - 让“读环境变量 + 类型校验 + 默认值/可选项”集中到一个地方，业务代码只消费 `Settings`。
    - `ty` 对 pydantic 的字段别名（alias）支持有限：用 `Settings` 可以避免在业务代码里到处写 `os.getenv(...)`。

    环境变量约定（推荐用带前缀的命名空间，避免和其他项目/系统变量冲突）：
    - 必填：`LVA_MODEL_NAME`、`LVA_API_KEY`
    - 可选：`LVA_BASE_URL`

    兼容旧变量名（便于迁移）：
    - `MODEL_NAME` / `API_KEY` / `BASE_URL`
    """

    # `validation_alias=AliasChoices(...)` 表示：按顺序尝试从这些 env key 取值。
    # 例如这里会优先读取 `LVA_MODEL_NAME`，如果没有就回退到 `MODEL_NAME`。
    model_name: str = Field(
        default="",
        validation_alias=AliasChoices("LVA_MODEL_NAME", "MODEL_NAME")
    )

    # 用 `SecretStr` 存 API Key：
    # - 打日志/打印对象时会自动打码，避免泄漏密钥
    # - `ChatOpenAI` 的类型也更倾向于接收 `SecretStr`
    api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices("LVA_API_KEY", "API_KEY"),
    )

    # Base URL 是可选的：用于自建/代理 OpenAI 兼容服务（比如本地网关、LM Studio、vLLM 等）
    base_url: str | None = Field(
        default=None, validation_alias=AliasChoices("LVA_BASE_URL", "BASE_URL")
    )

    # `env_file=".env"`：让 `Settings()` 自动读取项目根目录的 `.env` 文件（无需再调用 load_dotenv）。
    # `extra="ignore"`：如果 `.env` 里有其他无关变量，不会报错。
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 因为我们给必填字段设置了空默认值（为了让静态类型检查器不报 “缺少构造参数”），
    # 所以必须在运行时做一次“真必填”校验：如果 env 没提供值，就在启动时快速失败并给出清晰错误。
    @model_validator(mode="after")
    def _ensure_required(self) -> "Settings":
        if not self.model_name:
            raise ValueError(
                "Missing required environment variable: LVA_MODEL_NAME (or MODEL_NAME)"
            )
        if not self.api_key.get_secret_value():
            raise ValueError(
                "Missing required environment variable: LVA_API_KEY (or API_KEY)"
            )
        return self
