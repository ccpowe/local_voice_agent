"""
如果参数也在路径中声明，它将被用作路径参数。
如果参数是单一类型（如 int 、 float 、 str 、 bool 等），它将被解释为查询参数。
如果参数被声明为 Pydantic 模型的类型，它将被解释为请求体。
使用 Annotated 和 Query,Path,Body进行参数检验
Query(),Body()可以吧pyantic模型声明为查询参数，Body()可以吧单值声明为请求体

特性 | 查询参数 (Query) | 请求体 (Body)
默认定义方式 |"基本类型 (str, int)" | Pydantic 模型
HTTP 位置 | URL (?key=val) | Request Payload (JSON)
数据结构 | 扁平、简单 | 复杂、嵌套、树状
数据大小 | 受 URL 长度限制 (较小) | 无限制 (较大)
安全性 | 低 (记录在日志/历史中) | 高 (加密传输，不留痕)
典型用途 |搜索、分页、排序 (GET) | 创建、更新、表单提交 (POST/PUT)

"""

from enum import Enum
from typing import Annotated, Literal

from fastapi import Body, Cookie, FastAPI, Header, Path, Query
from pydantic import BaseModel, Field, HttpUrl

app = FastAPI()


# /{xxx}是路径参数
@app.get("/items/{item_id}")
async def root(item_id: str):
    return {"message": f"Hello World {item_id}"}


@app.get("/files/{file_path:path}")
async def read_file(
    file_path: Annotated[str, Path(title="文件路径", description="这里输入文件路径")],
):  #:path 告诉它该参数应匹配任何路径 /1561/sddv or 1561/sddv
    return {"file_path": file_path}


# 通过提前定义来声明路径参数，能够在文档更好的展示选项
class ModelName(str, Enum):
    xue = "xue"
    han = "han"
    xin = "xin"


@app.get("/models/{model_name}")
async def get_model(model_name: ModelName):
    match model_name:
        case ModelName.xue:
            return {"message": model_name}
        case ModelName.han:
            return {"message": "han"}
        case ModelName.xin:
            return {"message": "xin"}
        case _:
            return {"message": "unknown"}


# 查询参数
fake_game_db = [{"game_name": "lol"}, {"game_name": "doat"}]


@app.get(
    "/games"
)  # system是可选参数，因为有默认值且默认值为none,只有默认值的为非必填参数
async def get_game(skip: int = 0, limit: int = 1, system: str | None = None):
    return fake_game_db[skip : skip + limit]


# 路径参数与查询参数的混合
# 为查询参数添加字符串验证 Annotated Query,
# 除了Query内的验证手段，在 Annotated 中使用 Pydantic 的 AfterValidator 来进行自定义验证。docs
# AfterValidator 用来做轻量级的、纯逻辑的格式检查（比如你的 startswith）。如果你需要查数据库、查 Redis、调第三方接口，请等待后续章节学习 Depends。
@app.get("/users/{user_id}/items/{item_id}")
async def read_user_item(
    user_id: int,
    item_id: str,
    q: Annotated[
        str | None, Query(min_length=3, max_length=50)
    ] = None,  # str 长度不超50
    short: bool = False,
):
    item = {"item_id": item_id, "owner_id": user_id}
    if q:
        item.update({"q": q})
    if not short:
        item.update(
            {"description": "This is an amazing item that has a long description"}
        )
    return item


# pydantic as query
# model_config 与 Field为查询参数配置示例不生效
class FilterParams(BaseModel):
    model_config = {"extra": "forbid"}
    # FastAPI 允许你在 Pydantic 模型里直接用 Query/Path/Body
    # 由于特性查询参数显示默认值只能通过json_schema_extra and openapi_examples
    limit: int = Query(
        55,
        gt=0,
        le=100,
        description="每页数量",
        openapi_examples={
            "normal": {"summary": "常规值", "value": 10},
            "max": {"summary": "最大值", "value": 100},
        },
    )
    offset: int = Query(0, ge=0, json_schema_extra={"example": 20})
    order_by: Literal["created_at", "updated_at"] = "created_at"
    tags: list[str] = []


# 通过添加Query()，声明这个pydantic model 是查询参数
@app.get("/FilterParams")
async def read_filter(filter_params: Annotated[FilterParams, Query()]):
    return filter_params


# 学习请求体 定义客户端向服务器发送的字段。数据
class Image(BaseModel):
    url: HttpUrl  # 特殊类型
    name: str

    model_config = {
        "extra": "ignore",
        # 生成示例在文档中
        "json_schema_extra": {
            "examples": [
                {"url": "https://example.com/image.jpg", "name": "example_image"}
            ]
        },
    }


# 除了fastapi的body校验，还可以使用pydantic的field校验并且参数相同
class Item(BaseModel):
    name: str = Field(examples=["Foo"])
    description: str | None = Field(
        default=None, title="The description of the item", max_length=300
    )
    price: float = Field(gt=0, description="The price must be greater than zero")
    tax: float | None = None
    image: Image | None = None  # 嵌套


@app.post("/items/")  # 使用 Body 指示 FastAPI 将其视为另一个 body 键
async def create_item(item: Item, importance: Annotated[int, Body()]):
    item_dict = item.model_dump()
    if item.tax is not None:
        price_with_tax = item.price + item.tax
        item_dict.update({"price_with_tax": price_with_tax})
    return item_dict


@app.post("/images/multiple/")
async def create_multiple_images(images: list[Image]):
    return images


# 请求体，路径参数，查询参数的共用
@app.put("/items/{item_id}")
async def update_item(item_id: int, item: Item, q: str | None = None):
    result = {"item_id": item_id, **item.model_dump()}
    if q:
        result.update({"q": q})
    return result


# Cookie参数 docs
@app.get("/items_cookie/")
async def read_items_cookie(ads_id: Annotated[str | None, Cookie()] = None):
    return {"ads_id": ads_id}


# Header参数 docs
@app.get("/items_header/")
async def read_items_header(user_agent: Annotated[str | None, Header()] = None):
    return {"User-Agent": user_agent}
