`Cookie()` 类的作用非常直接：**它告诉 FastAPI：“不要去 URL 查询参数里找这个值，而是去 HTTP 请求头的 Cookie 里面找。”**

如果没有 `Cookie()`，FastAPI 会默认把 `ads_id` 当作**查询参数**（Query Parameter）。

我们可以从以下 3 个方面来理解它的作用：

### 1. 改变“查找数据的位置”

这是最核心的作用。

* **不加 `Cookie()` 时**：
FastAPI 会去 URL 里找 `?ads_id=xyz`。
* **加上 `Cookie()` 后**：
FastAPI 会忽略 URL，转而去读取 HTTP 请求头中的 `Cookie` 字段。

**模拟 HTTP 请求的对比：**

假设浏览器发送了一个请求，Cookie 里存了 `ads_id=123`。

```http
GET /items_cookie/ HTTP/1.1
Host: localhost:8000
Cookie: ads_id=123   <--- 你的代码就是从这里提取数据

```

### 2. 作为“元数据”声明

就像 `Query()` 和 `Path()` 一样，`Cookie()` 也是 `Param` 类的兄弟。它允许你为这个参数定义额外的校验和文档信息：

```python
# 可以在 Cookie() 里加参数，就像在 Query() 里一样
ads_id: Annotated[str | None, Cookie(
    title="广告ID",
    description="用于追踪用户的广告来源",
    alias="ADS_ID_KEY" # 如果 Cookie 里的键名和变量名不一样
)] = None

```

### 3. 区分数据来源（安全与逻辑）

在 HTTP 协议中，Cookie、Header 和 Query 是完全不同的载体。

* **Query**: 用户可见，容易伪造，存在历史记录里。
* **Cookie**: 通常由浏览器自动携带，用于维持会话（Session）、追踪用户习惯等。

使用 `Cookie()` 类，你可以明确地在代码层面声明：“这个 `ads_id` 必须是来自浏览器的 Cookie 环境，而不是用户在地址栏里随便敲进去的一个参数。”

### 总结

`Cookie()` 的作用就是**“定位符”**。

* **`Path()`**  去路由路径里找 (`/items/{id}`)
* **`Query()`**  去 `?` 后面找 (`?id=...`)
* **`Body()`**  去请求体里找 (JSON)
* **`Header()`**  去请求头里找
* **`Cookie()`**  **去 Cookie 字段里找**

**注意**：这个类仅用于**读取** Cookie。如果你想**设置**（发送）Cookie 给客户端，你需要使用 `Response` 对象。
