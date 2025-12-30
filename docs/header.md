`Header()` 类的作用和刚才提到的 `Cookie()` 非常相似，核心也是为了**指定数据来源**，但它多了一个非常关键的**自动转换功能**。

它的作用可以总结为两点：

### 1. 定位数据来源：去 HTTP 请求头里找

* **默认行为**：如果你只写 `user_agent: str`，FastAPI 会以为这是个**查询参数**，去 URL 里找 `?user_agent=...`。
* **使用了 `Header()**`：FastAPI 会忽略 URL，直接去 **HTTP Request Headers** 区域查找对应的值。

### 2. 核心魔法：自动转换下划线 (`_`) 到连字符 (`-`)

这是 `Header()` 最特殊的地方。

在 Python 变量命名中，我们习惯使用**下划线**（snake_case），比如 `user_agent`。
但在 HTTP 协议中，Header 的字段名习惯使用**连字符**（Kebab-Case），比如 `User-Agent`。

**`Header()` 默认帮你做了翻译：**

* 你在 Python 里写：`user_agent`
* FastAPI 自动去 Header 里找：`user-agent` (忽略大小写)

**为什么要这样？**
因为 `User-Agent` 在 Python 中不是一个合法的变量名（减号会被当成减法运算）。如果没有这个自动转换，你根本没法定义一个变量来接收它。

#### 实际流程演示：

1. **浏览器发送请求**：
```http
GET /items_header/ HTTP/1.1
Host: example.com
User-Agent: Mozilla/5.0 (Windows NT 10.0; ...)  <-- 真实 Header 是带横杠的

```


2. **FastAPI 接收**：
* 看到你的变量叫 `user_agent`。
* 因为你用了 `Header()`，它自动把 `_` 变成 `-`。
* 去请求头里找 `user-agent`。
* 找到了！把值赋给你的变量。



### 3. 如何禁止这种“自作聪明”的转换？

如果你真的有一个自定义 Header 就叫 `strange_header`（带下划线），你可以关掉在这个功能：

```python
# 强制去 Header 里找 "strange_header"，而不是 "strange-header"
data: str = Header(convert_underscores=False)

```

### 总结

`Header()` 的作用是：

1. **位置**：告诉 FastAPI 去 **HTTP Headers** 里取值。
2. **翻译**：把 Python 的 `variable_name` 自动映射为 HTTP 的 `Variable-Name`。
3. **不区分大小写**：HTTP 协议规定 Header 不区分大小写，所以 `user-agent` 和 `User-Agent` 是一样的。
