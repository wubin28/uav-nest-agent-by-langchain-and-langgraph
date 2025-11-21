# UAV Nest Agent (LangChain)

🤖 一个基于 LangChain 的智能无人机产品问答系统，使用 RAG (Retrieval-Augmented Generation) 技术回答关于 Autel Robotics 无人机产品的问题。

> 💡 **从 Agno 转换而来，使用 LangChain v1.0（2025最新版），无需数据库，免费本地嵌入，完美支持中文！**

## Quickstart for uav-nest-agent-by-langchain.py

```bash
python3.12 -m venv .venv
source .venv/bin/activate
which python3
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

# Optional: when you run the commands on wsl2 ubuntu
uv pip install -U 'httpx[socks]'

# Run single-source RAG demo
python3 uav-nest-agent-by-langchain.py
```

## 🎯 快速导航

| 我想... | 推荐文档 | 预计时间 |
|---------|----------|---------|
| 🚀 **立即运行** | 运行 `./setup.sh` | 5 分钟 |
| 📖 **了解使用方法** | 继续阅读本文档 | 10 分钟 |
| 🎓 **学习 LangChain** | [LangChain 官方文档](https://python.langchain.com/) | 2 小时 |
| 🔍 **理解代码实现** | 查看 `uav-nest-agent-by-langchain.py` | 30 分钟 |

---

## ⚡ 超快速开始（两种方式）

### 方式 A: 自动安装脚本（最简单）

```bash
# 1. 进入项目目录
cd /Users/binwu/OOR-local/katas/uav-nest-agent

# 2. 运行自动安装脚本
./setup.sh

# 3. 按提示配置 API Key 和 PDF 文件

# 4. 运行
python uav-nest-agent-by-langchain.py
```

### 方式 B: 手动安装（5 分钟）

```bash
# 1. 进入项目目录
cd /Users/binwu/OOR-local/katas/uav-nest-agent

# 2. 创建虚拟环境并激活（使用 Python 3.12）
python3.12 -m venv venv && source venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 准备 DeepSeek API Key（从 https://platform.deepseek.com/ 获取）
# 注意：无需创建 .env 文件，运行时会提示输入

# 5. 放置 PDF 文件（命名为 Autel-Robotics-Products-Brochure.pdf）

# 6. 测试安装
python test_installation.py  # 验证所有依赖

# 7. 运行（会提示输入 API Key，输入时隐藏显示）
python uav-nest-agent-by-langchain.py  # 演示模式
```

---

## ✨ 特性

- ✅ **无需数据库安装**：使用 LanceDB 本地向量存储，无需安装 PostgreSQL 等数据库
- 🆓 **免费本地嵌入**：使用 FastEmbed 进行文本嵌入，无需 OpenAI API Key
- 🧠 **智能推理**：使用 DeepSeek Reasoner 模型进行智能回答
- 📚 **源文件引用**：自动引用来源页码，确保答案可追溯
- 🚀 **快速启动**：简单配置即可运行
- 💬 **中文问答**：优化的中文提示词，支持中文问答
- 🔄 **交互式模式**：支持持续问答，无需重启
- 📦 **2025 最新版本**：使用 LangChain v1.0+ 最新稳定版本

## 📋 前置要求

- macOS 系统
- **Python 3.12**（重要：必须是 3.12，不支持 3.13+）
- iTerm2 或其他终端
- DeepSeek API Key（[获取地址](https://platform.deepseek.com/)）
- Autel Robotics 产品手册 PDF 文件（命名为 `Autel-Robotics-Products-Brochure.pdf`）

> ⚠️ **Python 版本说明**：本项目需要 Python 3.12。如果你的系统是 Python 3.13+，某些依赖包（如 fastembed）可能不兼容。请安装 Python 3.12：
> ```bash
> # macOS 用户
> brew install python@3.12
> ```

## 🚀 快速开始

### 步骤 1: 克隆或进入项目目录

```bash
cd /Users/binwu/OOR-local/katas/uav-nest-agent
```

### 步骤 2: 创建 Python 虚拟环境（使用 Python 3.12）

```bash
# 创建虚拟环境
python3.12 -m venv venv

# 激活虚拟环境
source venv/bin/activate
```

激活成功后，你的终端提示符前面会显示 `(venv)`。

### 步骤 3: 安装依赖

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

这个过程可能需要几分钟，会安装以下主要组件：
- LangChain 核心库
- LanceDB（本地向量数据库）
- FastEmbed（免费的本地嵌入模型）
- PyPDF（PDF 解析）
- DeepSeek LLM 支持

### 步骤 4: 准备 API Key（两种方式）

**方式 A: 运行时输入（推荐，更安全）**

直接运行程序，会自动提示输入 API Key（输入时会以星号隐藏显示）：
- ✅ 无需创建配置文件
- ✅ API Key 不会保存到文件
- ✅ 更加安全

**方式 B: 预先配置 .env 文件（可选，避免每次输入）**

```bash
# 复制示例配置文件
cp env.example.txt .env

# 编辑配置文件
nano .env

# 填入: DEEPSEEK_API_KEY=your_actual_deepseek_api_key_here
```

**如何获取 DeepSeek API Key：**
1. 访问 [DeepSeek 平台](https://platform.deepseek.com/)
2. 注册/登录账号
3. 在控制台创建 API Key
4. 准备好在运行时输入（或保存到 .env）

### 步骤 5: 准备 PDF 文件

将 Autel Robotics 产品手册 PDF 文件放在项目根目录，命名为 `Autel-Robotics-Products-Brochure.pdf`：

```bash
# 如果 PDF 在其他位置，可以复制或创建软链接
# cp /path/to/your/brochure.pdf ./Autel-Robotics-Products-Brochure.pdf
```

### 步骤 6: 运行应用

运行演示程序：

```bash
python uav-nest-agent-by-langchain.py
```
运行预设的示例查询，快速了解系统功能。

## 📖 运行过程说明

首次运行时，应用会：

1. **提示输入 API Key（如果未配置）**
   ```
   🔑 请输入你的 DEEPSEEK_API_KEY
      获取地址: https://platform.deepseek.com/
      API Key (输入时会隐藏): ********
   ✅ DEEPSEEK_API_KEY 已接收
   ```

2. **初始化嵌入模型**
   ```
   ✅ Using FastEmbedEmbeddings (free local embedder)
   🔧 Initializing FastEmbed embeddings (free, local)...
   ```

3. **加载 PDF 并创建索引**
   ```
   📄 Loading PDF from ./Autel-Robotics-Products-Brochure.pdf...
   ✅ Loaded X pages from PDF
   ✂️  Splitting documents into chunks...
   ✅ Created X text chunks
   🗄️  Creating vector store (this may take a moment)...
   ✅ Vector store created at ./tmp/lancedb
   ```

3. **执行示例查询**
   ```
   ❓ Question: Autel Robotics 有哪些主要的无人机产品？
   💬 Answer:
   [答案内容...]
   ```

第二次及以后运行时，会直接加载已存在的向量数据库，速度更快：
```
📂 Loading existing vector store from ./tmp/lancedb...
✅ Vector store loaded successfully
```

## 🔧 使用指南

### 修改演示程序

如果想修改演示程序的问题，编辑 `uav-nest-agent-by-langchain.py` 的 `main()` 函数：

```python
# 修改查询问题
agent.ask(
    "Autel 无人机的电池续航时间是多少？",  # 修改这里
    stream=True
)
```

## 📁 项目结构

```
uav-nest-agent/
├── 📄 核心代码
│   └── uav-nest-agent-by-langchain.py   # 主应用代码（核心 RAG 实现）
│
├── 🛠️ 工具脚本
│   └── setup.sh                         # 自动安装脚本
│
├── 📚 文档
│   └── README.md                        # 完整使用指南（本文档）
│
├── ⚙️ 配置文件
│   ├── .env                             # 环境变量配置（需手动创建）
│   └── .gitignore                       # Git 忽略规则
│
├── 📂 数据文件
│   └── Autel-Robotics-Products-Brochure.pdf  # Autel Robotics 产品手册 PDF
│
└── 💾 自动生成
    └── tmp/lancedb/                     # 向量数据库（自动创建）
```

## 🔍 核心技术架构

### LangChain 实现

| 组件 | 技术选型 |
|------|----------|
| **向量数据库** | `LanceDB` (from langchain_community) |
| **嵌入模型** | `FastEmbedEmbeddings` (免费本地) |
| **LLM** | `ChatOpenAI(model="deepseek-reasoner")` |
| **文档加载** | `PyPDFLoader` |
| **文本分割** | `RecursiveCharacterTextSplitter` |
| **RAG 实现** | 自定义 RAG Chain（Retriever + Prompt + LLM） |
| **流式输出** | `rag_chain.stream()` |

### LangChain 核心组件说明

1. **Document Loaders**: `PyPDFLoader` - 加载 PDF 文档
2. **Text Splitters**: `RecursiveCharacterTextSplitter` - 智能分割文本
3. **Embeddings**: `FastEmbedEmbeddings` - 将文本转换为向量
4. **Vector Store**: `LanceDB` - 存储和检索向量
5. **Retriever**: 从向量库检索相关文档
6. **Prompt Template**: 定义 LLM 的提示词格式
7. **LLM**: `ChatOpenAI` - 生成回答
8. **Chain**: 使用 LCEL (LangChain Expression Language) 组合组件

## ❓ 常见问题

### Q1: Python 版本要求

**Q**: 为什么必须使用 Python 3.12？

**A**: 
- 本项目使用经过充分测试的依赖版本组合
- Python 3.12 是当前最稳定的生产版本
- Python 3.13 对某些包的兼容性还在完善中
- 使用 Python 3.12 可以确保所有依赖正常安装和稳定运行

**如何安装 Python 3.12**：
```bash
# macOS 用户
brew install python@3.12

# 验证安装
python3.12 --version
```

### Q2: 首次运行很慢怎么办？

**A**: 首次运行需要：
- 下载 FastEmbed 模型（约 100MB）
- 解析 PDF 并创建向量索引

这是正常的，后续运行会直接使用缓存的向量数据库，速度会快很多。

### Q3: 如何重建索引？

**A**: 如果 PDF 内容更新了，需要重建索引：

```python
# 在代码中设置 force_reload=True
agent.load_and_index_pdf(force_reload=True)
```

或者手动删除向量数据库：

```bash
rm -rf tmp/lancedb
python uav-nest-agent-by-langchain.py
```

### Q4: 如何使用 OpenAI Embeddings 替代 FastEmbed？

**A**: 修改初始化代码：

```python
agent = UAVNestAgent(
    pdf_path="./Autel-Robotics-Products-Brochure.pdf",
    use_fastembed=False,  # 改为 False
)
```

并在 `.env` 中添加：
```
OPENAI_API_KEY=your_openai_api_key_here
```

### Q5: 如何调整检索的文档数量？

**A**: 编辑 `_setup_retriever()` 方法中的 `k` 参数：

```python
self.retriever = self.vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 8}  # 默认是 4，可以改为 8
)
```

### Q6: DeepSeek API 调用失败怎么办？

**A**: 检查以下几点：
- API Key 是否正确
- 网络连接是否正常
- DeepSeek 账户是否有余额
- 尝试更换为其他模型（如 `deepseek-chat`）

### Q7: 如何更换 PDF 文件？

**A**: 
```bash
# 1. 替换 PDF 文件
cp /path/to/new-brochure.pdf ./Autel-Robotics-Products-Brochure.pdf

# 2. 删除旧索引
rm -rf tmp/lancedb

# 3. 重新运行
python uav-nest-agent-by-langchain.py
```

## 🛠️ 高级配置

### 调整文本分割参数

编辑 `load_and_index_pdf()` 方法：

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,      # 增大块大小以包含更多上下文
    chunk_overlap=300,    # 增大重叠以提高连续性
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)
```

### 调整 LLM 参数

编辑 `_initialize_llm()` 方法：

```python
return ChatOpenAI(
    model="deepseek-reasoner",
    openai_api_key=DEEPSEEK_API_KEY,
    openai_api_base="https://api.deepseek.com/v1",
    temperature=0.3,     # 0-1，越高越有创造性
    max_tokens=8000,     # 增大输出长度
)
```

### 更换不同的 FastEmbed 模型

编辑 `_initialize_embeddings()` 方法：

```python
return FastEmbedEmbeddings(
    model_name="BAAI/bge-base-en-v1.5"  # 更大更准确的模型
)
```

可用模型列表：
- `BAAI/bge-small-en-v1.5` - 快速，准确度中等（默认）
- `BAAI/bge-base-en-v1.5` - 中速，准确度高
- `BAAI/bge-large-en-v1.5` - 慢速，准确度最高

## 🧹 清理和维护

### 停用虚拟环境

```bash
deactivate
```

### 清理临时文件

```bash
# 清理 Python 缓存
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

# 清理向量数据库
rm -rf tmp/lancedb
```

### 完全重置项目

```bash
# 删除虚拟环境
rm -rf venv

# 删除所有临时文件
rm -rf tmp __pycache__ .pytest_cache

# 重新开始（使用 Python 3.12）
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 📚 学习资源

### 官方文档

- [LangChain 官方文档](https://python.langchain.com/)
- [LangChain RAG 教程](https://python.langchain.com/docs/tutorials/rag/)
- [DeepSeek API 文档](https://platform.deepseek.com/docs)
- [LanceDB 文档](https://lancedb.github.io/lancedb/)
- [FastEmbed 文档](https://qdrant.github.io/fastembed/)

### 推荐学习顺序

1. 运行 `uav-nest-agent-by-langchain.py` 体验功能
2. 阅读代码理解 RAG 实现
3. 修改代码尝试不同配置
4. 查阅官方文档学习更多

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可

MIT License

## 🙋 支持

如果遇到问题，请检查：
1. Python 版本是否 >= 3.9
2. 所有依赖是否正确安装
3. `.env` 文件配置是否正确
4. PDF 文件是否存在且可读

---

**祝你使用愉快！🎉**

如有任何问题，欢迎随时提问。

