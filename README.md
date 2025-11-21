# UAV Nest Agent (LangChain)

🤖 一个基于 LangChain 的智能无人机产品问答系统，使用 RAG (Retrieval-Augmented Generation) 技术回答关于 Autel Robotics 无人机产品的问题。

> 💡 **从 Agno 转换而来，使用 LangChain v1.0（2025最新版），无需数据库，免费本地嵌入，完美支持中文！**

## Quickstart for uav-nest-agent-multi-source.py

```bash
deactivate
rm -rf ./.venv
python3.12 -m venv .venv
source .venv/bin/activate
which python3
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 uav-nest-agent-multi-source.py
```

## Quickstart for uav-nest-agent-by-langchain.py

```bash
./setup.sh

source ./.venv/bin/activate

# Optional: when you run the commands on wsl2 ubuntu
uv pip install -U 'httpx[socks]'

# Run single-source RAG demo
python uav-nest-agent-by-langchain.py

# Run multi-source RAG demo (compare 3 merge strategies)
python uav-nest-agent-multi-source.py
```

## 🎯 快速导航

| 我想... | 推荐文档 | 预计时间 |
|---------|----------|---------|
| 🚀 **立即运行单源RAG** | 运行 `python uav-nest-agent-by-langchain.py` | 5 分钟 |
| 🔬 **体验多源RAG** | 跳转到[多源RAG演示](#-多源rag演示体验langchain的局限性) | 10 分钟 |
| 📖 **了解使用方法** | 继续阅读本文档 | 10 分钟 |
| 🎓 **学习 LangChain** | [LangChain 官方文档](https://python.langchain.com/) | 2 小时 |
| 🔍 **理解代码实现** | 查看 `uav-nest-agent-by-langchain.py` | 30 分钟 |

---

## 🔬 多源RAG演示（体验LangChain的局限性）

本项目包含两个版本的RAG实现，用于对比单源和多源场景：

| 版本 | 文件 | 数据源数量 | 适用场景 | 代码量 |
|-----|------|-----------|---------|--------|
| **单源RAG** | `uav-nest-agent-by-langchain.py` | 1个PDF | 简单问答 | ~400行 |
| **多源RAG** | `uav-nest-agent-multi-source.py` | 6个数据源 | 产品对比分析 | ~800行 |

### 🎯 演示目的

对比EVO Nest和DJI Dock的数据存储方案，**清晰展示LangChain在多数据源场景下的5个局限性**：

1. ❌ 需要手动管理多个检索器，缺乏统一的管理接口
2. ❌ EnsembleRetriever无法配置数据源优先级权重
3. ❌ 无法实现条件查询逻辑（如"先查A，不够再查B"）
4. ❌ 缺乏查询路由功能，无法根据问题类型自动选择数据源
5. ❌ 多检索器的日志输出混乱，难以追踪

### 📦 数据源配置

多源RAG使用6个独立的数据源，按优先级分组：

| 优先级 | 数据源 | 文件 | 类型 |
|-------|--------|------|------|
| **P1 (最高)** | EVO Nest技术白皮书 | `evo-nest-data-storage-spec.md` | Markdown |
| **P1 (最高)** | DJI Dock技术白皮书 | `dji-dock-data-storage-spec.md` | Markdown |
| **P2 (中)** | EVO Nest用户手册 | `EN_EVO-Nest-Kit-User-Manual_V1.0.1.pdf` | PDF |
| **P2 (中)** | DJI Dock用户手册 | `M30_Series_Dock_Bundle_User_Manual_v1.8_CHS.pdf` | PDF (关键词过滤) |
| **P3 (低)** | EVO Nest官网介绍 | `evo-nest-official-webpage.md` | Markdown |
| **P3 (低)** | DJI Dock官网介绍 | `dji-dock-official-webpage.md` | Markdown |

### 🎮 运行步骤

**步骤1：准备环境**
```bash
# 确保虚拟环境已激活
source .venv/bin/activate

# 或者使用setup.sh
./setup.sh && source .venv/bin/activate
```

**步骤2：确认数据源文件**

所有6个数据源文件应该已经存在于项目根目录（已包含）：
- ✅ `evo-nest-data-storage-spec.md`
- ✅ `dji-dock-data-storage-spec.md`
- ✅ `evo-nest-official-webpage.md`
- ✅ `dji-dock-official-webpage.md`
- ✅ `EN_EVO-Nest-Kit-User-Manual_V1.0.1.pdf`
- ✅ `M30_Series_Dock_Bundle_User_Manual_v1.8_CHS.pdf`

**步骤3：运行多源RAG演示**

```bash
# 首次运行会创建6个向量库，需要几分钟
python uav-nest-agent-multi-source.py
```

### 📊 预期输出示例

程序会依次展示3种合并策略的效果：

```
═══════════════════════════════════════════════════════════════
🎯 多源RAG策略对比实验
═══════════════════════════════════════════════════════════════

问题：EVO Nest机巢的数据存储方案与DJI Dock机场有什么区别？

───────────────────────────────────────────────────────────────
📊 策略1: 简单拼接
───────────────────────────────────────────────────────────────

🔍 检索阶段：
  ✅ P1 (技术白皮书): 10 chunks
     - EVO Nest技术白皮书: 5 chunks
     - DJI Dock技术白皮书: 5 chunks
  ✅ P2 (用户手册): 10 chunks
     - EVO Nest用户手册: 5 chunks
     - DJI Dock用户手册: 5 chunks
  ✅ P3 (官网介绍): 10 chunks
     - EVO Nest官网介绍: 5 chunks
     - DJI Dock官网介绍: 5 chunks
  📦 总计: 30 chunks

💬 生成答案中...
[详细的产品对比分析，基于30个chunks的信息...]

⏱️  耗时: 15.3秒

───────────────────────────────────────────────────────────────
📊 策略3: RRF融合
───────────────────────────────────────────────────────────────
⚠️  LangChain的EnsembleRetriever无法配置数据源优先级权重

🔍 检索阶段（使用LangChain的EnsembleRetriever）：
  ✅ 融合后返回: 20 chunks (LangChain自动去重和重排序)
  📊 数据源分布:
     - P1: 8 chunks
     - P2: 7 chunks
     - P3: 5 chunks

💬 生成答案中...
[基于RRF融合的对比分析...]

⏱️  耗时: 12.1秒

───────────────────────────────────────────────────────────────
📊 策略4: 优先级过滤
───────────────────────────────────────────────────────────────
⚠️  LangChain标准RAG链不支持条件分支逻辑
   需要完全自定义实现（阈值=8）

🔍 阶段1: 查询P1 (技术白皮书)...
   → 检索到 10 chunks
   ✅ 超过阈值 (8)，使用P1结果
   ⏭️  跳过P2和P3查询（P1结果已足够）

📦 最终使用: 10 chunks (全部来自技术白皮书)

💬 生成答案中...
[基于高质量技术白皮书的精准分析...]

⏱️  耗时: 8.7秒

───────────────────────────────────────────────────────────────
📈 策略对比总结
───────────────────────────────────────────────────────────────

指标                  策略1           策略3           策略4          
──────────────────────────────────────────────────────────────
检索chunks数          30              20              10             
高优先级chunks        10/30 (33%)     8/20 (40%)      10/10 (100%)   
生成耗时(秒)          15.3            12.1            8.7            

⚠️  LangChain在多源RAG场景下的局限性
═══════════════════════════════════════════════════════════════

1. ❌ 需要手动管理多个检索器，缺乏统一的管理接口
   - 必须分别创建6个vector store和6个retriever
   - 没有DataSourceManager或类似的统一抽象

2. ❌ EnsembleRetriever无法配置数据源优先级权重
   - 所有数据源被平等对待
   - 无法体现"技术白皮书>用户手册>官网"的质量差异

3. ❌ 无法实现条件查询逻辑（如"先查A，不够再查B"）
   - 标准RAG Chain不支持条件分支
   - 策略4需要完全自定义实现100+行代码
   - 无法使用LangChain的Chain抽象

4. ❌ 缺乏查询路由功能
   - 无法根据问题类型自动选择数据源
   - 不能实现"技术问题→白皮书，操作问题→手册"

5. ❌ 多检索器的日志输出混乱，难以追踪
   - 需要手动添加大量print语句才能看清过程
   - 没有内置的可观测性工具

💡 这些问题正是LangGraph要解决的！
═══════════════════════════════════════════════════════════════

LangGraph提供：
  ✅ 图结构的工作流编排（支持条件分支、循环）
  ✅ 状态管理和条件路由
  ✅ 多Agent协作能力
  ✅ 内置的可观测性和调试工具

下一步可以探索使用LangGraph重构多源RAG系统。
```

### 🔍 观察要点

通过运行多源RAG演示，您会清晰地看到：

**✅ 策略1（简单拼接）的问题**：
- Context包含30个chunks，可能过长导致LLM无法有效利用所有信息
- 低优先级数据源（官网介绍）的营销性内容混入技术对比中
- 简单粗暴，没有智能筛选

**✅ 策略3（RRF融合）的问题**：
- LangChain的`EnsembleRetriever`无法为数据源设置优先级权重
- 所有数据源被平等对待，无法体现"技术白皮书>用户手册>官网"的质量差异
- 虽然有去重和重排序，但缺乏业务逻辑

**✅ 策略4（优先级过滤）的问题**：
- LangChain的标准RAG链不支持条件分支
- 需要完全自定义实现控制流（100+行代码）
- 无法使用LangChain的Chain抽象
- 虽然效果最好，但实现成本最高

### 📊 单源 vs 多源 RAG 对比

| 维度 | 单源RAG | 多源RAG |
|-----|---------|---------|
| **实现复杂度** | ⭐ 简单 | ⭐⭐⭐⭐⭐ 复杂 |
| **代码量** | ~400行 | ~800行 |
| **向量库数量** | 1个 | 6个 |
| **检索器管理** | 1个retriever | 6个retrievers手动协调 |
| **合并策略** | 无需合并 | 需要自定义3种策略 |
| **LangChain支持度** | ✅ 原生支持 | ⚠️ 部分需要自定义 |
| **适用场景** | 单文档问答 | 多文档对比分析 |
| **局限性** | 不明显 | **非常明显** ⚠️ |

### 💡 为什么需要LangGraph？

通过多源RAG演示，我们发现LangChain在复杂场景下的局限：

- ❌ 缺乏灵活的控制流（条件、循环、分支）
- ❌ 缺乏状态管理（无法记录查询历史）
- ❌ 缺乏动态决策（无法根据结果质量调整策略）
- ❌ 缺乏统一的多源管理接口

**LangGraph** 正是为了解决这些问题而设计的，它提供：

- ✅ 图结构的工作流编排
- ✅ 状态管理和条件路由
- ✅ 多Agent协作能力
- ✅ 内置的可观测性和调试工具

下一步可以探索使用LangGraph重构多源RAG系统，解决上述局限性。

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
│   ├── uav-nest-agent-by-langchain.py        # 单源RAG实现
│   └── uav-nest-agent-multi-source.py        # 多源RAG实现（NEW! 🔬）
│
├── 🛠️ 工具脚本
│   └── setup.sh                              # 自动安装脚本
│
├── 📚 文档
│   └── README.md                             # 完整使用指南（本文档）
│
├── ⚙️ 配置文件
│   ├── .env                                  # 环境变量配置（需手动创建）
│   ├── .gitignore                            # Git 忽略规则
│   └── requirements.txt                      # Python依赖
│
├── 📂 单源RAG数据文件
│   └── Autel-Robotics-Products-Brochure.pdf  # Autel产品手册
│
├── 📂 多源RAG数据文件（6个数据源）
│   ├── evo-nest-data-storage-spec.md         # P1: EVO Nest技术白皮书
│   ├── dji-dock-data-storage-spec.md         # P1: DJI Dock技术白皮书
│   ├── EN_EVO-Nest-Kit-User-Manual_V1.0.1.pdf        # P2: EVO Nest用户手册
│   ├── M30_Series_Dock_Bundle_User_Manual_v1.8_CHS.pdf  # P2: DJI Dock用户手册
│   ├── evo-nest-official-webpage.md          # P3: EVO Nest官网
│   └── dji-dock-official-webpage.md          # P3: DJI Dock官网
│
└── 💾 自动生成
    └── tmp/lancedb/                          # 向量数据库（自动创建）
        ├── uav_nest/                         # 单源RAG向量库
        ├── evo_nest_whitepaper/              # 多源RAG向量库 (6个)
        ├── dji_dock_whitepaper/
        ├── evo_nest_manual/
        ├── dji_dock_manual/
        ├── evo_nest_webpage/
        └── dji_dock_webpage/
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

