"""
LangChain Multi-Source RAG Agent (Educational Demo)

This application demonstrates LangChain's capabilities and limitations in multi-source
RAG scenarios by comparing 3 different merge strategies.

Purpose: Compare data storage solutions between EVO Nest and DJI Dock using 6 data sources.

LangChain Limitations Demonstrated:
1. Manual management of multiple retrievers (no unified interface)
2. EnsembleRetriever cannot configure data source priority weights
3. No support for conditional queries (e.g., "query A first, then B if insufficient")
4. Lack of query routing based on question type
5. Messy logging output from multiple retrievers

Features:
- 6 independent vector stores (technical whitepapers, manuals, webpages)
- 3 merge strategies: Simple Concatenation, RRF Fusion, Priority Filtering
- Detailed logging to observe each strategy's behavior
- Clear annotation of LangChain's limitations
"""

import os
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import getpass

from dotenv import load_dotenv
import lancedb

from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader
from langchain_community.vectorstores import LanceDB
from langchain_core.documents import Document
from langchain.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Try importing FastEmbed (FREE local embedder)
try:
    from langchain_community.embeddings import FastEmbedEmbeddings
    FASTEMBED_AVAILABLE = True
    print("✅ Using FastEmbedEmbeddings (free local embedder)")
except ImportError as e:
    FASTEMBED_AVAILABLE = False
    print(f"⚠️  FastEmbedEmbeddings not available: {e}")
    print("   Install with: pip install fastembed")

# Try importing OpenAI embeddings as fallback
try:
    from langchain_openai import OpenAIEmbeddings
    OPENAI_EMBEDDINGS_AVAILABLE = True
except ImportError:
    OPENAI_EMBEDDINGS_AVAILABLE = False


# Load environment variables from .env file (optional)
load_dotenv()

# Load API keys (will be set interactively if not found)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def get_api_key_interactive(key_name: str = "DEEPSEEK_API_KEY") -> str:
    """
    交互式获取 API Key，以星号形式显示输入。
    
    Args:
        key_name: API Key 的名称
        
    Returns:
        用户输入的 API Key
    """
    print(f"\n🔑 请输入你的 {key_name}")
    if key_name == "DEEPSEEK_API_KEY":
        print("   获取地址: https://platform.deepseek.com/")
    else:
        print("   获取地址: https://platform.openai.com/")
    
    api_key = getpass.getpass("   API Key (输入时会隐藏): ").strip()
    
    if not api_key:
        raise ValueError(f"❌ {key_name} 不能为空")
    
    print(f"✅ {key_name} 已接收\n")
    return api_key


class MultiSourceRAGAgent:
    """
    Multi-Source RAG Agent (Educational Demo)
    
    This implementation is designed to showcase LangChain's limitations in multi-source scenarios:
    1. Need to manually manage 6 independent vector stores and retrievers
    2. EnsembleRetriever cannot configure data source priority weights
    3. Cannot implement conditional queries (e.g., "query A first, then B if needed")
    4. Lack of query routing (cannot auto-select data sources based on question type)
    
    These issues motivate the need for LangGraph for more complex workflows.
    
    Data Sources (6 total):
    - Priority 1 (P1): Technical whitepapers (EVO Nest, DJI Dock)
    - Priority 2 (P2): User manuals (EVO Nest, DJI Dock)
    - Priority 3 (P3): Official webpages (EVO Nest, DJI Dock)
    """
    
    # ⚠️ LangChain Limitation #3: Need to manually define and manage data source configuration
    # No built-in DataSourceManager or similar abstraction
    DATA_SOURCES = [
        {
            "id": "evo_nest_whitepaper",
            "name": "EVO Nest技术白皮书",
            "file_path": "evo-nest-data-storage-spec.md",
            "file_type": "markdown",
            "priority": 1,  # Highest priority
            "product": "evo_nest",
            "source_type": "whitepaper",
        },
        {
            "id": "dji_dock_whitepaper",
            "name": "DJI Dock技术白皮书",
            "file_path": "dji-dock-data-storage-spec.md",
            "file_type": "markdown",
            "priority": 1,  # Highest priority
            "product": "dji_dock",
            "source_type": "whitepaper",
        },
        {
            "id": "evo_nest_manual",
            "name": "EVO Nest用户手册",
            "file_path": "EN_EVO-Nest-Kit-User-Manual_V1.0.1.pdf",
            "file_type": "pdf",
            "priority": 2,  # Medium priority
            "product": "evo_nest",
            "source_type": "manual",
        },
        {
            "id": "dji_dock_manual",
            "name": "DJI Dock用户手册",
            "file_path": "M30_Series_Dock_Bundle_User_Manual_v1.8_CHS.pdf",
            "file_type": "pdf",
            "priority": 2,  # Medium priority
            "product": "dji_dock",
            "source_type": "manual",
            "filter_keywords": ["存储", "数据管理", "内存", "SD", "数据传输", "备份", "容量"],
        },
        {
            "id": "evo_nest_webpage",
            "name": "EVO Nest官网介绍",
            "file_path": "evo-nest-official-webpage.md",
            "file_type": "markdown",
            "priority": 3,  # Lowest priority
            "product": "evo_nest",
            "source_type": "webpage",
        },
        {
            "id": "dji_dock_webpage",
            "name": "DJI Dock官网介绍",
            "file_path": "dji-dock-official-webpage.md",
            "file_type": "markdown",
            "priority": 3,  # Lowest priority
            "product": "dji_dock",
            "source_type": "webpage",
        },
    ]
    
    def __init__(
        self,
        vector_db_path: str = "./tmp/lancedb",
        use_fastembed: bool = True,
        deepseek_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
    ):
        """
        Initialize the Multi-Source RAG Agent.
        
        Args:
            vector_db_path: Base path for LanceDB vector databases
            use_fastembed: Whether to use FastEmbed (free) or OpenAI embeddings
            deepseek_api_key: DeepSeek API Key (optional, will prompt if not provided)
            openai_api_key: OpenAI API Key (optional, only needed if not using FastEmbed)
        """
        self.vector_db_path = vector_db_path
        
        # Get DeepSeek API Key
        self.deepseek_api_key = deepseek_api_key or DEEPSEEK_API_KEY
        if not self.deepseek_api_key:
            self.deepseek_api_key = get_api_key_interactive("DEEPSEEK_API_KEY")
        
        # Get OpenAI API Key if needed
        self.openai_api_key = openai_api_key or OPENAI_API_KEY
        if not use_fastembed and not self.openai_api_key:
            self.openai_api_key = get_api_key_interactive("OPENAI_API_KEY")
        
        # Initialize embeddings
        self.embeddings = self._initialize_embeddings(use_fastembed)
        
        # Initialize LLM (DeepSeek)
        self.llm = self._initialize_llm()
        
        # Storage for vector stores and retrievers
        # ⚠️ LangChain Limitation #3: Need to manually track multiple vector stores and retrievers
        # No unified management interface
        self.vector_stores: Dict[str, LanceDB] = {}
        self.retrievers: Dict[str, any] = {}
        
        # Group retrievers by priority for strategy 4
        self.retrievers_by_priority: Dict[int, List[any]] = {1: [], 2: [], 3: []}
        
        # RAG chains for different strategies
        self.rag_chains: Dict[str, any] = {}
    
    @staticmethod
    def filter_documents_by_keywords(
        documents: List[Document],
        keywords: List[str]
    ) -> List[Document]:
        """
        Filter documents by keywords (for large PDF files).
        
        Args:
            documents: List of documents to filter
            keywords: List of keywords to search for
            
        Returns:
            Filtered list of documents containing at least one keyword
        """
        filtered = []
        for doc in documents:
            content_lower = doc.page_content.lower()
            # Check if any keyword exists in the content
            if any(keyword.lower() in content_lower for keyword in keywords):
                filtered.append(doc)
        return filtered
    
    def _load_markdown_file(self, file_path: str, source_config: dict) -> List[Document]:
        """
        Load a Markdown file and add metadata.
        
        Args:
            file_path: Path to the Markdown file
            source_config: Configuration dict for this data source
            
        Returns:
            List of documents with metadata
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"❌ File not found: {file_path}")
        
        print(f"  📄 Loading {source_config['name']} from {file_path}...")
        loader = UnstructuredMarkdownLoader(file_path)
        documents = loader.load()
        
        # Add metadata
        for doc in documents:
            doc.metadata.update({
                "source_id": source_config["id"],
                "source_name": source_config["name"],
                "source_type": source_config["source_type"],
                "priority": source_config["priority"],
                "product": source_config["product"],
                "source_file": file_path,
            })
        
        print(f"  ✅ Loaded {len(documents)} documents")
        return documents
    
    def _load_pdf_file(
        self,
        file_path: str,
        source_config: dict,
        filter_keywords: Optional[List[str]] = None
    ) -> List[Document]:
        """
        Load a PDF file and optionally filter by keywords.
        
        Args:
            file_path: Path to the PDF file
            source_config: Configuration dict for this data source
            filter_keywords: Optional list of keywords to filter content
            
        Returns:
            List of documents with metadata
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"❌ File not found: {file_path}")
        
        print(f"  📄 Loading {source_config['name']} from {file_path}...")
        loader = PyPDFLoader(file_path)
        
        try:
            documents = loader.load()
            print(f"  ✅ Loaded {len(documents)} pages")
            
            # Filter by keywords if specified
            if filter_keywords:
                print(f"  🔍 Filtering by keywords: {', '.join(filter_keywords)}")
                documents = self.filter_documents_by_keywords(documents, filter_keywords)
                print(f"  ✅ Filtered to {len(documents)} relevant pages")
            
            # Add metadata
            for doc in documents:
                doc.metadata.update({
                    "source_id": source_config["id"],
                    "source_name": source_config["name"],
                    "source_type": source_config["source_type"],
                    "priority": source_config["priority"],
                    "product": source_config["product"],
                    "source_file": file_path,
                })
            
            return documents
            
        except Exception as e:
            print(f"  ⚠️  Error loading PDF with PyPDFLoader: {e}")
            print(f"  💡 Tip: If this is a binary PDF, try installing: pip install unstructured")
            raise
    
    def load_all_data_sources(self, force_reload: bool = False) -> Dict[str, List[Document]]:
        """
        Load all 6 data sources and return documents grouped by source ID.
        
        Args:
            force_reload: If True, reload even if already loaded
            
        Returns:
            Dictionary mapping source_id to list of documents
        """
        print("\n" + "="*60)
        print("📚 Loading All Data Sources")
        print("="*60)
        
        all_documents = {}
        
        # ⚠️ LangChain Limitation #3: Need to manually iterate and load each data source
        # No built-in batch loading or data source manager
        for source_config in self.DATA_SOURCES:
            source_id = source_config["id"]
            file_path = source_config["file_path"]
            file_type = source_config["file_type"]
            
            print(f"\n[{source_config['priority']}/P{source_config['priority']}] {source_config['name']}")
            
            try:
                if file_type == "markdown":
                    documents = self._load_markdown_file(file_path, source_config)
                elif file_type == "pdf":
                    filter_keywords = source_config.get("filter_keywords")
                    documents = self._load_pdf_file(file_path, source_config, filter_keywords)
                else:
                    raise ValueError(f"Unsupported file type: {file_type}")
                
                all_documents[source_id] = documents
                
            except Exception as e:
                print(f"  ❌ Error loading {source_id}: {e}")
                # Store empty list for failed sources
                all_documents[source_id] = []
        
        print("\n" + "="*60)
        print(f"✅ Successfully loaded {len([d for d in all_documents.values() if d])} data sources")
        print("="*60)
        
        return all_documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks using RecursiveCharacterTextSplitter.
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of chunked documents
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Smaller than single-source RAG to fit more sources
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", "。", "；", " ", ""]  # Support Chinese and English
        )
        
        chunks = text_splitter.split_documents(documents)
        return chunks
    
    def create_vector_stores(
        self,
        all_documents: Dict[str, List[Document]],
        force_reload: bool = False
    ) -> Dict[str, LanceDB]:
        """
        Create vector stores for all data sources.
        
        Args:
            all_documents: Dictionary mapping source_id to documents
            force_reload: If True, recreate vector stores even if they exist
            
        Returns:
            Dictionary mapping source_id to LanceDB vector store
        """
        print("\n" + "="*60)
        print("🗄️  Creating Vector Stores")
        print("="*60)
        
        db = lancedb.connect(self.vector_db_path)
        vector_stores = {}
        
        # ⚠️ LangChain Limitation #3: Need to manually create each vector store
        # No batch processing or unified vector store manager
        for source_id, documents in all_documents.items():
            if not documents:
                print(f"\n⚠️  Skipping {source_id} (no documents loaded)")
                continue
            
            source_config = next(s for s in self.DATA_SOURCES if s["id"] == source_id)
            table_name = source_id
            
            print(f"\n[P{source_config['priority']}] {source_config['name']}")
            print(f"  📊 Table: {table_name}")
            
            # Check if table already exists
            table_exists = table_name in db.table_names()
            
            if table_exists and not force_reload:
                try:
                    table = db.open_table(table_name)
                    row_count = table.count_rows()
                    if row_count > 0:
                        print(f"  📂 Loading existing table ({row_count} rows)")
                        vector_store = LanceDB(
                            connection=db,
                            embedding=self.embeddings,
                            table_name=table_name,
                        )
                        vector_stores[source_id] = vector_store
                        continue
                except Exception as e:
                    print(f"  ⚠️  Error loading existing table: {e}")
            
            # Split documents into chunks
            print(f"  ✂️  Splitting {len(documents)} documents into chunks...")
            chunks = self.split_documents(documents)
            print(f"  ✅ Created {len(chunks)} chunks")
            
            # Create vector store
            print(f"  🔧 Creating vector store (this may take a moment)...")
            vector_store = LanceDB.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                connection=db,
                table_name=table_name,
            )
            vector_stores[source_id] = vector_store
            print(f"  ✅ Vector store created")
        
        print("\n" + "="*60)
        print(f"✅ Successfully created {len(vector_stores)} vector stores")
        print("="*60)
        
        self.vector_stores = vector_stores
        return vector_stores
    
    def create_retrievers(self, k: int = 5) -> Dict[str, any]:
        """
        Create retrievers for all vector stores.
        
        Args:
            k: Number of documents to retrieve per source
            
        Returns:
            Dictionary mapping source_id to retriever
        """
        print("\n" + "="*60)
        print("🔍 Creating Retrievers")
        print("="*60)
        
        retrievers = {}
        
        # ⚠️ LangChain Limitation #3: Need to manually create each retriever
        # No unified retriever manager or batch creation
        for source_id, vector_store in self.vector_stores.items():
            source_config = next(s for s in self.DATA_SOURCES if s["id"] == source_id)
            
            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k}
            )
            
            retrievers[source_id] = retriever
            
            # Group by priority for strategy 4
            priority = source_config["priority"]
            self.retrievers_by_priority[priority].append(retriever)
            
            print(f"  ✅ [P{priority}] {source_config['name']}: retriever created (k={k})")
        
        print("\n" + "="*60)
        print(f"✅ Created {len(retrievers)} retrievers")
        print(f"   P1 (Whitepapers): {len(self.retrievers_by_priority[1])} retrievers")
        print(f"   P2 (Manuals): {len(self.retrievers_by_priority[2])} retrievers")
        print(f"   P3 (Webpages): {len(self.retrievers_by_priority[3])} retrievers")
        print("="*60)
        
        self.retrievers = retrievers
        return retrievers
    
    def retrieve_strategy_simple_concat(self, query: str) -> Tuple[List[Document], Dict]:
        """
        Strategy 1: Simple Concatenation
        
        Retrieve from all sources and concatenate results in priority order.
        
        Args:
            query: Query string
            
        Returns:
            Tuple of (documents list, statistics dict)
        """
        print("\n" + "─"*60)
        print("📊 策略1: 简单拼接 (Simple Concatenation)")
        print("─"*60)
        
        all_docs = []
        stats = {
            "P1": {"count": 0, "sources": []},
            "P2": {"count": 0, "sources": []},
            "P3": {"count": 0, "sources": []},
        }
        
        print("\n🔍 检索阶段：")
        
        # Retrieve from P1, P2, P3 in order
        for priority in [1, 2, 3]:
            priority_docs = []
            
            for retriever in self.retrievers_by_priority[priority]:
                docs = retriever.get_relevant_documents(query)
                priority_docs.extend(docs)
            
            # Get source names for this priority
            priority_sources = set()
            for doc in priority_docs:
                if "source_name" in doc.metadata:
                    priority_sources.add(doc.metadata["source_name"])
            
            stats[f"P{priority}"]["count"] = len(priority_docs)
            stats[f"P{priority}"]["sources"] = list(priority_sources)
            
            all_docs.extend(priority_docs)
            
            priority_label = {1: "技术白皮书", 2: "用户手册", 3: "官网介绍"}[priority]
            print(f"  ✅ P{priority} ({priority_label}): {len(priority_docs)} chunks")
            for source in sorted(priority_sources):
                source_count = len([d for d in priority_docs if d.metadata.get("source_name") == source])
                print(f"     - {source}: {source_count} chunks")
        
        print(f"  📦 总计: {len(all_docs)} chunks")
        
        return all_docs, stats
    
    def retrieve_strategy_rrf(self, query: str) -> Tuple[List[Document], Dict]:
        """
        Strategy 3: RRF Fusion (using LangChain's EnsembleRetriever)
        
        Uses Reciprocal Rank Fusion to combine results from all retrievers.
        
        ⚠️ LangChain Limitation #1: EnsembleRetriever cannot configure data source priority weights.
        All retrievers are treated equally, cannot reflect quality differences.
        
        Args:
            query: Query string
            
        Returns:
            Tuple of (documents list, statistics dict)
        """
        print("\n" + "─"*60)
        print("📊 策略3: RRF融合 (RRF Fusion)")
        print("─"*60)
        print("⚠️  LangChain的EnsembleRetriever无法配置数据源优先级权重")
        
        # ⚠️ LangChain Limitation #1: Cannot set weights for different data sources
        # All retrievers are treated equally in EnsembleRetriever
        all_retrievers = list(self.retrievers.values())
        
        # Create ensemble retriever with equal weights
        ensemble_retriever = EnsembleRetriever(
            retrievers=all_retrievers,
            weights=[1.0] * len(all_retrievers)  # Equal weights, cannot be customized by priority
        )
        
        print("\n🔍 检索阶段（使用LangChain的EnsembleRetriever）：")
        docs = ensemble_retriever.get_relevant_documents(query)
        
        # Analyze distribution by priority
        stats = {
            "P1": {"count": 0, "sources": set()},
            "P2": {"count": 0, "sources": set()},
            "P3": {"count": 0, "sources": set()},
        }
        
        for doc in docs:
            priority = doc.metadata.get("priority", 0)
            source_name = doc.metadata.get("source_name", "Unknown")
            if priority in [1, 2, 3]:
                stats[f"P{priority}"]["count"] += 1
                stats[f"P{priority}"]["sources"].add(source_name)
        
        # Convert sets to lists for stats
        for p in ["P1", "P2", "P3"]:
            stats[p]["sources"] = list(stats[p]["sources"])
        
        print(f"  ✅ 融合后返回: {len(docs)} chunks (LangChain自动去重和重排序)")
        print(f"  📊 数据源分布:")
        print(f"     - P1: {stats['P1']['count']} chunks")
        print(f"     - P2: {stats['P2']['count']} chunks")
        print(f"     - P3: {stats['P3']['count']} chunks")
        
        return docs, stats
    
    def retrieve_strategy_priority_filter(
        self,
        query: str,
        threshold: int = 8
    ) -> Tuple[List[Document], Dict]:
        """
        Strategy 4: Priority Filtering (Custom Implementation)
        
        Query high-priority sources first, only query lower priorities if insufficient results.
        
        ⚠️ LangChain Limitation #2: Standard RAG chains do not support conditional branching logic.
        This "query A first, then B if needed" pattern requires complete custom implementation.
        Cannot use LangChain's Chain abstraction, must manually write control flow.
        
        Args:
            query: Query string
            threshold: Minimum number of documents required
            
        Returns:
            Tuple of (documents list, statistics dict)
        """
        print("\n" + "─"*60)
        print("📊 策略4: 优先级过滤 (Priority Filtering)")
        print("─"*60)
        print("⚠️  LangChain标准RAG链不支持条件分支逻辑")
        print(f"   需要完全自定义实现（阈值={threshold}）\n")
        
        all_docs = []
        stats = {
            "stages": [],
            "P1": {"count": 0, "sources": []},
            "P2": {"count": 0, "sources": []},
            "P3": {"count": 0, "sources": []},
        }
        
        # ⚠️ LangChain Limitation #2: Need to manually implement conditional query logic
        # Cannot use standard Chain abstraction for this pattern
        
        # Stage 1: Query P1 (highest priority)
        print("🔍 阶段1: 查询P1 (技术白皮书)...")
        p1_docs = []
        for retriever in self.retrievers_by_priority[1]:
            docs = retriever.get_relevant_documents(query)
            p1_docs.extend(docs)
        
        p1_sources = set(doc.metadata.get("source_name") for doc in p1_docs if "source_name" in doc.metadata)
        stats["P1"]["count"] = len(p1_docs)
        stats["P1"]["sources"] = list(p1_sources)
        
        print(f"   → 检索到 {len(p1_docs)} chunks")
        
        if len(p1_docs) >= threshold:
            print(f"   ✅ 超过阈值 ({threshold})，使用P1结果")
            stats["stages"].append("P1 only (sufficient)")
            all_docs = p1_docs
            print(f"   ⏭️  跳过P2和P3查询（P1结果已足够）")
        else:
            print(f"   ⚠️  低于阈值 ({threshold})，继续查询P2...")
            
            # Stage 2: Query P2 (medium priority)
            print("\n🔍 阶段2: 查询P2 (用户手册)...")
            p2_docs = []
            for retriever in self.retrievers_by_priority[2]:
                docs = retriever.get_relevant_documents(query)
                p2_docs.extend(docs)
            
            p2_sources = set(doc.metadata.get("source_name") for doc in p2_docs if "source_name" in doc.metadata)
            stats["P2"]["count"] = len(p2_docs)
            stats["P2"]["sources"] = list(p2_sources)
            
            print(f"   → 检索到 {len(p2_docs)} chunks")
            
            if len(p1_docs) + len(p2_docs) >= threshold:
                print(f"   ✅ P1+P2 超过阈值 ({threshold})，使用P1+P2结果")
                stats["stages"].append("P1+P2 (sufficient)")
                all_docs = p1_docs + p2_docs
                print(f"   ⏭️  跳过P3查询")
            else:
                print(f"   ⚠️  仍低于阈值 ({threshold})，查询P3...")
                
                # Stage 3: Query P3 (lowest priority)
                print("\n🔍 阶段3: 查询P3 (官网介绍)...")
                p3_docs = []
                for retriever in self.retrievers_by_priority[3]:
                    docs = retriever.get_relevant_documents(query)
                    p3_docs.extend(docs)
                
                p3_sources = set(doc.metadata.get("source_name") for doc in p3_docs if "source_name" in doc.metadata)
                stats["P3"]["count"] = len(p3_docs)
                stats["P3"]["sources"] = list(p3_sources)
                
                print(f"   → 检索到 {len(p3_docs)} chunks")
                print(f"   ✅ 使用全部结果 (P1+P2+P3)")
                stats["stages"].append("P1+P2+P3 (all sources)")
                all_docs = p1_docs + p2_docs + p3_docs
        
        print(f"\n📦 最终使用: {len(all_docs)} chunks")
        
        return all_docs, stats
    
    @staticmethod
    def format_docs_with_source(docs: List[Document]) -> str:
        """
        Format documents with source information for the prompt.
        
        Args:
            docs: List of documents to format
            
        Returns:
            Formatted string with source annotations
        """
        # Group by priority
        docs_by_priority = {1: [], 2: [], 3: []}
        for doc in docs:
            priority = doc.metadata.get("priority", 3)
            if priority in docs_by_priority:
                docs_by_priority[priority].append(doc)
        
        formatted_parts = []
        
        priority_labels = {
            1: "P1: 技术白皮书 (最高优先级)",
            2: "P2: 用户手册 (中优先级)",
            3: "P3: 官网介绍 (低优先级)"
        }
        
        for priority in [1, 2, 3]:
            priority_docs = docs_by_priority[priority]
            if not priority_docs:
                continue
            
            formatted_parts.append(f"\n{'='*60}")
            formatted_parts.append(f"{priority_labels[priority]}")
            formatted_parts.append(f"{'='*60}\n")
            
            for i, doc in enumerate(priority_docs, 1):
                source_name = doc.metadata.get("source_name", "Unknown")
                product = doc.metadata.get("product", "")
                page = doc.metadata.get("page", "")
                
                header = f"[{source_name}"
                if page:
                    header += f", Page {page + 1}"
                header += "]"
                
                formatted_parts.append(header)
                formatted_parts.append(doc.page_content)
                formatted_parts.append("")  # Empty line
        
        return "\n".join(formatted_parts)
    
    def setup_rag_chains(self):
        """
        Setup RAG chains for all 3 strategies.
        """
        # Common prompt template
        template = """你是一个专业的无人机产品对比分析专家，擅长比较EVO Nest和DJI Dock的技术方案。

请根据以下产品文档来回答问题。文档按优先级排序：
- P1 (技术白皮书): 最权威的技术规格信息
- P2 (用户手册): 详细的操作和配置说明  
- P3 (官网介绍): 产品概述和营销信息

产品文档：
{context}

问题：{question}

请提供结构化的对比分析，包括：
1. EVO Nest的方案特点（引用具体数据）
2. DJI Dock的方案特点（引用具体数据）
3. 两者的关键差异总结
4. 标注引用来源（如：[EVO Nest技术白皮书, Page X]）

回答："""
        
        prompt = ChatPromptTemplate.from_template(template)
        output_parser = StrOutputParser()
        
        # Note: We cannot create RAG chains here because they need query-specific context
        # Store the prompt and parser for later use
        self.prompt = prompt
        self.output_parser = output_parser
        
        print("\n✅ RAG chains configured (prompt template and output parser ready)")
    
    def compare_strategies(self, question: str) -> Dict:
        """
        Compare all 3 strategies on the same question.
        
        Args:
            question: Question to ask
            
        Returns:
            Dictionary with results from all strategies
        """
        print("\n" + "="*60)
        print("🎯 多源RAG策略对比实验")
        print("="*60)
        print(f"\n问题：{question}\n")
        
        results = {}
        
        # Strategy 1: Simple Concatenation
        print("\n" + "="*60)
        print("策略1: 简单拼接")
        print("="*60)
        start_time = time.time()
        docs1, stats1 = self.retrieve_strategy_simple_concat(question)
        context1 = self.format_docs_with_source(docs1)
        
        print("\n💬 生成答案中...")
        answer1 = (self.prompt | self.llm | self.output_parser).invoke({
            "context": context1,
            "question": question
        })
        elapsed1 = time.time() - start_time
        
        results["strategy1"] = {
            "name": "简单拼接",
            "docs": docs1,
            "stats": stats1,
            "answer": answer1,
            "elapsed": elapsed1,
        }
        
        print(f"\n⏱️  耗时: {elapsed1:.1f}秒")
        
        # Strategy 3: RRF Fusion
        print("\n" + "="*60)
        print("策略3: RRF融合")
        print("="*60)
        start_time = time.time()
        docs3, stats3 = self.retrieve_strategy_rrf(question)
        context3 = self.format_docs_with_source(docs3)
        
        print("\n💬 生成答案中...")
        answer3 = (self.prompt | self.llm | self.output_parser).invoke({
            "context": context3,
            "question": question
        })
        elapsed3 = time.time() - start_time
        
        results["strategy3"] = {
            "name": "RRF融合",
            "docs": docs3,
            "stats": stats3,
            "answer": answer3,
            "elapsed": elapsed3,
        }
        
        print(f"\n⏱️  耗时: {elapsed3:.1f}秒")
        
        # Strategy 4: Priority Filtering
        print("\n" + "="*60)
        print("策略4: 优先级过滤")
        print("="*60)
        start_time = time.time()
        docs4, stats4 = self.retrieve_strategy_priority_filter(question, threshold=8)
        context4 = self.format_docs_with_source(docs4)
        
        print("\n💬 生成答案中...")
        answer4 = (self.prompt | self.llm | self.output_parser).invoke({
            "context": context4,
            "question": question
        })
        elapsed4 = time.time() - start_time
        
        results["strategy4"] = {
            "name": "优先级过滤",
            "docs": docs4,
            "stats": stats4,
            "answer": answer4,
            "elapsed": elapsed4,
        }
        
        print(f"\n⏱️  耗时: {elapsed4:.1f}秒")
        
        return results
    
    def print_comparison_report(self, results: Dict):
        """
        Print a comprehensive comparison report.
        
        Args:
            results: Results from compare_strategies()
        """
        print("\n" + "="*60)
        print("📈 策略对比总结")
        print("="*60)
        
        # Print answers
        for strategy_key in ["strategy1", "strategy3", "strategy4"]:
            result = results[strategy_key]
            print(f"\n{'─'*60}")
            print(f"【{result['name']}】的回答：")
            print(f"{'─'*60}")
            print(result["answer"])
            print(f"\n⏱️  生成耗时: {result['elapsed']:.1f}秒")
            print(f"📦 使用chunks数: {len(result['docs'])}个")
        
        # Comparison table
        print("\n" + "="*60)
        print("📊 策略对比指标")
        print("="*60)
        
        print(f"\n{'指标':<20} {'策略1':<15} {'策略3':<15} {'策略4':<15}")
        print("─" * 70)
        
        # Chunks count
        print(f"{'检索chunks数':<20} "
              f"{len(results['strategy1']['docs']):<15} "
              f"{len(results['strategy3']['docs']):<15} "
              f"{len(results['strategy4']['docs']):<15}")
        
        # P1 ratio
        def calc_p1_ratio(docs):
            p1_count = len([d for d in docs if d.metadata.get("priority") == 1])
            return f"{p1_count}/{len(docs)} ({100*p1_count/len(docs):.0f}%)" if docs else "0/0"
        
        print(f"{'高优先级chunks':<20} "
              f"{calc_p1_ratio(results['strategy1']['docs']):<15} "
              f"{calc_p1_ratio(results['strategy3']['docs']):<15} "
              f"{calc_p1_ratio(results['strategy4']['docs']):<15}")
        
        # Time
        print(f"{'生成耗时(秒)':<20} "
              f"{results['strategy1']['elapsed']:<15.1f} "
              f"{results['strategy3']['elapsed']:<15.1f} "
              f"{results['strategy4']['elapsed']:<15.1f}")
        
        # LangChain Limitations Summary
        print("\n" + "="*60)
        print("⚠️  LangChain在多源RAG场景下的局限性")
        print("="*60)
        
        limitations = [
            "1. ❌ 需要手动管理多个检索器，缺乏统一的管理接口",
            "   - 必须分别创建6个vector store和6个retriever",
            "   - 没有DataSourceManager或类似的统一抽象",
            "",
            "2. ❌ EnsembleRetriever无法配置数据源优先级权重",
            "   - 所有数据源被平等对待",
            "   - 无法体现\"技术白皮书>用户手册>官网\"的质量差异",
            "",
            "3. ❌ 无法实现条件查询逻辑（如\"先查A，不够再查B\"）",
            "   - 标准RAG Chain不支持条件分支",
            "   - 策略4需要完全自定义实现100+行代码",
            "   - 无法使用LangChain的Chain抽象",
            "",
            "4. ❌ 缺乏查询路由功能",
            "   - 无法根据问题类型自动选择数据源",
            "   - 不能实现\"技术问题→白皮书，操作问题→手册\"",
            "",
            "5. ❌ 多检索器的日志输出混乱，难以追踪",
            "   - 需要手动添加大量print语句才能看清过程",
            "   - 没有内置的可观测性工具",
        ]
        
        for limitation in limitations:
            print(limitation)
        
        print("\n" + "="*60)
        print("💡 这些问题正是LangGraph要解决的！")
        print("="*60)
        print("\nLangGraph提供：")
        print("  ✅ 图结构的工作流编排（支持条件分支、循环）")
        print("  ✅ 状态管理和条件路由")
        print("  ✅ 多Agent协作能力")
        print("  ✅ 内置的可观测性和调试工具")
        print("\n下一步可以探索使用LangGraph重构多源RAG系统。")
        print("="*60)
        
    def _initialize_embeddings(self, use_fastembed: bool):
        """Initialize embedding model (FastEmbed or OpenAI)."""
        if use_fastembed and FASTEMBED_AVAILABLE:
            print("🔧 Initializing FastEmbed embeddings (free, local)...")
            return FastEmbedEmbeddings()
        elif OPENAI_EMBEDDINGS_AVAILABLE and self.openai_api_key:
            print("🔧 Initializing OpenAI embeddings...")
            return OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        else:
            raise ValueError(
                "❌ No embedding model available.\n"
                "   Option 1: Install FastEmbed: pip install fastembed\n"
                "   Option 2: Set OPENAI_API_KEY in .env file"
            )
    
    def _initialize_llm(self):
        """Initialize DeepSeek LLM via OpenAI-compatible API."""
        print("🔧 Initializing DeepSeek Reasoner model...")
        return ChatOpenAI(
            model="deepseek-reasoner",
            openai_api_key=self.deepseek_api_key,
            openai_api_base="https://api.deepseek.com/v1",
            temperature=0,
            max_tokens=4000,
        )


def main():
    """
    Main function to run the multi-source RAG demonstration.
    """
    print("="*60)
    print("🤖 Multi-Source RAG Agent (LangChain Educational Demo)")
    print("="*60)
    print("\n⚠️  This demo is designed to showcase LangChain's limitations")
    print("   in multi-source RAG scenarios.\n")
    
    try:
        # Initialize agent
        print("🔧 Initializing Multi-Source RAG Agent...")
        agent = MultiSourceRAGAgent(
            vector_db_path="./tmp/lancedb",
            use_fastembed=True,
        )
        
        # Load all data sources
        print("\n📚 Step 1: Loading data sources...")
        all_documents = agent.load_all_data_sources(force_reload=False)
        
        # Check if any documents were loaded
        total_docs = sum(len(docs) for docs in all_documents.values())
        if total_docs == 0:
            print("\n❌ No documents loaded. Please check data source files.")
            return
        
        # Create vector stores
        print("\n🗄️  Step 2: Creating vector stores...")
        agent.create_vector_stores(all_documents, force_reload=False)
        
        # Create retrievers
        print("\n🔍 Step 3: Creating retrievers...")
        agent.create_retrievers(k=5)
        
        # Setup RAG chains
        print("\n⚙️  Step 4: Setting up RAG chains...")
        agent.setup_rag_chains()
        
        # Run comparison
        print("\n🚀 Step 5: Running strategy comparison...")
        question = "EVO Nest机巢的数据存储方案与DJI Dock机场有什么区别？"
        
        results = agent.compare_strategies(question)
        
        # Print comparison report
        agent.print_comparison_report(results)
        
        print("\n" + "="*60)
        print("✨ 演示完成！")
        print("="*60)
        print("\n💡 下一步建议:")
        print("   1. 查看输出中的LangChain局限性标注")
        print("   2. 对比3种策略的答案质量差异")
        print("   3. 探索使用LangGraph重构多源RAG系统\n")
        
    except FileNotFoundError as e:
        print(f"\n❌ 文件未找到错误: {e}")
        print("   请确保以下文件存在于项目根目录:")
        for source in MultiSourceRAGAgent.DATA_SOURCES:
            print(f"   - {source['file_path']}")
        print()
        
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        print()


if __name__ == "__main__":
    main()

