"""
LangGraph Agentic RAG Agent (Stage 3 - Educational Demo)

This application demonstrates LangGraph's intelligent routing capabilities for multi-source
RAG scenarios by implementing an agentic system that can:

1. Intelligently classify questions and route to appropriate data sources
2. Retrieve documents in parallel from multiple vector stores
3. Automatically supplement queries if initial results are insufficient
4. Generate answers with clear source citations

Purpose: Compare EVO Nest and DJI Dock using intelligent routing across 6 data sources.

LangGraph Features Demonstrated:
1. Conditional Edges: Route based on question type and information sufficiency
2. Parallel Nodes: Concurrent retrieval from multiple data sources
3. State Management: Track query history, sources used, and iteration count
4. Cycles: Automatic supplementary queries when information is insufficient

Key Improvements over Stage 2 (LangChain):
- ✅ Automatic question classification and routing
- ✅ Intelligent source selection based on question type
- ✅ Parallel retrieval with built-in orchestration
- ✅ Automatic information sufficiency checking
- ✅ Multi-level supplementary query strategy
- ✅ Clear visualization of decision flow
"""

import os
import time
from pathlib import Path
from typing import List, Dict, Optional, TypedDict, Annotated
from concurrent.futures import ThreadPoolExecutor, as_completed
import getpass

from dotenv import load_dotenv
import lancedb

from langchain_community.vectorstores import LanceDB
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# LangGraph imports
from langgraph.graph import StateGraph, END, START

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


# ============================================================================
# GraphState Definition (Work Item 7)
# ============================================================================

class GraphState(TypedDict):
    """
    State for the LangGraph agentic RAG workflow.
    
    This state is passed between all nodes and tracks the entire query lifecycle.
    """
    question: str                          # Original user question
    question_type: str                     # Classified type: basic_info/technical/comparison
    target_sources: List[str]              # List of target data source IDs to query
    chunks: List[Document]                 # Retrieved document chunks
    sources_used: Dict[str, int]           # Statistics: {source_name: chunk_count}
    iteration: int                         # Current iteration count (for supplementary queries)
    is_sufficient: bool                    # Whether information is sufficient
    answer: str                            # Final generated answer
    route_decision: str                    # Routing decision log (for visualization)


# ============================================================================
# Data Source Configuration (Work Item 8)
# ============================================================================

DATA_SOURCES = [
    {
        "id": "evo_nest_whitepaper",
        "name": "EVO Nest技术白皮书",
        "priority": 1,  # Highest priority
        "product": "evo_nest",
        "source_type": "whitepaper",
    },
    {
        "id": "dji_dock_whitepaper",
        "name": "DJI Dock技术白皮书",
        "priority": 1,  # Highest priority
        "product": "dji_dock",
        "source_type": "whitepaper",
    },
    {
        "id": "evo_nest_manual",
        "name": "EVO Nest用户手册",
        "priority": 2,  # Medium priority
        "product": "evo_nest",
        "source_type": "manual",
    },
    {
        "id": "dji_dock_manual",
        "name": "DJI Dock用户手册",
        "priority": 2,  # Medium priority
        "product": "dji_dock",
        "source_type": "manual",
    },
    {
        "id": "evo_nest_webpage",
        "name": "EVO Nest官网介绍",
        "priority": 3,  # Lowest priority
        "product": "evo_nest",
        "source_type": "webpage",
    },
    {
        "id": "dji_dock_webpage",
        "name": "DJI Dock官网介绍",
        "priority": 3,  # Lowest priority
        "product": "dji_dock",
        "source_type": "webpage",
    },
]


# ============================================================================
# Prompt Templates
# ============================================================================

CLASSIFICATION_PROMPT = """你是一个智能问题分类器。请分析以下问题，判断其类型。

问题类型定义：
1. basic_info (基础信息查询): 询问产品型号、支持设备、基本参数、操作范围等基础信息
2. technical (技术细节查询): 询问技术规格、传输速度、存储容量、数据处理等具体技术参数
3. comparison (对比分析查询): 对比两个产品的优劣、差异、特点、优势等

问题: {question}

请只返回以下之一: basic_info, technical, comparison

分类结果:"""

ANSWER_GENERATION_PROMPT = """你是一个专业的无人机产品对比分析专家，擅长比较EVO Nest和DJI Dock的技术方案。

**重要要求**: 你必须在答案中使用内联标注格式，标注每个关键信息的来源。

标注格式示例：
- 传输速度为150-200 MB/s [来源: EVO Nest技术白皮书, Page 3]
- 支持EVO II系列无人机 [来源: EVO Nest官网]
- 机巢重量为154磅 [来源: EVO Nest用户手册, Page 12]

请根据以下产品文档来回答问题：

{context}

问题：{question}

请提供详细的对比分析，并确保每个关键信息都使用内联标注格式标注来源。

回答："""


# ============================================================================
# AgenticRAGAgent Class (Work Items 9-30)
# ============================================================================

class AgenticRAGAgent:
    """
    Agentic RAG Agent using LangGraph for intelligent routing.
    
    This agent demonstrates how LangGraph solves the limitations of LangChain
    in multi-source RAG scenarios through:
    - Intelligent question classification
    - Automatic data source routing
    - Parallel retrieval
    - Iterative supplementary queries
    - Clear source citation
    """
    
    def __init__(
        self,
        vector_db_path: str = "./tmp/lancedb",
        use_fastembed: bool = True,
        deepseek_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
    ):
        """
        Initialize the Agentic RAG Agent.
        
        Args:
            vector_db_path: Path to existing LanceDB vector databases
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
        
        # Initialize dual LLMs (Work Item 11)
        self.classifier_llm, self.generator_llm = self._initialize_llms()
        
        # Storage for vector stores and retrievers
        self.vector_stores: Dict[str, LanceDB] = {}
        self.retrievers: Dict[str, any] = {}
        
        # Compiled LangGraph
        self.compiled_graph = None
    
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
    
    def _initialize_llms(self):
        """
        Initialize dual LLM strategy (Work Item 11).
        
        Returns:
            Tuple of (classifier_llm, generator_llm)
        """
        print("🔧 Initializing dual LLM strategy...")
        
        # Classifier LLM: Fast and cheap for question classification
        classifier_llm = ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=self.deepseek_api_key,
            openai_api_base="https://api.deepseek.com/v1",
            temperature=0,
            max_tokens=100,
        )
        print("   ✅ Classifier LLM: deepseek-chat (for fast classification)")
        
        # Generator LLM: High-quality reasoning for answer generation
        generator_llm = ChatOpenAI(
            model="deepseek-reasoner",
            openai_api_key=self.deepseek_api_key,
            openai_api_base="https://api.deepseek.com/v1",
            temperature=0,
            max_tokens=4000,
        )
        print("   ✅ Generator LLM: deepseek-reasoner (for high-quality answers)")
        
        return classifier_llm, generator_llm
    
    def load_vector_stores(self) -> Dict[str, LanceDB]:
        """
        Load existing vector stores from LanceDB (Work Item 12).
        
        Reuses vector stores created in Stage 2 to save time and computation.
        
        Returns:
            Dictionary mapping source_id to LanceDB vector store
        """
        print("\n" + "="*60)
        print("🗄️  Loading Existing Vector Stores")
        print("="*60)
        
        db = lancedb.connect(self.vector_db_path)
        vector_stores = {}
        
        for source_config in DATA_SOURCES:
            source_id = source_config["id"]
            table_name = source_id
            
            print(f"\n[P{source_config['priority']}] {source_config['name']}")
            print(f"  📊 Table: {table_name}")
            
            # Check if table exists
            if table_name not in db.table_names():
                print(f"  ⚠️  Table '{table_name}' not found!")
                print(f"  💡 Tip: Run uav-nest-agent-multi-source.py first to create vector stores")
                continue
            
            try:
                table = db.open_table(table_name)
                row_count = table.count_rows()
                
                if row_count == 0:
                    print(f"  ⚠️  Table '{table_name}' is empty, skipping...")
                    continue
                
                print(f"  📂 Loading existing table ({row_count} rows)")
                vector_store = LanceDB(
                    connection=db,
                    embedding=self.embeddings,
                    table_name=table_name,
                )
                vector_stores[source_id] = vector_store
                
                # Create retriever
                retriever = vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 5}
                )
                self.retrievers[source_id] = retriever
                print(f"  ✅ Loaded successfully")
                
            except Exception as e:
                print(f"  ❌ Error loading table: {e}")
        
        print("\n" + "="*60)
        print(f"✅ Successfully loaded {len(vector_stores)} vector stores")
        print("="*60)
        
        self.vector_stores = vector_stores
        return vector_stores
    
    # ========================================================================
    # LangGraph Node: classify_question (Work Item 13)
    # ========================================================================
    
    def classify_question(self, state: GraphState) -> GraphState:
        """
        Node: Classify question and determine target data sources.
        
        Uses LLM to intelligently classify the question type and routes to
        appropriate data sources based on predefined rules.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with question_type and target_sources
        """
        print("\n" + "="*60)
        print("🧠 Node: classify_question")
        print("="*60)
        
        question = state["question"]
        print(f"Question: {question}")
        
        # Use classifier LLM to classify question type
        prompt = ChatPromptTemplate.from_template(CLASSIFICATION_PROMPT)
        chain = prompt | self.classifier_llm | StrOutputParser()
        
        print("\n🔍 Classifying question type using LLM...")
        classification_result = chain.invoke({"question": question}).strip().lower()
        
        # Parse classification result
        if "basic_info" in classification_result:
            question_type = "basic_info"
        elif "technical" in classification_result:
            question_type = "technical"
        elif "comparison" in classification_result:
            question_type = "comparison"
        else:
            # Default to comparison if unclear
            question_type = "comparison"
            print(f"⚠️  Unclear classification result: '{classification_result}', defaulting to 'comparison'")
        
        print(f"✅ Question classified as: {question_type}")
        
        # Route to target data sources based on question type
        if question_type == "basic_info":
            # Basic info → Query P3 (official webpages)
            target_sources = ["evo_nest_webpage", "dji_dock_webpage"]
            route_decision = "基础信息查询 → P3 (官网)"
        elif question_type == "technical":
            # Technical details → Query P1 (technical whitepapers)
            target_sources = ["evo_nest_whitepaper", "dji_dock_whitepaper"]
            route_decision = "技术细节查询 → P1 (白皮书)"
        else:  # comparison
            # Comparison analysis → Query P2+P3 (manuals + webpages)
            target_sources = [
                "evo_nest_manual", "dji_dock_manual",
                "evo_nest_webpage", "dji_dock_webpage"
            ]
            route_decision = "对比分析查询 → P2+P3 (手册+官网)"
        
        print(f"📍 Route Decision: {route_decision}")
        print(f"🎯 Target Sources: {len(target_sources)} sources")
        for src in target_sources:
            src_config = next(s for s in DATA_SOURCES if s["id"] == src)
            print(f"   - {src_config['name']} (P{src_config['priority']})")
        
        return {
            **state,
            "question_type": question_type,
            "target_sources": target_sources,
            "route_decision": route_decision,
            "iteration": 0,
        }
    
    # ========================================================================
    # LangGraph Node: retrieve_parallel (Work Item 14)
    # ========================================================================
    
    def retrieve_parallel(self, state: GraphState) -> GraphState:
        """
        Node: Retrieve documents in parallel from multiple sources.
        
        Uses ThreadPoolExecutor to query multiple vector stores concurrently,
        significantly improving performance compared to sequential queries.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with chunks and sources_used
        """
        print("\n" + "="*60)
        print("🔍 Node: retrieve_parallel")
        print("="*60)
        
        question = state["question"]
        target_sources = state["target_sources"]
        iteration = state.get("iteration", 0)
        
        print(f"Iteration: {iteration + 1}")
        print(f"Querying {len(target_sources)} sources in parallel...")
        
        chunks = []
        sources_used = {}
        
        # Use ThreadPoolExecutor for parallel retrieval
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all retrieval tasks
            futures = {}
            for source_id in target_sources:
                if source_id not in self.retrievers:
                    print(f"  ⚠️  Retriever for '{source_id}' not found, skipping...")
                    continue
                
                retriever = self.retrievers[source_id]
                future = executor.submit(
                    retriever.get_relevant_documents,
                    question
                )
                futures[future] = source_id
            
            # Collect results as they complete
            for future in as_completed(futures):
                source_id = futures[future]
                try:
                    docs = future.result()
                    chunks.extend(docs)
                    
                    # Get source name for display
                    source_config = next(s for s in DATA_SOURCES if s["id"] == source_id)
                    source_name = source_config["name"]
                    sources_used[source_name] = len(docs)
                    
                    print(f"  ✅ {source_name}: {len(docs)} chunks")
                except Exception as e:
                    print(f"  ❌ Error retrieving from {source_id}: {e}")
        
        total_chunks = len(chunks)
        print(f"\n📦 Total chunks retrieved: {total_chunks}")
        
        return {
            **state,
            "chunks": chunks,
            "sources_used": sources_used,
            "iteration": iteration + 1,
        }
    
    # ========================================================================
    # LangGraph Node: check_sufficiency (Work Item 15)
    # ========================================================================
    
    def check_sufficiency(self, state: GraphState) -> GraphState:
        """
        Node: Check if retrieved information is sufficient.
        
        Uses a simple rule-based approach: if chunks >= 3, information is sufficient.
        This threshold can be adjusted based on domain requirements.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with is_sufficient flag
        """
        print("\n" + "="*60)
        print("✅ Node: check_sufficiency")
        print("="*60)
        
        chunks = state["chunks"]
        iteration = state["iteration"]
        chunk_count = len(chunks)
        
        # Threshold for information sufficiency
        THRESHOLD = 3
        
        print(f"Chunks retrieved: {chunk_count}")
        print(f"Threshold: {THRESHOLD}")
        print(f"Current iteration: {iteration}")
        
        if chunk_count >= THRESHOLD:
            is_sufficient = True
            print(f"✅ Information is sufficient ({chunk_count} >= {THRESHOLD})")
        else:
            is_sufficient = False
            print(f"⚠️  Information is insufficient ({chunk_count} < {THRESHOLD})")
            if iteration < 3:
                print(f"   → Will expand sources and retry")
            else:
                print(f"   → Maximum iterations reached (3), will proceed with available data")
        
        return {
            **state,
            "is_sufficient": is_sufficient,
        }
    
    # ========================================================================
    # LangGraph Node: expand_sources (Work Item 16)
    # ========================================================================
    
    def expand_sources(self, state: GraphState) -> GraphState:
        """
        Node: Expand target data sources for supplementary query.
        
        Implements a multi-level expansion strategy:
        - Iteration 1: Add adjacent priority sources
        - Iteration 2+: Add all remaining sources
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with expanded target_sources
        """
        print("\n" + "="*60)
        print("📈 Node: expand_sources")
        print("="*60)
        
        question_type = state["question_type"]
        current_sources = set(state["target_sources"])
        iteration = state["iteration"]
        
        print(f"Current iteration: {iteration}")
        print(f"Question type: {question_type}")
        print(f"Current sources: {len(current_sources)}")
        
        # Expansion strategy based on iteration
        if iteration == 1:
            # First expansion: Add adjacent priority sources
            if question_type == "basic_info":
                # P3 → add P2 (manuals)
                additional = ["evo_nest_manual", "dji_dock_manual"]
                strategy = "P3 → P3+P2 (添加手册)"
            elif question_type == "technical":
                # P1 → add P2 (manuals)
                additional = ["evo_nest_manual", "dji_dock_manual"]
                strategy = "P1 → P1+P2 (添加手册)"
            else:  # comparison (already has P2+P3)
                # P2+P3 → add P1 (whitepapers)
                additional = ["evo_nest_whitepaper", "dji_dock_whitepaper"]
                strategy = "P2+P3 → P1+P2+P3 (添加白皮书)"
        else:
            # Second+ expansion: Add all remaining sources
            all_source_ids = [s["id"] for s in DATA_SOURCES]
            additional = [sid for sid in all_source_ids if sid not in current_sources]
            strategy = "添加所有剩余数据源"
        
        # Add new sources
        expanded_sources = list(current_sources) + [s for s in additional if s not in current_sources]
        
        print(f"📍 Expansion Strategy: {strategy}")
        print(f"➕ Adding {len(additional)} sources:")
        for src_id in additional:
            if src_id not in current_sources:
                src_config = next(s for s in DATA_SOURCES if s["id"] == src_id)
                print(f"   + {src_config['name']} (P{src_config['priority']})")
        print(f"🎯 Total sources after expansion: {len(expanded_sources)}")
        
        return {
            **state,
            "target_sources": expanded_sources,
        }
    
    # ========================================================================
    # LangGraph Node: generate_answer (Work Item 17)
    # ========================================================================
    
    def generate_answer(self, state: GraphState) -> GraphState:
        """
        Node: Generate final answer with source citations.
        
        Uses the high-quality reasoning model (deepseek-reasoner) to generate
        a comprehensive answer with inline source citations.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with generated answer
        """
        print("\n" + "="*60)
        print("💬 Node: generate_answer")
        print("="*60)
        
        question = state["question"]
        chunks = state["chunks"]
        
        print(f"Generating answer using {len(chunks)} chunks...")
        print("Using deepseek-reasoner for high-quality answer generation...")
        
        # Format chunks with source information
        context = self.format_docs_with_source(chunks)
        
        # Generate answer using generator LLM
        prompt = ChatPromptTemplate.from_template(ANSWER_GENERATION_PROMPT)
        chain = prompt | self.generator_llm | StrOutputParser()
        
        answer = chain.invoke({
            "context": context,
            "question": question
        })
        
        print("✅ Answer generated successfully")
        
        return {
            **state,
            "answer": answer,
        }
    
    # ========================================================================
    # Helper Method: format_docs_with_source (Work Item 18)
    # ========================================================================
    
    @staticmethod
    def format_docs_with_source(chunks: List[Document]) -> str:
        """
        Format document chunks with source information.
        
        Groups chunks by data source and adds metadata for better context.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            Formatted string with source annotations
        """
        if not chunks:
            return "No relevant documents found."
        
        # Group chunks by source
        chunks_by_source = {}
        for chunk in chunks:
            source_name = chunk.metadata.get("source_name", "Unknown Source")
            if source_name not in chunks_by_source:
                chunks_by_source[source_name] = []
            chunks_by_source[source_name].append(chunk)
        
        # Format output
        formatted_parts = []
        for source_name, source_chunks in chunks_by_source.items():
            formatted_parts.append(f"\n{'='*60}")
            formatted_parts.append(f"【{source_name}】")
            formatted_parts.append(f"{'='*60}\n")
            
            for i, chunk in enumerate(source_chunks, 1):
                page = chunk.metadata.get("page", "")
                page_info = f", Page {page + 1}" if page != "" else ""
                
                formatted_parts.append(f"[Chunk {i}{page_info}]")
                formatted_parts.append(chunk.page_content)
                formatted_parts.append("")  # Empty line
        
        return "\n".join(formatted_parts)
    
    # ========================================================================
    # Conditional Edge Function: route_after_check (Work Item 19)
    # ========================================================================
    
    def route_after_check(self, state: GraphState) -> str:
        """
        Conditional edge: Decide next node after sufficiency check.
        
        Decision logic:
        - If sufficient: Go to generate_answer
        - If insufficient and iteration < 3: Go to expand_sources
        - If insufficient and iteration >= 3: Go to generate_answer (force)
        
        Args:
            state: Current graph state
            
        Returns:
            Next node name as string
        """
        is_sufficient = state["is_sufficient"]
        iteration = state["iteration"]
        
        if is_sufficient:
            return "generate_answer"
        elif iteration < 3:
            return "expand_sources"
        else:
            # Maximum iterations reached, force generation
            return "generate_answer"
    
    # ========================================================================
    # Build LangGraph (Work Item 20)
    # ========================================================================
    
    def build_graph(self):
        """
        Build and compile the LangGraph workflow.
        
        Graph structure:
        START → classify_question → retrieve_parallel → check_sufficiency
                                                              ↓
                                                    (conditional edge)
                                                       ↓            ↓
                                              expand_sources    generate_answer
                                                       ↓                ↓
                                              retrieve_parallel      END
        
        Returns:
            Compiled LangGraph
        """
        print("\n" + "="*60)
        print("🔧 Building LangGraph Workflow")
        print("="*60)
        
        # Create StateGraph
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("classify_question", self.classify_question)
        workflow.add_node("retrieve_parallel", self.retrieve_parallel)
        workflow.add_node("check_sufficiency", self.check_sufficiency)
        workflow.add_node("expand_sources", self.expand_sources)
        workflow.add_node("generate_answer", self.generate_answer)
        
        # Add edges
        workflow.add_edge(START, "classify_question")
        workflow.add_edge("classify_question", "retrieve_parallel")
        workflow.add_edge("retrieve_parallel", "check_sufficiency")
        
        # Add conditional edge from check_sufficiency
        workflow.add_conditional_edges(
            "check_sufficiency",
            self.route_after_check,
            {
                "generate_answer": "generate_answer",
                "expand_sources": "expand_sources",
            }
        )
        
        # Add edge from expand_sources back to retrieve_parallel (cycle)
        workflow.add_edge("expand_sources", "retrieve_parallel")
        
        # Add edge from generate_answer to END
        workflow.add_edge("generate_answer", END)
        
        # Compile graph
        compiled = workflow.compile()
        
        print("✅ LangGraph compiled successfully")
        print("\nGraph structure:")
        print("  START → classify_question → retrieve_parallel → check_sufficiency")
        print("                                                         |")
        print("                                                  (conditional)")
        print("                                                    /        \\")
        print("                                          expand_sources  generate_answer")
        print("                                                |              |")
        print("                                       retrieve_parallel      END")
        
        self.compiled_graph = compiled
        return compiled
    
    # ========================================================================
    # Visualization: Generate Mermaid Diagram (Work Item 24)
    # ========================================================================
    
    def visualize_graph(self, output_file: str = "agentic-rag-workflow.mmd"):
        """
        Generate Mermaid diagram for workflow visualization.
        
        Args:
            output_file: Output file path for Mermaid diagram
        """
        print("\n" + "="*60)
        print("📊 Generating Workflow Visualization")
        print("="*60)
        
        if not self.compiled_graph:
            print("⚠️  Graph not compiled yet. Call build_graph() first.")
            return
        
        try:
            # Generate Mermaid diagram
            mermaid_code = self.compiled_graph.get_graph().draw_mermaid()
            
            # Save to file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(mermaid_code)
            
            print(f"✅ Mermaid diagram saved to: {output_file}")
            print("\n💡 You can visualize this diagram at: https://mermaid.live/")
            print("\nMermaid code preview:")
            print("-" * 60)
            print(mermaid_code[:500] + "..." if len(mermaid_code) > 500 else mermaid_code)
            print("-" * 60)
            
        except Exception as e:
            print(f"⚠️  Could not generate Mermaid diagram: {e}")
    
    # ========================================================================
    # Main Interface: ask (Work Item 22)
    # ========================================================================
    
    def ask(self, question: str) -> GraphState:
        """
        Ask a question and get an answer through the agentic RAG workflow.
        
        Args:
            question: User question
            
        Returns:
            Final graph state with answer
        """
        if not self.compiled_graph:
            raise RuntimeError(
                "❌ Graph not compiled. Call build_graph() first."
            )
        
        # Initialize state
        initial_state: GraphState = {
            "question": question,
            "question_type": "",
            "target_sources": [],
            "chunks": [],
            "sources_used": {},
            "iteration": 0,
            "is_sufficient": False,
            "answer": "",
            "route_decision": "",
        }
        
        # Execute graph
        final_state = self.compiled_graph.invoke(initial_state)
        
        return final_state
    
    # ========================================================================
    # Output Formatting: print_result (Work Item 23)
    # ========================================================================
    
    def print_result(self, state: GraphState):
        """
        Print the final result with layered display.
        
        Shows:
        1. Question
        2. Routing decision
        3. Retrieval results
        4. Answer with source citations
        
        Args:
            state: Final graph state
        """
        print("\n" + "="*60)
        print("📋 FINAL RESULT")
        print("="*60)
        
        # Layer 1: Question
        print("\n【问题】")
        print(state["question"])
        
        # Layer 2: Routing Decision
        print("\n【路由决策】")
        print(f"  问题分类: {state['question_type']}")
        print(f"  路由策略: {state['route_decision']}")
        
        # Layer 3: Retrieval Results
        print("\n【检索结果】")
        sources_used = state["sources_used"]
        total_chunks = sum(sources_used.values())
        print(f"  总计: {total_chunks} chunks")
        print(f"  数据源分布:")
        for source_name, count in sources_used.items():
            print(f"    - {source_name}: {count} chunks")
        
        print(f"  迭代次数: {state['iteration']}")
        print(f"  信息充分性: {'充分 ✅' if state['is_sufficient'] else '不足但已达最大迭代次数 ⚠️'}")
        
        # Layer 4: Answer
        print("\n【答案】")
        print(state["answer"])
        
        print("\n" + "="*60)
    
    # ========================================================================
    # Demo Mode: run_demo (Work Item 25)
    # ========================================================================
    
    def run_demo(self):
        """
        Run demonstration with 3 predefined questions.
        
        Tests the three main question types:
        1. Basic information query
        2. Technical details query
        3. Comparison analysis query
        """
        print("\n" + "="*80)
        print("🎯 AGENTIC RAG DEMONSTRATION")
        print("="*80)
        print("\n本演示将依次回答3个预设问题，展示LangGraph的智能路由能力：")
        print("  Q1: 基础信息查询 (路由到官网)")
        print("  Q2: 技术细节查询 (路由到白皮书)")
        print("  Q3: 对比分析查询 (路由到手册+官网)")
        print()
        
        questions = [
            "Autel EVO Nest机巢和Dji Dock机场分别支持哪些无人机型号？",
            "Autel EVO Nest机巢和Dji Dock机场各自的数据传输速度是多少？",
            "与DJI Dock机场相比，Autel EVO Nest机巢的优势在哪？",
        ]
        
        start_time = time.time()
        
        for i, question in enumerate(questions, 1):
            print("\n" + "="*80)
            print(f"QUESTION {i}/3")
            print("="*80)
            
            question_start = time.time()
            
            # Execute query
            final_state = self.ask(question)
            
            # Print result
            self.print_result(final_state)
            
            question_elapsed = time.time() - question_start
            print(f"\n⏱️  Question {i} elapsed time: {question_elapsed:.1f} seconds")
            
            # Pause between questions
            if i < len(questions):
                print("\n" + "─"*80)
                input("Press Enter to continue to next question...")
        
        total_elapsed = time.time() - start_time
        
        print("\n" + "="*80)
        print("✨ DEMONSTRATION COMPLETED")
        print("="*80)
        print(f"\n⏱️  Total elapsed time: {total_elapsed:.1f} seconds")
        print(f"📊 Average time per question: {total_elapsed/len(questions):.1f} seconds")


# ============================================================================
# Main Function (Work Item 27)
# ============================================================================

def main():
    """
    Main function to run the Agentic RAG demonstration.
    """
    print("="*80)
    print("🤖 Agentic RAG Agent with LangGraph (Stage 3 - Educational Demo)")
    print("="*80)
    print("\n本演示展示LangGraph如何解决LangChain在多源RAG场景下的局限性。\n")
    
    try:
        # Initialize agent
        print("🔧 Step 1: Initializing Agentic RAG Agent...")
        agent = AgenticRAGAgent(
            vector_db_path="./tmp/lancedb",
            use_fastembed=True,
        )
        
        # Load vector stores (reuse from Stage 2)
        print("\n🗄️  Step 2: Loading existing vector stores...")
        vector_stores = agent.load_vector_stores()
        
        if len(vector_stores) == 0:
            print("\n❌ No vector stores found!")
            print("   Please run uav-nest-agent-multi-source.py first to create vector stores.")
            return
        
        # Build LangGraph
        print("\n🔧 Step 3: Building LangGraph workflow...")
        agent.build_graph()
        
        # Generate visualization
        print("\n📊 Step 4: Generating workflow visualization...")
        agent.visualize_graph()
        
        # Run demonstration
        print("\n🚀 Step 5: Running demonstration with 3 test questions...")
        input("\nPress Enter to start the demonstration...")
        agent.run_demo()
        
        # Print summary
        print("\n" + "="*80)
        print("📚 LEARNING SUMMARY")
        print("="*80)
        print("\n✅ LangGraph核心特性演示完成：")
        print("  1. ✅ 条件边（Conditional Edges）: 根据信息充分性自动路由")
        print("  2. ✅ 并行节点（Parallel Nodes）: 并发查询多个数据源")
        print("  3. ✅ 状态管理（State Management）: 跟踪查询历史和数据源")
        print("  4. ✅ 循环逻辑（Cycles）: 自动补充查询直到信息充分")
        print("\n✅ 相比阶段2（LangChain）的改进：")
        print("  - ✅ 自动问题分类和智能路由")
        print("  - ✅ 并行检索提升性能")
        print("  - ✅ 自动补充查询逻辑")
        print("  - ✅ 清晰的信息源标注")
        print("  - ✅ 内置可视化支持")
        print("\n💡 下一步建议：")
        print("  1. 查看生成的 agentic-rag-workflow.mmd 可视化图")
        print("  2. 尝试提出自己的问题，观察路由决策")
        print("  3. 调整阈值和路由规则，实验不同策略")
        print("  4. 对比阶段2和阶段3的代码复杂度和可维护性\n")
        
    except FileNotFoundError as e:
        print(f"\n❌ 文件未找到错误: {e}")
        print("   请确保已运行uav-nest-agent-multi-source.py创建向量库")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# Entry Point (Work Item 28)
# ============================================================================

if __name__ == "__main__":
    main()

