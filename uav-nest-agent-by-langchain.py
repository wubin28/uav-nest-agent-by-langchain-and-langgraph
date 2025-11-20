"""
LangChain-based UAV Nest RAG Agent

This application uses LangChain to create a RAG (Retrieval-Augmented Generation)
system that answers questions about UAV products based on Autel Robotics product brochure.

Features:
- Local vector storage using LanceDB (no database installation required)
- FREE local embeddings using FastEmbed (no API key needed)
- DeepSeek Reasoner model for intelligent responses
- Automatic knowledge retrieval and source citation
"""

import os
from pathlib import Path
from typing import Optional
import getpass

from dotenv import load_dotenv
import lancedb

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import LanceDB
from langchain_core.documents import Document
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


class UAVNestAgent:
    """
    A RAG agent that answers questions about UAV products based on Autel Robotics product brochure.
    """
    
    def __init__(
        self,
        pdf_path: str,
        vector_db_path: str = "./tmp/lancedb",
        table_name: str = "uav_nest",
        use_fastembed: bool = True,
        deepseek_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
    ):
        """
        Initialize the UAV Nest Agent.
        
        Args:
            pdf_path: Path to the Autel Robotics product brochure PDF
            vector_db_path: Path to store the LanceDB vector database
            table_name: Name of the LanceDB table
            use_fastembed: Whether to use FastEmbed (free) or OpenAI embeddings
            deepseek_api_key: DeepSeek API Key (optional, will prompt if not provided)
            openai_api_key: OpenAI API Key (optional, only needed if not using FastEmbed)
        """
        self.pdf_path = pdf_path
        self.vector_db_path = vector_db_path
        self.table_name = table_name
        
        # Get DeepSeek API Key (from parameter, environment, or interactive input)
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
        
        # Initialize vector store
        self.vector_store = None
        self.retriever = None
        self.rag_chain = None
        
    def _initialize_embeddings(self, use_fastembed: bool):
        """Initialize embedding model (FastEmbed or OpenAI)."""
        if use_fastembed and FASTEMBED_AVAILABLE:
            print("🔧 Initializing FastEmbed embeddings (free, local)...")
            # Using default FastEmbed model (same as agno for consistency)
            # The default model works well for both English and Chinese text
            return FastEmbedEmbeddings()  # Use default model
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
            temperature=0,  # More deterministic for factual answers
            max_tokens=4000,
        )
    
    def load_and_index_pdf(self, force_reload: bool = False):
        """
        Load PDF, split into chunks, and create vector store.
        
        Args:
            force_reload: If True, reload even if vector store exists
        """
        vector_db_dir = Path(self.vector_db_path)
        
        # Check if vector store already exists and has the table
        table_exists = False
        if vector_db_dir.exists() and not force_reload:
            try:
                db = lancedb.connect(self.vector_db_path)
                table_exists = self.table_name in db.table_names()
                if table_exists:
                    table = db.open_table(self.table_name)
                    row_count = table.count_rows()
                    if row_count > 0:
                        print(f"📂 Loading existing vector store from {self.vector_db_path}...")
                        print(f"   Table '{self.table_name}' has {row_count} rows")
                        self._load_existing_vector_store()
                        return
                    else:
                        print(f"⚠️  Table '{self.table_name}' exists but is empty, rebuilding...")
                        table_exists = False
                else:
                    print(f"⚠️  Table '{self.table_name}' not found, creating new index...")
            except Exception as e:
                print(f"⚠️  Error checking existing table: {e}")
                print(f"   Creating new index...")
                table_exists = False
        
        if force_reload and vector_db_dir.exists():
            print(f"🔄 Force reload requested, recreating vector store...")
        
        print(f"📄 Loading PDF from {self.pdf_path}...")
        
        # Check if PDF exists
        if not Path(self.pdf_path).exists():
            raise FileNotFoundError(
                f"❌ PDF file not found: {self.pdf_path}\n"
                f"   Please place your Autel Robotics product brochure PDF at this location."
            )
        
        # Load PDF
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()
        
        print(f"✅ Loaded {len(documents)} pages from PDF")
        
        # Split documents into chunks
        print("✂️  Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,  # Larger chunks to capture complete policy sections
            chunk_overlap=300,  # More overlap to ensure context continuity
            length_function=len,
            separators=["\n\n", "\n", "。", "；", " ", ""]  # Add Chinese punctuation separators
        )
        chunks = text_splitter.split_documents(documents)
        
        print(f"✅ Created {len(chunks)} text chunks")
        
        # Create vector store
        print("🗄️  Creating vector store (this may take a moment)...")
        
        # Create LanceDB connection
        db = lancedb.connect(self.vector_db_path)
        
        # Create vector store from documents
        self.vector_store = LanceDB.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            connection=db,
            table_name=self.table_name,
        )
        
        print(f"✅ Vector store created at {self.vector_db_path}")
        
        # Create retriever
        self._setup_retriever()
    
    def _load_existing_vector_store(self):
        """Load an existing vector store."""
        db = lancedb.connect(self.vector_db_path)
        
        self.vector_store = LanceDB(
            connection=db,
            embedding=self.embeddings,
            table_name=self.table_name,
        )
        
        print("✅ Vector store loaded successfully")
        
        # Create retriever
        self._setup_retriever()
    
    def _setup_retriever(self):
        """Setup retriever and RAG chain."""
        # Create retriever
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 20}  # Retrieve top 20 chunks for better coverage (agno uses 10 by default)
        )
        
        # Create RAG prompt template (支持中文回答)
        template = """你是一个专业的无人机产品问答助手，基于 Autel Robotics 产品手册内容回答关于无人机产品的问题。

请根据以下产品手册的上下文来回答问题。如果在提供的上下文中找不到答案，请明确说明。

回答时请遵循以下要求：
1. 使用中文回答
2. 确保答案与产品手册内容一致
3. 尽可能引用来源页码
4. 回答要具体、准确

产品手册相关内容：
{context}

问题：{question}

回答（请具体说明并引用来源）："""

        prompt = ChatPromptTemplate.from_template(template)
        
        # Create RAG chain
        def format_docs(docs):
            """Format retrieved documents for the prompt."""
            formatted = []
            for doc in docs:
                page = doc.metadata.get('page', 'unknown')
                content = doc.page_content
                formatted.append(f"[Page {page + 1}]\n{content}")
            return "\n\n".join(formatted)
        
        self.rag_chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        print("✅ RAG chain configured successfully")
    
    def ask(self, question: str, stream: bool = False) -> str:
        """
        Ask a question about the UAV products.
        
        Args:
            question: Question to ask
            stream: Whether to stream the response
            
        Returns:
            Answer from the agent
        """
        if not self.rag_chain:
            raise RuntimeError(
                "❌ RAG chain not initialized. Call load_and_index_pdf() first."
            )
        
        print(f"\n{'='*60}")
        print(f"❓ Question: {question}")
        print(f"{'='*60}\n")
        
        if stream:
            print("💬 Answer:\n")
            full_response = ""
            for chunk in self.rag_chain.stream(question):
                print(chunk, end="", flush=True)
                full_response += chunk
            print("\n")
            return full_response
        else:
            response = self.rag_chain.invoke(question)
            print(f"💬 Answer:\n{response}\n")
            return response


def main():
    """Main function to demonstrate the agent."""
    
    # Configuration
    PDF_PATH = "./Autel-Robotics-Products-Brochure.pdf"
    VECTOR_DB_PATH = "./tmp/lancedb"
    
    print("="*60)
    print("🤖 UAV Nest Agent (LangChain)")
    print("="*60)
    print()
    
    try:
        # Initialize agent
        agent = UAVNestAgent(
            pdf_path=PDF_PATH,
            vector_db_path=VECTOR_DB_PATH,
            use_fastembed=True,  # Use free local embeddings
        )
        
        # Load and index PDF (force reload to ensure using correct embeddings)
        # Note: Set to False after first successful run to skip re-indexing
        agent.load_and_index_pdf(force_reload=True)
        
        print("\n" + "="*60)
        print("🚀 Agent ready! Starting demo queries...")
        print("="*60)
        
        # Demo 1: 关于 Autel 产品的核心问题
        print("\n📋 演示问题 1: 关于产品技术规格")
        print("-" * 60)
        agent.ask(
            "EVO Nest产品的主要技术规格是什么？",
            stream=True
        )
        
        # 询问是否继续演示
        print("\n" + "-" * 60)
        try:
            response = input("\n按 Enter 继续第二个演示问题，或输入 'q' 退出: ").strip()
            if response.lower() == 'q':
                print("\n👋 演示结束\n")
                return
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 演示结束\n")
            return
        
        # Demo 2: 另一个常见问题
        print("\n📋 演示问题 2: 关于 Autel 产品特性")
        print("-" * 60)
        agent.ask(
            "Autel Robotics 有哪些主要的无人机产品？",
            stream=True
        )
        
        print("\n" + "="*60)
        print("✨ 演示完成！")
        print("="*60)
        print("\n💡 下一步建议:")
        print("   1. 修改代码尝试不同问题")
        print("   2. 探索更多产品特性\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        raise


if __name__ == "__main__":
    main()

