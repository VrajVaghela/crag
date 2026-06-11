# CRAG - Corrective Retrieval-Augmented Generation

Implementation of Corrective RAG (CRAG), an advanced retrieval-augmented generation system that validates and corrects retrieved documents through a feedback loop, improving answer accuracy and reliability.

## Overview

CRAG enhances traditional RAG by:
- **Document Retrieval**: Fetches relevant documents from PDF sources using vector similarity
- **Retrieval Validation**: Assesses the relevance of retrieved documents
- **Corrective Mechanisms**: Identifies and fixes retrieval failures
- **Web Augmentation**: Supplements with web search when needed
- **Iterative Refinement**: Continuously improves answers based on document quality

## Key Features

- **PDF Processing**: Load and process multi-document PDFs with metadata preservation
- **Vector Embeddings**: HuggingFace embeddings for semantic similarity
- **FAISS Vector Store**: Efficient similarity search with FAISS
- **Chunk-Based Retrieval**: Intelligent document chunking (900 tokens, 150 overlap)
- **LangGraph Workflows**: State-managed retrieval pipeline
- **Web Search Integration**: Tavily API for supplemental searches
- **Fallback Mechanisms**: Automatic degradation to web search on retrieval failure

## Project Structure

```
crag/
├── crag.py               # Core CRAG implementation
├── documents/            # PDF documents directory
│   ├── book1.pdf
│   ├── books2.pdf
│   └── books3.pdf
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables
└── vector_store/         # FAISS vector store (generated)
```

## Getting Started

### Prerequisites

- Python 3.10+
- PDFs in `documents/` directory
- HuggingFace account (free)
- Tavily API Key (optional, for web fallback)

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Create documents directory
mkdir -p documents

# Add your PDF files to documents/
# (Update file paths in crag.py as needed)
```

### Configuration

Create a `.env` file:

```
HUGGINGFACEHUB_API_TOKEN=your_hf_token
TAVILY_API_KEY=your_tavily_key
```

## How CRAG Works

### 1. Document Processing

```python
from langchain_community.document_loaders import PyPDFLoader

docs = (
    PyPDFLoader("./documents/book1.pdf").load()
    + PyPDFLoader("./documents/books2.pdf").load()
    + PyPDFLoader("./documents/books3.pdf").load()
)
```

### 2. Document Chunking

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

chunks = RecursiveCharacterTextSplitter(
    chunk_size=900,      # Tokens per chunk
    chunk_overlap=150    # Overlap for context preservation
).split_documents(docs)
```

### 3. Vector Store Creation

```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings()
vector_store = FAISS.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)
```

### 4. Retrieval Validation

The system evaluates retrieved documents for:
- **Relevance**: Do documents match the query intent?
- **Completeness**: Do they contain sufficient information?
- **Accuracy**: Is the information reliable?

### 5. Corrective Actions

If retrieval fails:
- **Re-rank**: Adjust similarity thresholds
- **Web Search**: Fetch information from web sources
- **Hybrid Approach**: Combine document and web results

### 6. Answer Generation

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace

prompt = ChatPromptTemplate.from_template("""
    Use the following documents to answer the question.
    If documents don't contain information, use web search results.
    
    Documents: {context}
    Question: {question}
    Answer:
""")

chain = prompt | llm | output_parser
```

## Technologies

- **LangChain**: RAG pipeline and tool integration
- **LangGraph**: Workflow orchestration and state management
- **FAISS**: Vector similarity search
- **HuggingFace**: Embeddings and LLM models
- **Tavily**: Web search API
- **PyPDF**: PDF document loading

## Configuration Parameters

### Document Chunking

```python
chunk_size = 900        # Adjust based on context window
chunk_overlap = 150     # Balance between redundancy and efficiency
```

### Retrieval

```python
search_kwargs = {"k": 4}  # Number of documents to retrieve
search_type = "similarity"  # Or "mmr" for diversity
```

### LLM Parameters

```python
temperature = 0.1       # Lower = more deterministic
max_tokens = 1024       # Response length limit
top_p = 0.95            # Nucleus sampling
```

## Usage Example

```python
from crag import create_crag_chain

# Initialize CRAG
crag = create_crag_chain()

# Query the system
result = crag.invoke({
    "question": "What is the company's revenue in 2024?",
    "documents": retrieved_docs  # From vector store
})

print(result["answer"])
```

## Performance Optimization

### Vector Store Indexing
- FAISS uses Flat L2 indexing by default
- For large document sets, consider IVFFlat or HNSW indexes

### Batch Processing
```python
# Process multiple queries efficiently
queries = ["Q1", "Q2", "Q3"]
results = [crag.invoke({"question": q}) for q in queries]
```

### Caching
- Cache embeddings for frequently queried documents
- Use Redis for distributed caching

## Troubleshooting

### Poor Retrieval Results
- Increase `k` in `search_kwargs`
- Adjust `chunk_size` and `chunk_overlap`
- Try different embedding models

### Out of Memory
- Reduce `chunk_size`
- Process documents in batches
- Use a lighter embedding model

### Slow Performance
- Pre-compute and save FAISS index
- Use GPU acceleration for embeddings
- Implement caching layer

## Evaluation Metrics

CRAG effectiveness measured by:
- **Retrieval Precision**: Accuracy of retrieved documents
- **Answer Faithfulness**: Answers grounded in documents
- **Hallucination Rate**: Unsupported claims in answers
- **Coverage**: Percentage of questions with sufficient documents

## Advanced Features

### Metadata Filtering

```python
retriever.search_kwargs = {
    "k": 4,
    "filter": {"source": "book1.pdf"}
}
```

### Hybrid Search

Combine keyword and semantic search:

```python
from langchain_community.retrievers import BM25Retriever

bm25_retriever = BM25Retriever.from_documents(docs)
ensemble = EnsembleRetriever(
    retrievers=[bm25_retriever, semantic_retriever],
    weights=[0.3, 0.7]
)
```

## License

MIT

## Contributing

Contributions welcome! Areas for improvement:
- Additional correction strategies
- Advanced ranking mechanisms
- Multi-modal document support
- Distributed processing
