# 🔍 RAG: Retrieval-Augmented Generation

---

> © 2025–2026 kirankkai4-lab. Licensed under the [MIT License](../LICENSE).
>
> This is an **independent study guide** written in the author's own words for educational purposes. All AWS product names, service names, and trademarks — including Amazon Bedrock, Amazon OpenSearch, and Amazon Aurora — are the property of **Amazon.com, Inc. or its affiliates**. This guide is not affiliated with, endorsed by, or officially associated with AWS.
>
> Pricing figures, model IDs, API parameters, and service features described in this document **may change without notice**. Always verify current information at [docs.aws.amazon.com](https://docs.aws.amazon.com) before use in production or exam preparation.

---

> A practical guide to understanding and implementing Retrieval-Augmented Generation on AWS — with real-world examples, diagrams, and Bedrock Knowledge Base walkthroughs.

---

## 📚 Table of Contents

- [Why RAG Exists](#why-rag-exists)
- [How RAG Works](#how-rag-works)
- [Vector Databases](#vector-databases)
- [Chunking Strategies](#chunking-strategies)
- [AWS Bedrock Knowledge Bases](#aws-bedrock-knowledge-bases)
- [RAG vs Fine-Tuning vs Prompt Engineering](#rag-vs-fine-tuning-vs-prompt-engineering)
- [Common RAG Patterns](#common-rag-patterns)
- [Quick Reference](#quick-reference)

---

## Why RAG Exists

### The Problem: LLMs Have Limitations

Large Language Models are powerful, but they have two fundamental problems in production:

#### Problem 1: Knowledge Cutoff

```
LLM Training Data: Everything up to [cutoff date]

User asks: "What is our company's refund policy?"
           "What happened in the news yesterday?"
           "What is the current stock price?"

LLM response: ❌ Cannot answer — this data didn't exist at training time
```

**Analogy:** Imagine hiring an expert consultant who read every book ever written — but only up to 2023. They're incredibly knowledgeable, but they can't answer questions about anything that happened after they stopped reading.

#### Problem 2: Hallucination

When an LLM doesn't know something, it often makes up a plausible-sounding answer:

```
User: "What does Section 4.2 of our employee handbook say?"

LLM (without RAG): "Section 4.2 states that employees are entitled
                    to 15 days of annual leave..."
                    ← MADE UP! This could be completely wrong.

LLM (with RAG):    "According to your employee handbook (Section 4.2):
                    [actual text retrieved from your document]"
                    ← GROUNDED in real data ✅
```

#### Problem 3: Private/Proprietary Data

LLMs are trained on public internet data. Your company's internal documents, databases, and knowledge bases were never part of training:

```
Things LLMs don't know about YOUR business:
├── Internal product documentation
├── Customer support tickets
├── Legal contracts
├── Proprietary research
└── Real-time operational data
```

### The Solution: RAG

RAG solves all three problems by **retrieving relevant information at query time** and injecting it into the LLM's context:

```
WITHOUT RAG:
  User Question ──► LLM (static knowledge) ──► Answer (may be wrong/outdated)

WITH RAG:
  User Question ──► Search Knowledge Base ──► Retrieve Relevant Docs
                                                        │
                                                        ▼
                 Final Answer ◄── LLM ◄── Question + Retrieved Context
```

---

## How RAG Works

RAG has two phases: **Indexing** (done once) and **Retrieval** (done at query time).

### Phase 1: Indexing Pipeline

```
Your Documents (PDFs, Word docs, web pages, databases)
        │
        ▼
┌───────────────────┐
│   CHUNKING        │  Break documents into smaller pieces
│   "Split it up"   │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│   EMBEDDING       │  Convert each chunk into a vector (list of numbers)
│   "Encode it"     │  that captures its meaning
└───────────────────┘
        │
        ▼
┌───────────────────┐
│   VECTOR STORE    │  Store all vectors in a searchable database
│   "Store it"      │
└───────────────────┘
```

### Phase 2: Retrieval Pipeline (at query time)

```
User asks: "What is our return policy for electronics?"
        │
        ▼
┌───────────────────┐
│   EMBED QUERY     │  Convert question into a vector
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  SIMILARITY       │  Find the chunks whose vectors are
│  SEARCH           │  most similar to the query vector
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  TOP-K RESULTS    │  Retrieve the top 3-5 most relevant chunks
└───────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│  AUGMENTED PROMPT                           │
│                                             │
│  "Answer the question using the context     │
│   below.                                    │
│                                             │
│   Context:                                  │
│   [Chunk 1: Returns policy text...]         │
│   [Chunk 2: Electronics exceptions...]      │
│                                             │
│   Question: What is our return policy       │
│   for electronics?"                         │
└─────────────────────────────────────────────┘
        │
        ▼
┌───────────────────┐
│   LLM             │  Generates answer grounded in your actual documents
└───────────────────┘
        │
        ▼
"Based on our policy, electronics can be returned within
 30 days with original receipt. See Section 3.2 for details."
```

### Similarity Search: How Vectors Are Compared

The core of RAG is finding documents that are *semantically similar* to the query — not just keyword matching:

```
Query: "How do I reset my password?"

Keyword search finds:        Semantic search finds:
✓ "reset password"           ✓ "reset password"
✗ "change login credentials" ✓ "change login credentials"  ← Same meaning!
✗ "forgot account access"    ✓ "forgot account access"     ← Same meaning!
```

Similarity is measured using **cosine similarity** — how close two vectors point in the same direction in high-dimensional space.

```
Cosine Similarity Score:
1.0  = Identical meaning
0.8+ = Very similar
0.5  = Somewhat related
0.0  = Completely unrelated
```

---

## Vector Databases

A vector database is purpose-built to store, index, and search embeddings at scale.

### Why Not Just Use a Regular Database?

```
Regular Database (SQL):
  SELECT * FROM docs WHERE content LIKE '%password reset%'
  → Only finds exact keyword matches
  → Slow on billions of records for similarity search

Vector Database:
  search(query_vector, top_k=5)
  → Finds semantically similar content
  → Uses ANN (Approximate Nearest Neighbor) for fast search
  → Handles millions/billions of vectors efficiently
```

### Major Vector Database Options

| Database | Type | Best For | AWS Integration |
|----------|------|----------|-----------------|
| **Amazon OpenSearch** | Managed service | Full-text + vector hybrid search | Native AWS, Bedrock KB |
| **pgvector (PostgreSQL)** | Extension | Existing PostgreSQL users | Amazon RDS, Aurora |
| **Pinecone** | Fully managed | Pure vector search, simplicity | Third-party, API-based |
| **Chroma** | Open source | Local development, prototyping | Self-hosted |
| **Weaviate** | Open source | Multi-modal (text + images) | Self-hosted or cloud |
| **FAISS** | Library | In-memory, research, offline | Use with SageMaker |

### Amazon OpenSearch for RAG

OpenSearch is AWS's recommended vector store for production RAG workloads:

```
Features:
├── k-NN (k-Nearest Neighbor) vector search
├── Hybrid search (keyword + vector combined)
├── Metadata filtering (filter by date, category, author)
├── Serverless option (no cluster management)
└── Native integration with Amazon Bedrock

Index Configuration Example:
{
  "settings": {
    "index.knn": true
  },
  "mappings": {
    "properties": {
      "embedding": {
        "type": "knn_vector",
        "dimension": 1536        ← Must match embedding model output size
      },
      "text": { "type": "text" },
      "source": { "type": "keyword" }
    }
  }
}
```

### pgvector on Amazon RDS/Aurora

If you already use PostgreSQL, pgvector adds vector search without a new database:

```sql
-- Enable extension
CREATE EXTENSION vector;

-- Create table with vector column
CREATE TABLE documents (
  id SERIAL PRIMARY KEY,
  content TEXT,
  embedding vector(1536),    -- 1536 dimensions for Titan embeddings
  source VARCHAR(255),
  created_at TIMESTAMP
);

-- Create index for fast similarity search
CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops);

-- Search for similar documents
SELECT content, source,
       1 - (embedding <=> '[0.1, 0.5, ...]'::vector) AS similarity
FROM documents
ORDER BY embedding <=> '[0.1, 0.5, ...]'::vector
LIMIT 5;
```

---

## Chunking Strategies

Chunking is how you split documents before embedding. It's one of the most impactful decisions in a RAG system — bad chunking leads to poor retrieval.

### Why Chunking Matters

```
Problem with chunks too LARGE:
├── Retrieval returns too much irrelevant text
├── Exceeds LLM context window
└── Less precise — relevant sentence buried in noise

Problem with chunks too SMALL:
├── Loses important context
├── A sentence without surrounding context is meaningless
└── "The rate is 5%" — 5% of what?
```

### Chunking Strategy Comparison

#### 1. Fixed-Size Chunking

Split every N characters or tokens, with optional overlap:

```
Document: "The return policy allows 30 days. Electronics are excluded.
           Clothing can be returned within 60 days with tags attached."

Chunk size: 50 chars, Overlap: 10 chars

Chunk 1: "The return policy allows 30 days. Electronics"
Chunk 2: "Electronics are excluded. Clothing can be"
Chunk 3: "can be returned within 60 days with tags"

Overlap prevents context loss at boundaries ↑
```

**Best for:** Uniform documents (logs, structured data)
**AWS Bedrock default:** 300 tokens with 20% overlap

#### 2. Recursive Character Chunking

Split by natural boundaries (paragraphs → sentences → words), falling back to smaller units:

```
Priority order:
1. Try splitting by: \n\n  (paragraph breaks)
2. If still too large, split by: \n  (line breaks)
3. If still too large, split by: .  (sentences)
4. If still too large, split by: ' ' (words)

Result: Chunks that respect document structure
```

**Best for:** General text documents, mixed content
**Most common approach** in production RAG systems

#### 3. Semantic Chunking

Use an embedding model to find natural "topic shifts" in the document:

```
"...about the return policy for electronics.

[HIGH SEMANTIC SHIFT DETECTED HERE]

Now let's discuss shipping options..."

→ Split point identified automatically by meaning, not just characters
```

**Best for:** Long documents with clear topic sections
**Trade-off:** Slower to index, but higher quality retrieval

#### 4. Document-Structure Chunking

Split by the document's own structure (headers, sections):

```
Markdown/HTML document:
## Section 1: Return Policy      ← Chunk boundary
Content about returns...

## Section 2: Shipping Policy    ← Chunk boundary
Content about shipping...
```

**Best for:** Well-structured documents (manuals, wikis, documentation)

### Chunking Decision Guide

```
What type of document?
        │
        ├── Structured (manuals, wikis, handbooks)
        │   └── Use: Document-Structure Chunking
        │
        ├── General text (articles, reports, emails)
        │   └── Use: Recursive Character Chunking
        │
        ├── Highly variable length topics
        │   └── Use: Semantic Chunking
        │
        └── Uniform/structured data (logs, CSVs)
            └── Use: Fixed-Size Chunking
```

---

## AWS Bedrock Knowledge Bases

Amazon Bedrock Knowledge Bases is a fully managed RAG service — it handles ingestion, chunking, embedding, vector storage, and retrieval automatically.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│              Amazon Bedrock Knowledge Base                   │
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  Data Source │───►│  Ingestion   │───►│    Vector    │  │
│  │              │    │  Pipeline    │    │    Store     │  │
│  │  • S3 Bucket │    │  • Chunking  │    │              │  │
│  │  • Web crawl │    │  • Embedding │    │  OpenSearch  │  │
│  │  • SharePoint│    │    (Titan)   │    │  or Aurora   │  │
│  │  • Confluence│    │              │    │  pgvector    │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                   │          │
│  ┌────────────────────────────────────────────────▼──────┐  │
│  │                    Retrieval API                       │  │
│  │  retrieve() / retrieve_and_generate()                  │  │
│  └────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Setting Up a Knowledge Base (Step-by-Step)

**Step 1: Prepare your data in S3**
```
s3://my-knowledge-base-bucket/
├── employee-handbook.pdf
├── product-documentation/
│   ├── user-guide.pdf
│   └── api-reference.md
└── faqs/
    └── common-questions.txt
```

**Step 2: Create Knowledge Base via AWS Console or SDK**
```python
import boto3

bedrock_agent = boto3.client('bedrock-agent', region_name='us-east-1')

# Create Knowledge Base
kb = bedrock_agent.create_knowledge_base(
    name='company-knowledge-base',
    description='Internal company documents for RAG',
    roleArn='arn:aws:iam::123456789:role/BedrockKBRole',
    knowledgeBaseConfiguration={
        'type': 'VECTOR',
        'vectorKnowledgeBaseConfiguration': {
            'embeddingModelArn': 'arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v1'
        }
    },
    storageConfiguration={
        'type': 'OPENSEARCH_SERVERLESS',
        'opensearchServerlessConfiguration': {
            'collectionArn': 'arn:aws:aoss:us-east-1:123456789:collection/my-collection',
            'vectorIndexName': 'my-index',
            'fieldMapping': {
                'vectorField': 'embedding',
                'textField': 'text',
                'metadataField': 'metadata'
            }
        }
    }
)
```

**Step 3: Add Data Source and Sync**
```python
# Add S3 data source
data_source = bedrock_agent.create_data_source(
    knowledgeBaseId=kb['knowledgeBase']['knowledgeBaseId'],
    name='s3-documents',
    dataSourceConfiguration={
        'type': 'S3',
        's3Configuration': {
            'bucketArn': 'arn:aws:s3:::my-knowledge-base-bucket'
        }
    }
)

# Trigger ingestion
bedrock_agent.start_ingestion_job(
    knowledgeBaseId=kb['knowledgeBase']['knowledgeBaseId'],
    dataSourceId=data_source['dataSource']['dataSourceId']
)
```

**Step 4: Query the Knowledge Base**
```python
bedrock_runtime = boto3.client('bedrock-agent-runtime', region_name='us-east-1')

# Option A: Retrieve only (get relevant chunks)
response = bedrock_runtime.retrieve(
    knowledgeBaseId='YOUR_KB_ID',
    retrievalQuery={
        'text': 'What is the return policy for electronics?'
    },
    retrievalConfiguration={
        'vectorSearchConfiguration': {
            'numberOfResults': 5
        }
    }
)

# Option B: Retrieve AND Generate (full RAG pipeline)
response = bedrock_runtime.retrieve_and_generate(
    input={'text': 'What is the return policy for electronics?'},
    retrieveAndGenerateConfiguration={
        'type': 'KNOWLEDGE_BASE',
        'knowledgeBaseConfiguration': {
            'knowledgeBaseId': 'YOUR_KB_ID',
            'modelArn': 'arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0'
        }
    }
)

print(response['output']['text'])
# "Based on the company policy, electronics can be returned
#  within 30 days with original receipt..."
```

### Bedrock KB Chunking Options

| Strategy | Description | Chunk Size |
|----------|-------------|------------|
| **Default** | Fixed-size with overlap | 300 tokens, 20% overlap |
| **Fixed size** | Custom token count | Configurable |
| **Hierarchical** | Parent + child chunks for context | Configurable |
| **Semantic** | Embedding-based natural boundaries | Dynamic |
| **None** | Each file = one chunk | Full document |

---

## RAG vs Fine-Tuning vs Prompt Engineering

Understanding when to use each approach is critical for the AWS exam and real-world decisions:

```
┌─────────────────────────────────────────────────────────────────┐
│                    DECISION FRAMEWORK                            │
│                                                                  │
│  Is your data private or changing frequently?                   │
│         │                                                        │
│    YES ─┼─► Use RAG                                             │
│         │                                                        │
│    NO   └─► Is it a knowledge problem or a behavior problem?    │
│                    │                                             │
│            KNOWLEDGE ──► RAG                                    │
│            BEHAVIOR  ──► Fine-tuning                            │
└─────────────────────────────────────────────────────────────────┘
```

| Approach | Best For | Cost | Effort | Freshness |
|----------|----------|------|--------|-----------|
| **Prompt Engineering** | Simple tasks, well-defined formats | Low | Low | N/A |
| **RAG** | Private data, frequently updated data, grounded answers | Medium | Medium | Real-time |
| **Fine-tuning** | Specific style/tone, domain expertise, consistent behavior | High | High | Static |

---

## Common RAG Patterns

### Naive RAG (Basic)
```
Query → Embed → Search → Retrieve → Augment → Generate
```
Simple, works well for straightforward use cases.

### Advanced RAG with Re-ranking
```
Query → Embed → Search → Retrieve Top-20
                                  │
                              Re-ranker model
                              (scores relevance more precisely)
                                  │
                              Top-5 chunks
                                  │
                              Augment → Generate
```

### Hybrid Search (Recommended for Production)
```
Query ──► Keyword Search (BM25)  ──┐
       │                           ├──► Combine scores ──► Top results
       └──► Vector Search         ──┘
```
Combines exact keyword matching with semantic similarity for best results. Supported natively in **Amazon OpenSearch**.

### Agentic RAG
```
Query ──► Agent decides which knowledge base(s) to search
              │
              ├── Search KB 1 (product docs)
              ├── Search KB 2 (legal contracts)
              └── Search external API
                        │
                    Synthesize results ──► Response
```

---

## Quick Reference

| Concept | Simple Explanation |
|---------|-------------------|
| **RAG** | Give the LLM relevant documents at query time so it can answer accurately |
| **Embedding** | Convert text into a list of numbers that captures meaning |
| **Vector Store** | A database optimized for finding similar embeddings |
| **Chunking** | Splitting documents into smaller pieces before indexing |
| **Cosine Similarity** | How similar two vectors are (1.0 = identical, 0.0 = unrelated) |
| **Top-K** | How many chunks to retrieve (typically 3-5) |
| **Bedrock KB** | AWS managed service that does all RAG steps for you |

### AWS Services to Know for the Exam

| Service | Role in RAG |
|---------|------------|
| **Amazon Bedrock Knowledge Bases** | Fully managed RAG pipeline |
| **Amazon Titan Embeddings** | Converts text to vectors (embedding model) |
| **Amazon OpenSearch Serverless** | Vector store for Bedrock KB |
| **Amazon Aurora (pgvector)** | PostgreSQL-based vector store option |
| **Amazon S3** | Store source documents |
| **AWS Lambda** | Custom ingestion or retrieval logic |

---

*Written as part of the AWS Generative AI Developer 30-day study guide.*
*Last updated: June 2026*
