# ☁️ AWS AI/ML Services Overview

---

> © 2025–2026 kirankkai4-lab. Licensed under the [MIT License](../LICENSE).
>
> This is an **independent study guide** written in the author's own words for educational purposes. All AWS product names, service names, and trademarks — including Amazon Bedrock, Amazon SageMaker, Amazon Titan, Amazon Q, Amazon Kendra, Amazon Comprehend, and Amazon Rekognition — are the property of **Amazon.com, Inc. or its affiliates**. This guide is not affiliated with, endorsed by, or officially associated with AWS.
>
> Pricing figures, model IDs, API parameters, and service features described in this document **may change without notice**. Always verify current information at [docs.aws.amazon.com](https://docs.aws.amazon.com) before use in production or exam preparation.

---

> A practical guide to the AWS AI/ML ecosystem — covering Amazon Bedrock, SageMaker, Titan models, and the broader suite of AI services. Includes model IDs, code examples, pricing tiers, and when to use what.

---

> ⚠️ **DISCLAIMER**
>
> The information in this document — including model IDs, pricing figures, API parameters, service features, and availability — was accurate at the time of writing (**June 2026**) but is subject to change without notice.
>
> AWS frequently updates its services, introduces new models, deprecates old ones, and adjusts pricing. **Always verify the latest details using the official AWS documentation before using this guide in a production environment or exam preparation:**
>
> - 📖 [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
> - 📖 [AWS SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
> - 💰 [AWS Bedrock Pricing](https://aws.amazon.com/bedrock/pricing/)
> - 🆔 [Bedrock Model IDs Reference](https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html)
>
> This document is an independent study guide and is **not affiliated with, endorsed by, or officially associated with Amazon Web Services (AWS)**. All AWS product names, logos, and trademarks are the property of Amazon.com, Inc. or its affiliates.

---

## 📚 Table of Contents

- [The AWS AI/ML Landscape](#the-awsaiml-landscape)
- [Amazon Bedrock](#amazon-bedrock)
- [AWS Titan Models](#aws-titan-models)
- [Amazon SageMaker](#amazon-sagemaker)
- [Amazon Q](#amazon-q)
- [Amazon Kendra](#amazon-kendra)
- [Amazon Comprehend](#amazon-comprehend)
- [Amazon Rekognition](#amazon-rekognition)
- [Choosing the Right Service](#choosing-the-right-service)
- [Quick Reference](#quick-reference)

---

## The AWS AI/ML Landscape

AWS organizes its AI/ML services into three layers:

```
┌─────────────────────────────────────────────────────────────────┐
│                      LAYER 3: AI SERVICES                        │
│           Pre-built AI — no ML knowledge required                │
│                                                                  │
│  Rekognition  Comprehend  Transcribe  Polly  Translate           │
│  Textract     Kendra      Forecast    Fraud Detector  Lex        │
│  Amazon Q     Personalize CodeGuru                               │
├─────────────────────────────────────────────────────────────────┤
│                   LAYER 2: FOUNDATION MODELS                     │
│        Access and customize LLMs — some ML knowledge needed      │
│                                                                  │
│        Amazon Bedrock    (managed FM access + RAG + Agents)      │
│        AWS Titan Models  (Amazon's own foundation models)        │
├─────────────────────────────────────────────────────────────────┤
│                   LAYER 1: ML PLATFORM                           │
│         Build/train/deploy your own models — ML expertise needed │
│                                                                  │
│        Amazon SageMaker  (end-to-end ML platform)               │
│        AWS Inferentia / Trainium  (custom ML chips)              │
└─────────────────────────────────────────────────────────────────┘
```

**Rule of thumb:**
- Have a specific AI task (translation, image recognition)? → **Layer 3 AI Services**
- Need to use or customize a large language model? → **Layer 2 (Bedrock)**
- Building a fully custom ML model from scratch? → **Layer 1 (SageMaker)**

---

## Amazon Bedrock

Amazon Bedrock is AWS's fully managed service for accessing, customizing, and deploying **foundation models (FMs)** — without managing any infrastructure.

### What Bedrock Does

```
┌─────────────────────────────────────────────────────────────────┐
│                     AMAZON BEDROCK                               │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │               MODEL ACCESS                               │   │
│  │  Anthropic Claude  │  Amazon Titan  │  Meta Llama        │   │
│  │  Mistral           │  Cohere        │  Stability AI      │   │
│  │  AI21 Labs         │  Writer        │  (and more)        │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────┐    │
│  │  Knowledge  │  │   Agents    │  │  Model Customization  │    │
│  │  Bases (RAG)│  │ (Agentic AI)│  │  (Fine-tuning)        │    │
│  └─────────────┘  └─────────────┘  └──────────────────────┘    │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────┐    │
│  │  Guardrails │  │  Evaluation │  │  Model Invocation     │    │
│  │  (Safety)   │  │  (Quality)  │  │  Logging              │    │
│  └─────────────┘  └─────────────┘  └──────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### Invoking Models — Three Ways

#### Method 1: InvokeModel (Single Request)

```python
import boto3
import json

bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1'
)

# Invoke Claude 3 Sonnet
response = bedrock_runtime.invoke_model(
    modelId='anthropic.claude-3-sonnet-20240229-v1:0',
    body=json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": "Explain what Amazon Bedrock is in 3 bullet points."
            }
        ]
    }),
    contentType='application/json',
    accept='application/json'
)

result = json.loads(response['body'].read())
print(result['content'][0]['text'])
```

#### Method 2: InvokeModelWithResponseStream (Streaming)

```python
# Stream tokens as they're generated (better UX for chatbots)
response = bedrock_runtime.invoke_model_with_response_stream(
    modelId='anthropic.claude-3-sonnet-20240229-v1:0',
    body=json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Write a short poem about AWS."}]
    })
)

stream = response.get('body')
for event in stream:
    chunk = event.get('chunk')
    if chunk:
        delta = json.loads(chunk.get('bytes').decode())
        if delta['type'] == 'content_block_delta':
            print(delta['delta']['text'], end='', flush=True)
```

#### Method 3: Converse API (Unified, Recommended)

The Converse API works across all models with the same interface — no model-specific formatting:

```python
# Same code works for Claude, Titan, Llama, Cohere, etc.
response = bedrock_runtime.converse(
    modelId='anthropic.claude-3-sonnet-20240229-v1:0',
    messages=[
        {"role": "user", "content": [{"text": "What is RAG in AI?"}]}
    ],
    system=[
        {"text": "You are a helpful AWS cloud architect."}
    ],
    inferenceConfig={
        "maxTokens": 512,
        "temperature": 0.7,
        "topP": 0.9
    }
)

print(response['output']['message']['content'][0]['text'])
```

### Model IDs Reference

These are the model identifiers you pass to `modelId`:

#### Anthropic Claude (Most Popular for Text)

| Model | Model ID | Best For |
|-------|----------|----------|
| Claude 3.5 Sonnet | `anthropic.claude-3-5-sonnet-20241022-v2:0` | Best balance of speed + intelligence |
| Claude 3 Sonnet | `anthropic.claude-3-sonnet-20240229-v1:0` | General purpose, cost-effective |
| Claude 3 Haiku | `anthropic.claude-3-haiku-20240307-v1:0` | Fastest, lowest cost |
| Claude 3 Opus | `anthropic.claude-3-opus-20240229-v1:0` | Most capable, complex tasks |
| Claude 2.1 | `anthropic.claude-v2:1` | Legacy, 200K context window |

#### Amazon Titan

| Model | Model ID | Best For |
|-------|----------|----------|
| Titan Text Express | `amazon.titan-text-express-v1` | General text generation |
| Titan Text Lite | `amazon.titan-text-lite-v1` | Fast, cost-efficient text |
| Titan Text Premier | `amazon.titan-text-premier-v1:0` | High quality, RAG use cases |
| Titan Embeddings V2 | `amazon.titan-embed-text-v2:0` | Text embeddings for RAG |
| Titan Multimodal Embeddings | `amazon.titan-embed-image-v1` | Image + text embeddings |
| Titan Image Generator | `amazon.titan-image-generator-v1` | Image generation |

#### Meta Llama

| Model | Model ID | Best For |
|-------|----------|----------|
| Llama 3.1 8B Instruct | `meta.llama3-1-8b-instruct-v1:0` | Lightweight, fast |
| Llama 3.1 70B Instruct | `meta.llama3-1-70b-instruct-v1:0` | High quality, open weights |
| Llama 3.1 405B Instruct | `meta.llama3-1-405b-instruct-v1:0` | Largest Llama, near-GPT4 |

#### Other Models

| Provider | Model ID | Best For |
|----------|----------|----------|
| Mistral 7B | `mistral.mistral-7b-instruct-v0:2` | Efficient, multilingual |
| Mistral Large | `mistral.mistral-large-2402-v1:0` | Complex reasoning |
| Cohere Command R+ | `cohere.command-r-plus-v1:0` | RAG-optimized |
| Stable Diffusion XL | `stability.stable-diffusion-xl-v1` | Image generation |

> 💡 **Tip:** Always check the [Bedrock docs](https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html) for the latest model IDs — they update frequently.

### Inference Parameters (Key Settings)

```python
inferenceConfig = {
    "maxTokens": 1024,      # Max tokens in the response
    "temperature": 0.7,     # 0.0 = deterministic, 1.0 = creative/random
    "topP": 0.9,            # Nucleus sampling — controls diversity
    "stopSequences": ["\n\nHuman:"]  # Stop generation at these strings
}
```

```
Temperature Guide:
0.0  → Same answer every time. Best for: factual Q&A, classification
0.3  → Mostly consistent. Best for: structured output, data extraction
0.7  → Balanced. Best for: general chat, summaries
1.0  → Very creative/varied. Best for: brainstorming, creative writing
```

### Bedrock Pricing Tiers

Bedrock uses **on-demand pricing** (pay per token) and **Provisioned Throughput** (pay per hour):

#### On-Demand Pricing (Pay Per Token)

```
Pricing is per 1,000 input tokens + per 1,000 output tokens.
Approximate pricing (verify at aws.amazon.com/bedrock/pricing):

Model                    Input (per 1K tokens)   Output (per 1K tokens)
─────────────────────────────────────────────────────────────────────
Claude 3 Haiku           $0.00025                $0.00125
Claude 3 Sonnet          $0.003                  $0.015
Claude 3 Opus            $0.015                  $0.075
Titan Text Lite          $0.0003                 $0.0004
Titan Text Express       $0.0008                 $0.0016
Titan Text Premier       $0.0005                 $0.0015
Llama 3.1 8B             $0.00022                $0.00022
Llama 3.1 70B            $0.00099                $0.00099

Rule of thumb: ~750 words ≈ 1,000 tokens
```

#### Provisioned Throughput

```
Use when:
├── Consistent high-volume workloads
├── Required for custom fine-tuned models
├── Need guaranteed capacity (no throttling)
└── Potentially cheaper at very high volume

Pricing: Per Model Unit (MU) per hour
         Commitment: No commitment / 1 month / 6 months
         Longer commitment = lower hourly rate
```

#### Batch Inference (Lowest Cost Option)

```
Use when:
├── Processing large datasets (thousands of documents)
├── Not time-sensitive (async processing)
└── Want up to 50% cost reduction vs on-demand

How it works:
  1. Upload JSONL file of prompts to S3
  2. Submit batch inference job
  3. Results written back to S3
  4. Pay ~50% less per token
```

```python
# Submit a batch job
response = bedrock.create_model_invocation_job(
    jobName='batch-summarization-job',
    modelId='anthropic.claude-3-haiku-20240307-v1:0',
    roleArn='arn:aws:iam::123456789:role/BedrockBatchRole',
    inputDataConfig={
        's3InputDataConfig': {
            's3Uri': 's3://my-bucket/batch-inputs/prompts.jsonl',
            's3InputFormat': 'JSONLines'
        }
    },
    outputDataConfig={
        's3OutputDataConfig': {
            's3Uri': 's3://my-bucket/batch-outputs/'
        }
    }
)
```

---

## AWS Titan Models

Amazon Titan is AWS's own family of foundation models, built and maintained by Amazon. They are only available through Amazon Bedrock.

### Titan Model Family

```
┌─────────────────────────────────────────────────────────────────┐
│                   AWS TITAN MODEL FAMILY                         │
│                                                                  │
│  ┌──────────────────────┐   ┌──────────────────────────────┐   │
│  │    TITAN TEXT        │   │      TITAN EMBEDDINGS        │   │
│  │                      │   │                              │   │
│  │  • Express (fast)    │   │  • Text V1 (1536 dims)       │   │
│  │  • Lite (cheapest)   │   │  • Text V2 (1024/512/256)    │   │
│  │  • Premier (quality) │   │  • Multimodal (image+text)   │   │
│  └──────────────────────┘   └──────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────┐                                       │
│  │   TITAN IMAGE        │                                       │
│  │                      │                                       │
│  │  • Generator V1      │                                       │
│  │  • Generator V2      │                                       │
│  └──────────────────────┘                                       │
└─────────────────────────────────────────────────────────────────┘
```

### Titan Text Models — When to Use Each

| Model | Context Window | Best For | Avoid When |
|-------|---------------|----------|-----------|
| **Titan Text Lite** | 4K tokens | Simple tasks, high-volume, cost-sensitive | Complex reasoning needed |
| **Titan Text Express** | 8K tokens | General Q&A, summarization, chatbots | Very long documents |
| **Titan Text Premier** | 32K tokens | RAG applications, complex prompts, enterprise | Budget is primary concern |

```python
# Invoking Titan Text Premier
response = bedrock_runtime.invoke_model(
    modelId='amazon.titan-text-premier-v1:0',
    body=json.dumps({
        "inputText": "Summarize the key points of RAG in 3 bullet points.",
        "textGenerationConfig": {
            "maxTokenCount": 512,
            "temperature": 0.5,
            "topP": 0.9,
            "stopSequences": []
        }
    })
)

result = json.loads(response['body'].read())
print(result['results'][0]['outputText'])
```

### Titan Embeddings — For RAG Pipelines

Titan Embeddings converts text to dense vectors for similarity search:

```python
# Generate embeddings with Titan Embed Text V2
response = bedrock_runtime.invoke_model(
    modelId='amazon.titan-embed-text-v2:0',
    body=json.dumps({
        "inputText": "What is the return policy for electronics?",
        "dimensions": 1024,        # Options: 1024, 512, 256
        "normalize": True          # Normalize vectors for cosine similarity
    })
)

result = json.loads(response['body'].read())
embedding = result['embedding']    # List of 1024 floats
print(f"Embedding dimensions: {len(embedding)}")
```

**Titan Embeddings V2 advantages over V1:**
```
V1: Fixed 1536 dimensions
V2: Flexible dimensions (1024 / 512 / 256)
    └── Smaller dimensions = lower storage cost + faster search
        at slight quality trade-off

V2 also supports:
├── Better multilingual performance
└── Improved performance on RAG benchmarks
```

### Titan Image Generator

```python
import base64

# Generate an image with Titan Image Generator V2
response = bedrock_runtime.invoke_model(
    modelId='amazon.titan-image-generator-v2:0',
    body=json.dumps({
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {
            "text": "A professional diagram of cloud architecture on AWS, clean, technical style",
            "negativeText": "blurry, cartoon, low quality"
        },
        "imageGenerationConfig": {
            "numberOfImages": 1,
            "height": 1024,
            "width": 1024,
            "cfgScale": 8.0,        # How closely to follow the prompt (1-35)
            "seed": 42              # For reproducibility
        }
    })
)

result = json.loads(response['body'].read())
image_data = base64.b64decode(result['images'][0])

with open('generated_image.png', 'wb') as f:
    f.write(image_data)
```

### Why Choose Titan Over Other Models?

```
Reasons to use Titan:
✅ AWS-native — deep integration with all AWS services
✅ Data privacy — AWS commits no data used to train Titan from your inputs
✅ HIPAA eligible and supports compliance workloads
✅ Titan Embeddings is the default for Bedrock Knowledge Bases
✅ Fine-tunable via Bedrock customization
✅ No third-party data sharing

Reasons to consider other models:
→ Need best-in-class reasoning → Claude 3 Opus or GPT-4 class models
→ Need open weights for on-premise → Llama 3.1
→ Need specialized coding → Claude 3.5 Sonnet or CodeLlama
```

---

## Amazon SageMaker

Amazon SageMaker is AWS's end-to-end machine learning platform — for teams that need full control over model training, tuning, and deployment.

### SageMaker vs Bedrock

```
Use BEDROCK when:                    Use SAGEMAKER when:
─────────────────────────────────    ──────────────────────────────────
You want to USE foundation models    You want to TRAIN your own models
Quick to start, managed service      Full control over training pipeline
No ML infrastructure knowledge       Needs ML engineering expertise
RAG, chatbots, text generation       Custom CV, NLP, tabular ML models
Fine-tuning with simple datasets     Complex training with custom code
```

### SageMaker Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    AMAZON SAGEMAKER                              │
│                                                                  │
│  BUILD              TRAIN               DEPLOY                   │
│  ───────────────    ───────────────     ───────────────────────  │
│  Studio (IDE)       Training Jobs       Real-time Endpoints      │
│  Notebooks          Hyperparameter      Serverless Endpoints     │
│  Data Wrangler      Tuning (HPO)        Batch Transform          │
│  Feature Store      Distributed         Async Endpoints          │
│  Clarify (Bias)     Training            Multi-model Endpoints    │
│                     Experiments                                  │
│                     Pipelines           SPECIAL                  │
│                     Debugger            ─────────────────────    │
│                                         JumpStart (pre-built)    │
│                                         Canvas (no-code ML)      │
└─────────────────────────────────────────────────────────────────┘
```

### Training a Model on SageMaker

```python
import boto3
from sagemaker.estimator import Estimator
import sagemaker

session = sagemaker.Session()
role = sagemaker.get_execution_role()

# Define a training job using a built-in algorithm (XGBoost)
estimator = Estimator(
    image_uri=sagemaker.image_uris.retrieve('xgboost', 'us-east-1', '1.7-1'),
    instance_type='ml.m5.xlarge',
    instance_count=1,
    role=role,
    output_path='s3://my-bucket/model-output/',
    hyperparameters={
        'objective': 'binary:logistic',
        'num_round': 100,
        'max_depth': 5,
        'eta': 0.2
    }
)

# Point to training data in S3
estimator.fit({
    'train': 's3://my-bucket/data/train/',
    'validation': 's3://my-bucket/data/validation/'
})
```

### Deploying a Real-Time Endpoint

```python
# Deploy trained model to a real-time endpoint
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large',
    endpoint_name='my-classification-endpoint'
)

# Invoke the endpoint
response = predictor.predict([[5.1, 3.5, 1.4, 0.2]])
print(response)

# Clean up when done
predictor.delete_endpoint()
```

### Endpoint Types — When to Use Each

| Endpoint Type | Latency | Cost | Best For |
|---------------|---------|------|----------|
| **Real-time** | Milliseconds | Pay per hour (always on) | Live user-facing apps |
| **Serverless** | Seconds (cold start) | Pay per inference | Sporadic/unpredictable traffic |
| **Async** | Minutes | Pay per inference | Long-running predictions, batch |
| **Batch Transform** | Minutes–hours | Pay per instance-hour | Offline bulk scoring |
| **Multi-model** | Milliseconds | Pay per hour (shared) | Many models, low per-model traffic |

### SageMaker JumpStart

JumpStart is a model hub within SageMaker — pre-built models and solutions you can deploy with one click or a few lines of code:

```
JumpStart includes:
├── 300+ pre-trained models
│   ├── Computer vision (object detection, classification)
│   ├── NLP (text classification, summarization, translation)
│   ├── Tabular (regression, fraud detection)
│   └── Foundation models (Llama, Falcon, Stable Diffusion)
│
├── Pre-built solution templates
│   ├── Demand forecasting
│   ├── Fraud detection
│   └── Predictive maintenance
│
└── Fine-tuning notebooks for each model
```

```python
from sagemaker.jumpstart.model import JumpStartModel

# Deploy Llama 3.1 8B from JumpStart (no training needed)
model = JumpStartModel(model_id="meta-textgeneration-llama-3-1-8b-instruct")

predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.2xlarge",   # GPU instance required for LLMs
    accept_eula=True
)

response = predictor.predict({
    "inputs": "What are the benefits of cloud computing?",
    "parameters": {"max_new_tokens": 256, "temperature": 0.7}
})
print(response[0]['generated_text'])
```

### SageMaker Instance Types

```
For Training:
├── ml.m5.xlarge         General purpose (tabular, small NLP)
├── ml.c5.4xlarge        Compute-optimized (CPU-intensive training)
├── ml.p3.2xlarge        GPU (deep learning, LLM fine-tuning)
└── ml.p4d.24xlarge      8x A100 GPUs (large model training)

For Inference:
├── ml.t3.medium         Dev/test, very low traffic
├── ml.m5.large          Standard inference
├── ml.c5.xlarge         CPU-optimized inference
├── ml.g4dn.xlarge       GPU inference (fast LLM serving)
└── ml.inf2.xlarge       AWS Inferentia2 (cheapest GPU inference)

For LLM Fine-tuning (common choices):
├── ml.g5.2xlarge        Single A10G GPU, 24GB VRAM (~7B models)
├── ml.g5.12xlarge       4x A10G GPU, 96GB VRAM (~13B-70B models)
└── ml.p4d.24xlarge      8x A100 GPU, 320GB VRAM (largest models)
```

---

## Amazon Q

Amazon Q is AWS's AI assistant — a generative AI-powered helper for business and developer productivity.

### Amazon Q Variants

```
┌─────────────────────────────────────────────────────────────────┐
│                        AMAZON Q                                  │
│                                                                  │
│  ┌────────────────────────┐   ┌──────────────────────────────┐  │
│  │   Amazon Q Business    │   │    Amazon Q Developer        │  │
│  │                        │   │                              │  │
│  │  Enterprise chatbot     │   │  AI coding assistant         │  │
│  │  connected to your      │   │  integrated into IDEs        │  │
│  │  company's data         │   │  and AWS services            │  │
│  │                        │   │                              │  │
│  │  • Connects to 40+     │   │  • Code generation           │  │
│  │    data sources         │   │  • Security scanning         │  │
│  │  • Respects permissions │   │  • Code transformation       │  │
│  │  • Summarizes docs      │   │  • CLI assistant             │  │
│  │  • Creates tickets      │   │  • AWS Console chat          │  │
│  └────────────────────────┘   └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Amazon Q Business

Amazon Q Business connects to your organization's data and lets employees ask natural language questions:

```
Supported data connectors (40+):
├── Atlassian Confluence & Jira
├── Microsoft SharePoint & Teams
├── Salesforce
├── ServiceNow
├── Google Drive & Gmail
├── Amazon S3
├── Slack
└── Custom web crawlers

Permission-aware:
If User A can't see a document in SharePoint,
Amazon Q won't surface it in their answers.
```

**Use case example:**
```
Employee asks: "What is our policy on remote work for contractors?"

Amazon Q Business:
1. Searches connected sources (SharePoint, Confluence, S3)
2. Finds relevant HR policy documents
3. Returns grounded answer with citations
4. Respects document-level permissions
```

### Amazon Q Developer

Q Developer is the AI coding assistant embedded into AWS services and IDEs:

```
Available in:
├── AWS Console (chat about your AWS resources)
├── VS Code (via extension)
├── JetBrains IDEs
├── AWS CLI
└── Amazon CodeCatalyst

Capabilities:
├── Inline code suggestions (like GitHub Copilot)
├── Chat about your codebase
├── Security vulnerability scanning
├── Code transformation (Java 8 → Java 17, etc.)
├── Unit test generation
└── /dev task: implement a full feature from a description
```

### Amazon Q Pricing

```
Amazon Q Business:
├── Lite plan: $3/user/month (basic Q&A)
└── Pro plan:  $20/user/month (full features, document creation)

Amazon Q Developer:
├── Free tier: Basic code suggestions in IDEs
└── Pro plan:  $19/user/month (full features, security scans, /dev)
```

---

## Amazon Kendra

Amazon Kendra is an **intelligent enterprise search** service powered by ML. It searches across your documents and returns precise answers — not just links.

### Kendra vs Traditional Search vs RAG

```
Traditional keyword search:
  Query: "how many vacation days do I get?"
  Result: Links to 15 documents that contain those words
  User: "Which one has my answer?"  ← Frustrating

Kendra:
  Query: "how many vacation days do I get?"
  Result: "Full-time employees receive 15 days of PTO per year."
           Source: HR Handbook, Page 24
  ← Direct answer with source citation ✅

RAG with Bedrock:
  Similar to Kendra but integrated with LLM for more conversational,
  synthesized responses from multiple sources
```

### Kendra Key Features

```
Smart ranking:
├── Understands context and intent (not just keywords)
├── Learns from user feedback over time
└── Returns direct answers from within documents

Connectors (40+ data sources):
├── Amazon S3
├── SharePoint
├── Salesforce
├── ServiceNow
├── RDS / Aurora databases
├── Google Drive
└── Custom web crawler

Document types supported:
├── PDF, Word, PowerPoint, HTML
├── Structured: CSV, JSON
└── FAQs (explicit Q&A pairs for highest confidence answers)
```

### Setting Up Kendra

```python
kendra = boto3.client('kendra', region_name='us-east-1')

# 1. Create an index
index = kendra.create_index(
    Name='company-knowledge-index',
    Edition='ENTERPRISE_EDITION',    # or DEVELOPER_EDITION
    RoleArn='arn:aws:iam::123456789:role/KendraRole',
    Description='Search across company documents'
)
index_id = index['Id']

# 2. Add an S3 data source
data_source = kendra.create_data_source(
    Name='s3-documents',
    IndexId=index_id,
    Type='S3',
    Configuration={
        'S3Configuration': {
            'BucketName': 'my-company-docs',
            'InclusionPrefixes': ['hr/', 'policies/', 'procedures/']
        }
    },
    RoleArn='arn:aws:iam::123456789:role/KendraS3Role',
    Schedule='cron(0 12 * * ? *)'  # Sync daily at noon
)

# 3. Trigger a sync
kendra.start_data_source_sync_job(
    Id=data_source['Id'],
    IndexId=index_id
)

# 4. Query Kendra
response = kendra.query(
    IndexId=index_id,
    QueryText='What is the expense reimbursement policy?'
)

# Top result
top_result = response['ResultItems'][0]
print(f"Answer: {top_result.get('DocumentExcerpt', {}).get('Text')}")
print(f"Source: {top_result['DocumentURI']}")
print(f"Confidence: {top_result['ScoreAttributes']['ScoreConfidence']}")
```

### Kendra Pricing

```
Developer Edition:  $810/month  (testing and development)
Enterprise Edition: $1,008/month (production)
+ $0.00025 per query after first 40,000/month
+ Connector costs vary by source type
```

### When to Use Kendra vs Bedrock Knowledge Bases

| Factor | Amazon Kendra | Bedrock Knowledge Bases |
|--------|--------------|------------------------|
| **Response style** | Direct excerpts from docs | LLM-generated conversational answers |
| **Setup complexity** | Medium (managed connectors) | Low (fully managed) |
| **Customization** | Fine-grained relevance tuning | Prompt-level customization |
| **Compliance** | HIPAA eligible, FedRAMP | HIPAA eligible |
| **Best for** | Enterprise search portal | Conversational AI assistant |

---

## Amazon Comprehend

Amazon Comprehend is a natural language processing (NLP) service that extracts insights from text — no ML knowledge required.

### What Comprehend Can Do

```
Input: "I absolutely loved the new iPhone! The camera is amazing,
        but the battery life is disappointing. Bought it at the
        Apple Store in Seattle yesterday."

Comprehend extracts:

├── SENTIMENT:      Overall: MIXED
│                   Positive: 0.62, Negative: 0.31, Neutral: 0.07
│
├── ENTITIES:
│   ├── iPhone       → COMMERCIAL_ITEM
│   ├── camera       → OTHER
│   ├── Apple Store  → ORGANIZATION
│   └── Seattle      → LOCATION
│
├── KEY PHRASES:    "amazing camera", "battery life", "Apple Store"
│
├── LANGUAGE:       English (confidence: 0.99)
│
└── SYNTAX:         POS tags for each word (noun, verb, adjective...)
```

### Comprehend API Examples

```python
comprehend = boto3.client('comprehend', region_name='us-east-1')

text = "The patient has been diagnosed with Type 2 diabetes. Contact Dr. Smith at 555-1234."

# 1. Detect sentiment
sentiment = comprehend.detect_sentiment(
    Text=text,
    LanguageCode='en'
)
print(f"Sentiment: {sentiment['Sentiment']}")

# 2. Detect entities
entities = comprehend.detect_entities(
    Text=text,
    LanguageCode='en'
)
for entity in entities['Entities']:
    print(f"{entity['Text']} → {entity['Type']} ({entity['Score']:.2f})")

# 3. Detect PII (for privacy compliance)
pii = comprehend.detect_pii_entities(
    Text=text,
    LanguageCode='en'
)
for pii_entity in pii['Entities']:
    print(f"PII found: {pii_entity['Type']} at chars {pii_entity['BeginOffset']}-{pii_entity['EndOffset']}")
# Output: PII found: PHONE at chars 66-74

# 4. Custom classification (after training a custom classifier)
classification = comprehend.classify_document(
    Text="I want to cancel my subscription",
    EndpointArn='arn:aws:comprehend:us-east-1:123456789:document-classifier-endpoint/my-classifier'
)
print(f"Class: {classification['Classes'][0]['Name']}")
```

### Comprehend Use Cases

| Use Case | How Comprehend Helps |
|----------|---------------------|
| **Customer feedback analysis** | Sentiment analysis on reviews, NPS responses |
| **Support ticket routing** | Custom classification into ticket categories |
| **Content moderation** | Detect toxic content before publishing |
| **Compliance / PII detection** | Find and redact personal information |
| **Document processing** | Extract key entities from contracts, forms |
| **Social media monitoring** | Analyze brand sentiment at scale |

### Comprehend Medical

A specialized variant for healthcare text:

```python
comprehend_medical = boto3.client('comprehendmedical', region_name='us-east-1')

response = comprehend_medical.detect_entities_v2(
    Text="Patient has hypertension and takes Lisinopril 10mg daily."
)

for entity in response['Entities']:
    print(f"{entity['Text']} → {entity['Category']}/{entity['Type']}")

# Output:
# hypertension → MEDICAL_CONDITION/DX_NAME
# Lisinopril → MEDICATION/GENERIC_NAME
# 10mg → MEDICATION/DOSAGE
# daily → MEDICATION/FREQUENCY
```

---

## Amazon Rekognition

Amazon Rekognition is a computer vision service that analyzes images and videos — no ML expertise required.

### What Rekognition Can Do

```
┌─────────────────────────────────────────────────────────────────┐
│                   AMAZON REKOGNITION                             │
│                                                                  │
│  IMAGE ANALYSIS             VIDEO ANALYSIS                       │
│  ──────────────────────     ─────────────────────────────────   │
│  • Object detection         • Activity detection                 │
│  • Scene detection          • Person tracking                    │
│  • Face detection           • Celebrity recognition              │
│  • Face comparison          • Content moderation                 │
│  • Text in images (OCR)     • Stored video analysis             │
│  • Celebrity recognition    • Real-time streaming analysis       │
│  • Content moderation                                            │
│  • Custom labels            SPECIAL FEATURES                     │
│                             ─────────────────────────────────   │
│                             • Face Liveness detection            │
│                             • PPE detection                      │
│                             • Unsafe content detection           │
└─────────────────────────────────────────────────────────────────┘
```

### Rekognition Code Examples

```python
rekognition = boto3.client('rekognition', region_name='us-east-1')

# 1. Detect objects and scenes in an image
response = rekognition.detect_labels(
    Image={'S3Object': {'Bucket': 'my-bucket', 'Name': 'office.jpg'}},
    MaxLabels=10,
    MinConfidence=80.0
)
for label in response['Labels']:
    print(f"{label['Name']}: {label['Confidence']:.1f}%")
# Output:
# Person: 99.2%
# Computer: 98.7%
# Office: 95.1%
# Laptop: 94.3%

# 2. Detect text in an image (OCR)
response = rekognition.detect_text(
    Image={'S3Object': {'Bucket': 'my-bucket', 'Name': 'sign.jpg'}}
)
for text in response['TextDetections']:
    if text['Type'] == 'LINE':
        print(f"Text: {text['DetectedText']} ({text['Confidence']:.1f}%)")

# 3. Compare faces (verify identity)
response = rekognition.compare_faces(
    SourceImage={'S3Object': {'Bucket': 'my-bucket', 'Name': 'id-photo.jpg'}},
    TargetImage={'S3Object': {'Bucket': 'my-bucket', 'Name': 'selfie.jpg'}},
    SimilarityThreshold=90.0
)
if response['FaceMatches']:
    match = response['FaceMatches'][0]
    print(f"Face match: {match['Similarity']:.1f}% similar")
else:
    print("No match found")

# 4. Moderate content (detect inappropriate images)
response = rekognition.detect_moderation_labels(
    Image={'S3Object': {'Bucket': 'my-bucket', 'Name': 'user-upload.jpg'}},
    MinConfidence=60.0
)
for label in response['ModerationLabels']:
    print(f"Flagged: {label['Name']} (parent: {label['ParentName']})")
```

### Rekognition Custom Labels

For when the built-in labels don't match your specific use case:

```python
# After training a custom model in Rekognition Custom Labels console...

# Start the model (required before inference, billed per hour)
rekognition.start_project_version(
    ProjectVersionArn='arn:aws:rekognition:us-east-1:123456789:project/defect-detection/version/v1/...',
    MinInferenceUnits=1
)

# Use the custom model
response = rekognition.detect_custom_labels(
    ProjectVersionArn='arn:aws:rekognition:...',
    Image={'S3Object': {'Bucket': 'my-bucket', 'Name': 'product-image.jpg'}},
    MinConfidence=70.0
)
for label in response['CustomLabels']:
    print(f"{label['Name']}: {label['Confidence']:.1f}%")
# Output: Scratch_Defect: 94.2%
```

### Rekognition Use Cases

| Use Case | Rekognition Feature |
|----------|-------------------|
| **Identity verification** | Face comparison + Face Liveness |
| **User-generated content moderation** | Content moderation labels |
| **Workplace safety** | PPE detection |
| **Smart search of media libraries** | Object + celebrity detection |
| **Document processing** | Text detection (OCR) |
| **Manufacturing quality control** | Custom Labels |
| **Security surveillance** | Video person tracking |

---

## Choosing the Right Service

### Decision Tree

```
What are you trying to do?
          │
          ├── Understand or generate TEXT?
          │        │
          │        ├── Use a large language model (chat, summarize, generate)?
          │        │        ├── Yes, and I want managed + multiple model choices → Bedrock
          │        │        └── Yes, and I want full training control → SageMaker
          │        │
          │        ├── Analyze text (sentiment, entities, PII)?
          │        │        └── Amazon Comprehend
          │        │
          │        └── Search across documents with natural language?
          │                 ├── Conversational AI assistant → Bedrock Knowledge Bases
          │                 ├── Enterprise search portal → Amazon Kendra
          │                 └── Employee productivity chatbot → Amazon Q Business
          │
          ├── ANALYZE IMAGES or VIDEOS?
          │        └── Amazon Rekognition
          │
          ├── AI CODING ASSISTANT?
          │        └── Amazon Q Developer
          │
          └── BUILD A CUSTOM ML MODEL?
                   └── Amazon SageMaker
```

### Service Comparison at a Glance

| Service | Layer | ML Expertise | Primary Use Case |
|---------|-------|-------------|-----------------|
| **Amazon Bedrock** | Foundation Models | None–Low | LLM access, RAG, Agents, fine-tuning |
| **AWS Titan** | Foundation Models | None | AWS-native LLM and embeddings |
| **Amazon SageMaker** | ML Platform | High | Custom model training + deployment |
| **Amazon Q** | AI Service | None | Employee productivity, code assistance |
| **Amazon Kendra** | AI Service | None | Enterprise document search |
| **Amazon Comprehend** | AI Service | None | Text NLP (sentiment, entities, PII) |
| **Amazon Rekognition** | AI Service | None | Image and video analysis |

---

## Quick Reference

### Most Common Bedrock Model IDs (Exam Focus)

```
Claude 3 Haiku   → anthropic.claude-3-haiku-20240307-v1:0
Claude 3 Sonnet  → anthropic.claude-3-sonnet-20240229-v1:0
Claude 3 Opus    → anthropic.claude-3-opus-20240229-v1:0
Titan Text       → amazon.titan-text-express-v1
Titan Embeddings → amazon.titan-embed-text-v2:0
Llama 3.1 8B     → meta.llama3-1-8b-instruct-v1:0
Stable Diffusion → stability.stable-diffusion-xl-v1
```

### Key Differentiators (Exam Tips)

- **Bedrock** = managed FM access, no infrastructure, multiple providers
- **SageMaker** = full ML control, custom training, your own models
- **JumpStart** = pre-built models inside SageMaker (one-click deploy)
- **Kendra** = search returns excerpts from docs; **Bedrock KB** = LLM synthesizes answers
- **Comprehend** = text analysis (no generation); **Bedrock** = text generation
- **Rekognition** = images/video; **Comprehend** = text; **Transcribe** = audio
- **Amazon Q Business** = employee chatbot on company data; **Q Developer** = coding assistant
- **Titan Embeddings** = default embedding model for Bedrock Knowledge Bases
- Batch inference on Bedrock = up to 50% cheaper, async, S3 in/out
- Provisioned Throughput = required for custom/fine-tuned Bedrock models

### SageMaker Endpoint Types Summary

```
Real-time   → Always on, millisecond latency, pay per hour
Serverless  → Scales to zero, seconds latency, pay per call
Async       → Queue-based, minutes latency, long-running jobs
Batch       → Offline scoring, cheapest for bulk data
Multi-model → Many models on one endpoint, reduces cost
```

---

---

## ⚠️ Disclaimer & Important Notes

### Information Currency

This guide was written in **June 2026**. The AWS AI/ML landscape evolves rapidly — new models are released frequently, existing models are deprecated, pricing changes, and service features are added or modified. Specific areas most likely to change include:

- **Model IDs** — AWS regularly releases new model versions and retires older ones. Always check the [official model ID list](https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html).
- **Pricing** — All figures shown are approximate and for illustrative purposes only. Verify current pricing at [aws.amazon.com/bedrock/pricing](https://aws.amazon.com/bedrock/pricing).
- **API parameters** — Field names, supported values, and defaults may change between API versions.
- **Service availability** — Not all models and features are available in every AWS region.
- **Code examples** — SDK versions and method signatures may differ from what is shown here. Always refer to the [AWS SDK documentation](https://docs.aws.amazon.com/pythonsdk/) for the latest.

### Before Using This Guide

```
✅ Cross-reference model IDs with the official AWS Console or docs
✅ Verify pricing using the AWS Pricing Calculator
✅ Check service availability in your target AWS region
✅ Review the AWS changelog for recent feature updates
✅ Test all code examples in a development environment first
```

### Official AWS Resources

| Resource | URL |
|----------|-----|
| Bedrock Documentation | https://docs.aws.amazon.com/bedrock/ |
| SageMaker Documentation | https://docs.aws.amazon.com/sagemaker/ |
| Bedrock Pricing | https://aws.amazon.com/bedrock/pricing/ |
| SageMaker Pricing | https://aws.amazon.com/sagemaker/pricing/ |
| Bedrock Model IDs | https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html |
| AWS What's New | https://aws.amazon.com/new/ |
| AWS AI/ML Blog | https://aws.amazon.com/blogs/machine-learning/ |

### Legal

This document is an **independent study guide** created for educational purposes. It is **not affiliated with, endorsed by, or officially associated with Amazon Web Services (AWS)** or any of its subsidiaries.

All AWS product names (Amazon Bedrock, Amazon SageMaker, Amazon Titan, Amazon Q, Amazon Kendra, Amazon Comprehend, Amazon Rekognition, etc.), logos, and trademarks are the property of Amazon.com, Inc. or its affiliates.

---

*Written as part of the AWS Generative AI Developer 30-day study guide.*
*Last updated: June 2026*
*For corrections or updates, please open an issue or submit a pull request.*
