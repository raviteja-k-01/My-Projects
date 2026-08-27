# Enterprise RAG Assistant

A production-style Retrieval-Augmented Generation (RAG) application for answering questions from enterprise policy documents with grounded responses and source citations.

I built this project to understand and implement a complete RAG pipeline. The system handles document ingestion, hybrid retrieval, context construction, answer generation, citations, API serving, containerization, and cloud deployment.

## What the project does

The application works on a collection of fictional enterprise policy documents covering areas such as HR, information security, privacy, access management, incident response, travel, vendor risk, and AI governance.

A user can ask a policy question such as:

> What must be recorded when an incident severity is reclassified?

The system retrieves the relevant evidence, generates an answer only from that evidence, and returns the supporting document, page, section, and chunk information.

## Architecture

```text
Enterprise PDFs
      │
      ▼
Text + Table Extraction
      │
      ▼
Structure Parsing & Normalization
      │
      ▼
Section-Aware Chunking
      │
      ├──────────────┐
      ▼              ▼
Gemini Embeddings   BM25
      │              │
      ▼              ▼
Qdrant Search     Sparse Search
      └──────┬───────┘
             ▼
     Reciprocal Rank Fusion
             │
             ▼
        Context Builder
             │
             ▼
      Gemini Generation
             │
             ▼
       Citation Builder
             │
             ▼
    FastAPI + Streamlit
```

## Knowledge Base

The current corpus contains:

| Metric | Count |
|---|---:|
| Documents | 12 |
| Pages | 1,304 |
| Extracted tables | 219 |
| Body chunks | 1,766 |
| Table chunks | 219 |
| Total retrieval chunks | 1,985 |

The ingestion pipeline preserves useful metadata such as document ID, page number, section, subsection, and table information so that retrieved evidence can be traced back to its original source.

## Retrieval

The system uses **hybrid retrieval** instead of relying on semantic search alone.

Dense retrieval uses Gemini embeddings stored in Qdrant, while BM25 handles keyword-sensitive searches such as policy IDs, exact terminology, section names, and acronyms.

The two result sets are combined using Reciprocal Rank Fusion:

```text
RRF score = Σ 1 / (60 + rank)
```

The highest-ranked chunks are then passed to the context builder, which applies token limits and duplicate filtering before generation.

## Grounded Generation and Citations

The generation layer uses Gemini through `langchain-google-genai`.

The prompt instructs the model to answer only from the supplied evidence and to reference evidence identifiers such as:

```text
[Evidence 1]
[Evidence 2, Evidence 4]
```

Those references are mapped back to deterministic source metadata after generation.

A returned citation can include:

```text
Document: SEC-IR-004
Page: 16
Section: 3.6 | Reclassification and audit trail
Chunk: SEC-IR-004-B0026
```

This makes it possible to verify where an answer came from instead of treating the LLM response as a black box.

## Evaluation

I evaluated the retrieval pipeline using a separate gold question set.

The hybrid baseline produced:

| Metric | Score |
|---|---:|
| MRR | 0.6402 |
| Hit@1 | 0.4828 |
| Hit@3 | 0.7931 |
| Hit@5 | 0.8621 |
| Hit@10 | 0.9655 |
| Hit@20 | 0.9655 |

I also tested a cross-encoder reranker using `cross-encoder/ms-marco-MiniLM-L6-v2`.

It reduced retrieval quality rather than improving it, including a drop in MRR from `0.6402` to `0.5564`.

Because of that result, reranking is not part of the final V1 serving pipeline.

I kept the experiment in the project because it reflects an important design principle behind this implementation: components are retained based on measured improvement, not simply because they are common in RAG architectures.

## Testing

The project includes automated tests for the main deterministic parts of the system:

- Reciprocal Rank Fusion
- context construction
- citation generation
- FastAPI endpoints
- API error handling

Current test result:

```text
16 passed
```

Run them with:

```bash
python -m pytest -v
```

## Tech Stack

| Area | Technology |
|---|---|
| Language | Python 3.11 |
| PDF processing | PyMuPDF |
| Embeddings | Gemini Embeddings |
| Vector database | Qdrant |
| Sparse retrieval | BM25S |
| Fusion | Reciprocal Rank Fusion |
| Generation | Gemini |
| LLM integration | LangChain Google GenAI |
| Backend | FastAPI |
| UI | Streamlit |
| Testing | Pytest |
| Containerization | Docker |
| Cloud deployment | Azure Container Apps |
| Container registry | Azure Container Registry |

## Project Structure

```text
enterprise-rag-assistant/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── scripts/
├── src/
│   ├── ingestion/
│   ├── indexing/
│   ├── retrieval/
│   ├── generation/
│   ├── rag/
│   ├── api/
│   └── ui/
│
├── tests/
├── Dockerfile
├── requirements.txt
└── README.md
```

## Running Locally

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_key
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_key
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the API:

```bash
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Then start Streamlit in another terminal:

```bash
streamlit run src/ui/app.py
```

The API health endpoint is available at:

```text
http://127.0.0.1:8000/health
```

and the Streamlit application runs at:

```text
http://127.0.0.1:8501
```

## Docker

Build the application image:

```bash
docker build -t enterprise-rag-app .
```

Run the FastAPI backend:

```bash
docker run --env-file .env -p 8000:8000 enterprise-rag-app
```

The same image is used for the Streamlit service with a different startup command.

This keeps the deployment simple while allowing the backend and UI to run as separate containers.

## Deployment

The application is deployed on Microsoft Azure using:

```text
Azure Container Registry
        │
        ▼
enterprise-rag-app
        │
   ┌────┴────┐
   ▼         ▼
FastAPI   Streamlit
Container  Container
   App       App
```

Azure Container Apps runs the API and UI independently.

The backend connects to Qdrant Cloud and the Gemini API, while Azure-managed identities are used to pull the Docker image from Azure Container Registry.

Secrets are supplied at runtime and are not stored inside the Docker image or committed to Git.

## Performance

During warm requests, the complete RAG pipeline typically responded in approximately **5–7 seconds**.

Retrieval generally completed in around **1–3 seconds**.

During testing, Gemini occasionally showed significantly higher generation latency and eventually hit API quota limits. The measurements showed that these latency spikes occurred primarily in the generation stage rather than in hybrid retrieval.

Azure scale-to-zero can also make the first request after inactivity slower because of container cold starts.

## Current Scope

V1 intentionally focuses on a reliable document-based RAG workflow.

Features such as agents, GraphRAG, multimodal document understanding, fine-tuning, and complex distributed infrastructure were left outside the current scope because they were not required to solve the core problem.

Possible future improvements include better observability, caching, larger evaluation sets, authentication, multimodal document processing, and further LLM latency benchmarking.

## Status

The complete V1 pipeline has been implemented and validated end-to-end:

**documents → ingestion → indexing → hybrid retrieval → grounded generation → citations → FastAPI → Streamlit → Docker → Azure**

The project is currently deployed and working end-to-end.
