---
title: RAG Chatbot Saúde Itabira
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.0.0
app_file: app.py
pinned: false
license: mit
---

# RAG Chatbot Saúde Itabira

Prototype chatbot with **RAG (Retrieval-Augmented Generation)** and **LLM**
over information about COVID-19 and healthcare units in the city of Itabira/MG.

The code and documentation are in English; the chatbot itself (knowledge base,
prompts and interface) operates in **Portuguese**.

The corpus is loaded from a public dataset on Hugging Face Hub
(`HF_DATASET`), already split into chunks with source metadata. The app
contains no hardcoded texts — updating the corpus only requires regenerating
the JSONL and running `push_to_hub` on the dataset, then restarting the Space.

## Architecture

| Component | Where it lives |
|---|---|
| Source code | GitHub — this repository |
| RAG corpus | Hugging Face Dataset — [`lukinhapdm/corpus-rag-saude-itabira`](https://huggingface.co/datasets/lukinhapdm/corpus-rag-saude-itabira) |
| Chat interface | Hugging Face Space — `lukinhapdm/rag-chatbot-saude-itabira`, synced from GitHub on every push to `main` |

## How it works

1. **Indexing** — at startup, the dataset chunks are vectorized with
   `gemini-embedding-001` (task type `RETRIEVAL_DOCUMENT`, with the parent
   document title) and L2-normalized.
2. **Query rewriting** — follow-up questions are rewritten by the LLM into
   self-contained queries using the conversation history.
3. **Retrieval** — the query is vectorized with `RETRIEVAL_QUERY`; top-k
   chunks by cosine similarity, filtered by a relevance threshold. Without
   relevant context, the LLM is not called.
4. **Augmentation** — the retrieved chunks are injected into the prompt with
   their sources and scores.
5. **Generation** — Gemini streams the answer with few-shot examples,
   restrictive grounding, an explicit citation rule and a short conversation
   memory. Overloaded or retired models fall back to the next one in the list.
6. **Post-generation** — *faithfulness* check: phone numbers and sources must
   be grounded in the retrieved context.

Comments marked with 🧠 in `app.py` point out where each AI concept is applied.

## Repository structure

```
├── .github/workflows/sync-to-huggingface.yaml   # GitHub → Space sync
├── app.py                                       # Space application (Gradio)
├── requirements.txt
└── google-colab-notebook/chat_covid19.ipynb     # standalone didactic version (Colab)
```

## Setup

1. **Space:** create the Space `lukinhapdm/rag-chatbot-saude-itabira` (SDK
   Gradio, CPU basic hardware is enough — no GPU is needed, since all models
   run through the Gemini API).
2. **Space secret:** in the Space, go to *Settings → Variables and secrets* and
   add the secret `GOOGLE_API_KEY` with your Gemini API key.
3. **GitHub secret:** create a Hugging Face token with *write* permission and
   add it to this repository as the Actions secret `HF_TOKEN`
   (*Settings → Secrets and variables → Actions*).
4. Push to `main` (or run the workflow manually in the *Actions* tab). The
   [`huggingface/hub-sync`](https://huggingface.co/docs/hub/repositories-github-actions)
   action mirrors the repository files to the Space, excluding `.github/`.

### Optional Space variables

| Variable | Default | Purpose |
|---|---|---|
| `HF_DATASET` | `lukinhapdm/corpus-rag-saude-itabira` | Dataset with the RAG corpus |
| `SIMILARITY_THRESHOLD` | `0.55` | Minimum cosine similarity for a chunk to be used |
| `GENERATION_MODELS` | see `app.py` | Comma-separated generation models, in order of preference |

### Calibrating the threshold

Every question writes a line like the one below to the Space logs
(*Logs → Container*):

```
[Retrieval] 'quais são os grupos de risco da covid-19?' -> doc_004_c000=0.742, doc_001_c000=0.651, ...
```

Ask a few questions the corpus covers and a few that it doesn't, then set
`SIMILARITY_THRESHOLD` between the two groups of scores. Note that these logs
contain the users' questions.

## Running locally

```bash
pip install -r requirements.txt
export GOOGLE_API_KEY=...   # never commit this value
python app.py
```

## Limitations

- **Educational prototype.** Do not use for clinical decisions or in real
  healthcare situations.
- Small corpus (5 documents, 15 chunks). In production, it requires more
  sophisticated chunking, reranking, and quantitative evaluation (RAGAS).
- The index is rebuilt in memory at every Space restart (one embedding call
  per chunk). For a larger corpus, store the vectors in a vector database.

## Stack

- [Gradio](https://gradio.app) — chat interface
- [google-genai](https://ai.google.dev/gemini-api/docs) — unified Gemini SDK
- [datasets](https://huggingface.co/docs/datasets) — corpus loading
- [Hugging Face Spaces](https://huggingface.co/docs/hub/spaces) — hosting (free CPU tier)
