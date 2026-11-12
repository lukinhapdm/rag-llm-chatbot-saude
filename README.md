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

The corpus is loaded from a public dataset on Hugging Face Hub
(`HF_DATASET`), already split into chunks with source metadata. The app
contains no hardcoded texts — updating the corpus only requires regenerating
the JSONL and running `push_to_hub` on the dataset.

## How it works

1. **Indexing** — the dataset chunks are vectorized with
   `gemini-embedding-001` (task_type `RETRIEVAL_DOCUMENT`).
2. **Retrieval** — the question (possibly rewritten to become
   self-contained) is vectorized with `RETRIEVAL_QUERY`; top-k by cosine
   similarity with a score cutoff.
3. **Augmentation** — the retrieved context is injected into the prompt with
   source metadata and score.
4. **Generation** — Gemini Flash generates the answer with restrictive
   grounding and an explicit citation rule.
5. **Post-generation** — *faithfulness* check: phone numbers and sources
   grounded in the retrieved context.

## Dataset

The corpus lives in a separate repository on HF Hub
(`lukinhapdm/corpus-rag-saude-itabira`). To point to another dataset,
set the `HF_DATASET` variable in the Space.

## Limitations

- **Educational prototype.** Do not use for clinical decisions or in real
  healthcare situations.
- Small corpus (5 documents, ~14 chunks). In production, it requires more
  sophisticated chunking, reranking, and quantitative evaluation (RAGAS).

## Stack

- [Gradio](https://gradio.app) — chat interface
- [google-genai](https://ai.google.dev/gemini-api/docs) — unified Gemini SDK
- [datasets](https://huggingface.co/docs/datasets) — corpus loading
- Hugging Face Spaces — hosting (ZeroGPU, free tier) - COMING SOON
