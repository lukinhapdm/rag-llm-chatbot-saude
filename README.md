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

Protótipo de chatbot com **RAG (Retrieval-Augmented Generation)** e **LLM**
sobre informações de COVID-19 e unidades de saúde da cidade de Itabira/MG.

O corpus é carregado de um dataset público no Hugging Face Hub
(`HF_DATASET`), já fragmentado em chunks com metadados de fonte. O app não
contém textos hardcoded — atualizar o corpus é só regenerar o JSONL e fazer
`push_to_hub` no dataset.

## Como funciona

1. **Indexação** — os chunks do dataset são vetorizados com
   `gemini-embedding-001` (task_type `RETRIEVAL_DOCUMENT`).
2. **Retrieval** — a pergunta (eventualmente reescrita para se tornar
   autocontida) é vetorizada com `RETRIEVAL_QUERY`; top-k por similaridade
   de cosseno com corte de score.
3. **Augmentation** — o contexto recuperado é injetado no prompt com
   metadados de fonte e score.
4. **Generation** — Gemini Flash gera a resposta com grounding restritivo
   e regra explícita de citação.
5. **Pós-geração** — checagem de *faithfulness*: telefones e fontes com
   lastro no contexto recuperado.

## Dataset

O corpus fica em um repositório separado no HF Hub
(`lukinhapdm/corpus-rag-saude-itabira`). Para apontar para outro dataset,
configure a variável `HF_DATASET` no Space.

## Limitações

- **Protótipo com finalidade educacional.** Não utilizar para decisões
  clínicas ou em situações reais de saúde.
- Corpus pequeno (5 documentos, ~14 chunks). Em produção, exige chunking
  mais sofisticado, reranking e avaliação quantitativa (RAGAS).

## Stack

- [Gradio](https://gradio.app) — interface de chat
- [google-genai](https://ai.google.dev/gemini-api/docs) — SDK unificado Gemini
- [datasets](https://huggingface.co/docs/datasets) — carregamento do corpus
- Hugging Face Spaces — hospedagem (ZeroGPU, tier gratuito)
