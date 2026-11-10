---
title: RAG LLM CHATBOT SAÚDE
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.0.0
app_file: app.py
pinned: false
license: mit
---

# RAG Saúde Itabira

Protótipo de chatbot **RAG (Retrieval-Augmented Generation)** sobre informações
de COVID-19 e unidades de saúde da cidade de Itabira/MG.

## Como funciona

1. **Indexação** — os documentos são fragmentados em chunks com overlap e
   vetorizados com `gemini-embedding-001` (task_type `RETRIEVAL_DOCUMENT`).
2. **Retrieval** — a pergunta (eventualmente reescrita para se tornar
   autocontida) é vetorizada com task_type `RETRIEVAL_QUERY`; os chunks mais
   similares são recuperados via similaridade por cosseno com corte de score.
3. **Augmentation** — o contexto recuperado é injetado no prompt junto com
   metadados de fonte e score.
4. **Generation** — o modelo Gemini Flash gera a resposta com grounding
   restritivo (proibido adicionar informação fora do texto base) e citação
   obrigatória das fontes reais.
5. **Pós-geração** — checagem heurística de *faithfulness* (telefones e
   fontes com lastro no contexto).

## Limitações

- **Protótipo com finalidade educacional.** Não utilizar para decisões
  clínicas ou em situações reais de saúde.
- Corpus pequeno (5 documentos). Em produção, exige chunking mais sofisticado,
  reranking e avaliação quantitativa (RAGAS).

## Stack

- [Gradio](https://gradio.app) — interface de chat
- [google-genai](https://ai.google.dev/gemini-api/docs) — SDK unificado Gemini
- Hugging Face Spaces — hospedagem (ZeroGPU, tier gratuito)
