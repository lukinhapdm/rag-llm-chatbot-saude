"""RAG chatbot about Covid-19 and the healthcare network of Itabira (MG, Brazil).

Pipeline: indexing -> query rewriting -> retrieval -> augmentation ->
generation (streaming) -> faithfulness check.

The code and comments are in English; everything the end user sees
(knowledge base, system instruction, prompts and UI) is in Portuguese.
Comments marked with 🧠 point out where each AI concept is applied.
"""

import os
import random
import re
import time

import gradio as gr
import numpy as np
from datasets import load_dataset
from google import genai
from google.genai import errors, types


# CONFIGURATION
# Values can be overridden with Space variables (Settings -> Variables).


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY is not set. Add it as a Secret in the Space "
        "(Settings -> Variables and secrets) and restart the Space."
    )

HF_DATASET = os.getenv("HF_DATASET", "lukinhapdm/corpus-rag-saude-itabira")

# 🧠 Embedding models: the first one available is used for both indexing and queries
EMBEDDING_MODELS = ["models/gemini-embedding-001"]

# 🧠 Generative LLMs: preference order used for fallback when a model is overloaded or retired
GENERATION_MODELS = os.getenv(
    "GENERATION_MODELS",
    "models/gemini-3.5-flash-lite,models/gemini-3.1-flash-lite,models/gemini-3.5-flash,"
    "models/gemini-3.8-flash,models/gemini-3.6-flash,models/gemini-3.7-flash",
).split(",")

TOP_K = 3  # 🧠 Top-k retrieval: number of chunks that may be injected into the prompt
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.55"))  # 🧠 Relevance threshold: minimum cosine similarity
MAX_HISTORY_MESSAGES = 6  # 🧠 Context window: previous messages used as conversation memory

MAX_RETRIES = 3
INITIAL_BACKOFF_S = 2.0
TRANSIENT_ERROR_CODES = {429, 500, 503, 504}
SKIP_MODEL_ERROR_CODES = {404}  # model listed by the API but no longer served

client = genai.Client(api_key=GOOGLE_API_KEY)


# RESILIENCE


def backoff_delay(attempt):
    return INITIAL_BACKOFF_S * 2 ** (attempt - 1) + random.uniform(0, 1)


def with_retry(func, *args, **kwargs):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return func(*args, **kwargs)
        except errors.APIError as error:
            if error.code not in TRANSIENT_ERROR_CODES or attempt == MAX_RETRIES:
                raise
            delay = backoff_delay(attempt)
            print(f"[Retry] Error {error.code}, retrying in {delay:.1f}s ({attempt}/{MAX_RETRIES})")
            time.sleep(delay)


# MODELS


actions_by_model = {m.name: (m.supported_actions or []) for m in client.models.list()}

EMBEDDING_MODEL = next((m for m in EMBEDDING_MODELS if "embedContent" in actions_by_model.get(m, [])), None)
ACTIVE_MODELS = [m.strip() for m in GENERATION_MODELS if "generateContent" in actions_by_model.get(m.strip(), [])]

if not EMBEDDING_MODEL:
    raise RuntimeError(f"None of {EMBEDDING_MODELS} is available for this API key.")
if not ACTIVE_MODELS:
    available = sorted(n for n, a in actions_by_model.items() if "generateContent" in a)
    raise RuntimeError(f"None of GENERATION_MODELS is available. Models offered by the API: {available}")

print(f"[Embedding model]   {EMBEDDING_MODEL}")
print(f"[Generation models] {ACTIVE_MODELS}")


# KNOWLEDGE BASE


# 🧠 Knowledge base: corpus already split into chunks (id, titulo, texto, fonte, documento_origem)
corpus = load_dataset(HF_DATASET, split="train").to_pandas()
corpus["fontes"] = corpus["fonte"].apply(lambda s: [f.strip() for f in str(s).split(", ") if f.strip()])
print(f"[Corpus] {len(corpus)} chunks loaded from {HF_DATASET}")


# EMBEDDINGS AND INDEXING


def embed_text(text, task_type, title=None):
    response = with_retry(
        client.models.embed_content,
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type=task_type, title=title),  # 🧠 Asymmetric embedding: different task types for documents and queries
    )
    vector = np.asarray(response.embeddings[0].values, dtype=np.float32)
    return vector / np.linalg.norm(vector)  # 🧠 L2 normalization: the dot product becomes cosine similarity


EMBEDDING_MATRIX = np.stack([  # 🧠 Indexing: the corpus becomes a vector index (chunks x dimensions)
    embed_text(row.texto, "RETRIEVAL_DOCUMENT", title=row.titulo)  # 🧠 The parent document title anchors each chunk semantically
    for row in corpus.itertuples()
])
print(f"[Index] {EMBEDDING_MATRIX.shape[0]} vectors of {EMBEDDING_MATRIX.shape[1]} dimensions")


# RETRIEVAL


def retrieve_context(query):
    scores = EMBEDDING_MATRIX @ embed_text(query, "RETRIEVAL_QUERY")  # 🧠 Vector search: compares the query with every chunk
    ranking = np.argsort(scores)[::-1][:TOP_K]  # 🧠 Top-k retrieval: the most similar chunks first

    print(f"[Retrieval] '{query}' -> " + ", ".join(f"{corpus.iloc[i]['id']}={scores[i]:.3f}" for i in ranking))

    return [
        {
            "chunk_id": corpus.iloc[i]["id"],
            "title": corpus.iloc[i]["titulo"],
            "content": corpus.iloc[i]["texto"],
            "sources": list(corpus.iloc[i]["fontes"]),
            "score": float(scores[i]),
        }
        for i in ranking
        if scores[i] >= SIMILARITY_THRESHOLD  # 🧠 Relevance threshold: discards unrelated context
    ]


# GENERATION SETTINGS


SAFETY_SETTINGS = [  # 🧠 Safety filters: block harmful content with medium or high probability
    types.SafetySetting(category=category, threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE)
    for category in (
        types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
    )
]

# 🧠 System instruction + few-shot prompting + restrictive grounding + citation rule (sources copied only from the "Fontes:" header)
SYSTEM_INSTRUCTION = """
Utilizando como base os exemplos de perguntas e respostas abaixo, responda à pergunta atual utilizando o texto base num formato organizado, separando as informações em tópicos para destaque, incluindo as fontes do texto base, mas sem adicionar informações que não fazem parte do texto base.

REGRA DE CITAÇÃO DE FONTES (obrigatória):
- Copie APENAS as fontes listadas no campo "Fontes:" do cabeçalho de cada documento do texto base.
- NUNCA use o título do documento como se fosse fonte.
- NUNCA deduza a fonte a partir do conteúdo; se o cabeçalho não trouxer "Fontes:", escreva "Fonte: não informada no texto base".
- Preserve a esfera administrativa (Municipal, Estadual, Federal) exatamente como aparece.

EXEMPLO DE PERGUNTA 1: O que é a Covid?
EXEMPLO DE RESPOSTA 1:
* **COVID-19:** Doença causada pelo coronavírus SARS-CoV-2.
* **Espectro Clínico:** De infecções assintomáticas a quadros graves.
* **Pacientes (OMS):**
  * ~80% assintomáticos ou oligossintomáticos.
  * ~20% requerem hospitalização.
  * ~5% dos hospitalizados necessitam de suporte ventilatório.
* **Fontes:** Secretaria de Estado de Saúde de Minas Gerais, Organização Mundial de Saúde (OMS).

EXEMPLO DE PERGUNTA 2: Quais são as UBS perto do bairro São Bento?
EXEMPLO DE RESPOSTA 2:
* **Região do Bairro São Bento:** Região 3
* **Unidades de Saúde da Região 3:**
  * UBS Amazonas (3839-2225)
  * UBS Clóvis Alvim (3839-2165)
  * UBS Major Lage (3839-2324)
  * UBS Areão (3839-2482)
  * UBS Água Fresca (3839-2222)
  * UBS Machado (3839-2497)
  * UBS Juca Batista (3839-2495)
* **Fontes:** Secretaria Municipal de Saúde de Itabira, Prefeitura de Itabira.
"""

GENERATION_CONFIG = types.GenerateContentConfig(
    temperature=0.5,  # 🧠 Temperature: controls randomness when choosing each token
    top_p=0.9,  # 🧠 Nucleus sampling: samples only from tokens adding up to 90% probability
    max_output_tokens=2048,
    safety_settings=SAFETY_SETTINGS,
    system_instruction=SYSTEM_INSTRUCTION,
)

REWRITE_CONFIG = types.GenerateContentConfig(
    temperature=0.0,  # 🧠 Deterministic decoding: rewriting must be faithful, not creative
    max_output_tokens=256,
    safety_settings=SAFETY_SETTINGS,
)


# CONVERSATION HISTORY


ALERT_PREFIXES = ("\n\n> ⚠️", "\n\n> ℹ️")


def extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return content.get("text") or ""
    if isinstance(content, (list, tuple)):
        return " ".join(extract_text(part) for part in content)
    return ""


def convert_history(gradio_history):
    messages = []
    for item in gradio_history:
        role, text = item.get("role"), extract_text(item.get("content")).strip()
        for prefix in ALERT_PREFIXES:
            text = text.split(prefix)[0]
        if text and role in ("user", "assistant"):
            messages.append(types.Content(
                role="model" if role == "assistant" else "user",
                parts=[types.Part(text=text)],
            ))

    messages = messages[-MAX_HISTORY_MESSAGES:]  # 🧠 Short-term memory: a sliding window caps the tokens sent
    while messages and messages[0].role == "model":
        messages.pop(0)
    return messages


# QUERY REWRITING


def rewrite_query(question, history):
    if not history:
        return question

    transcript = "\n".join(f"{m.role}: {m.parts[0].text}" for m in history)
    prompt = (  # 🧠 Query rewriting: turns follow-ups ("e o telefone?") into self-contained search queries
        "Reescreva a NOVA PERGUNTA como uma consulta AUTOCONTIDA, "
        "substituindo pronomes e referências implícitas por termos explícitos. "
        "Responda APENAS a consulta reescrita.\n\n"
        f"HISTÓRICO:\n{transcript}\n\nNOVA PERGUNTA: {question}\n\nCONSULTA REESCRITA:"
    )
    for model in ACTIVE_MODELS:
        try:
            response = with_retry(client.models.generate_content, model=model, contents=prompt, config=REWRITE_CONFIG)
            return (response.text or "").strip() or question
        except errors.APIError as error:
            if error.code not in TRANSIENT_ERROR_CODES | SKIP_MODEL_ERROR_CODES:
                break
    return question  # silent fallback: the original question still works for most cases


# GENERATION (STREAMING WITH MODEL FALLBACK)


def stream_answer(contents):
    last_error = None
    for model in ACTIVE_MODELS:
        for attempt in range(1, MAX_RETRIES + 1):
            partial = ""
            try:
                stream = client.models.generate_content_stream(  # 🧠 LLM inference: autoregressive generation streamed token by token
                    model=model, contents=contents, config=GENERATION_CONFIG,
                )
                for chunk in stream:
                    if chunk.text:
                        partial += chunk.text
                        yield partial
                return
            except errors.APIError as error:
                last_error = error
                if partial or error.code not in TRANSIENT_ERROR_CODES | SKIP_MODEL_ERROR_CODES:
                    raise
                if error.code in SKIP_MODEL_ERROR_CODES or attempt == MAX_RETRIES:
                    break
                time.sleep(backoff_delay(attempt))
        print(f"[Fallback] {model} unavailable ({last_error.code}), trying the next model")
    raise RuntimeError("All generation models are unavailable.") from last_error


# FAITHFULNESS CHECK


PHONE_PATTERN = r"\b\d{4}-\d{4}\b"


def check_faithfulness(answer, context, expected_sources, expected_titles):  # 🧠 Faithfulness check: post-generation hallucination detection
    alerts, notices = [], []

    orphan_phones = set(re.findall(PHONE_PATTERN, answer)) - set(re.findall(PHONE_PATTERN, context))
    if orphan_phones:
        alerts.append(f"telefone(s) sem lastro no texto base: {sorted(orphan_phones)}")

    for block in re.findall(r"Fontes?:?\s*\**\s*([^.\n]+)", answer, re.I):
        block = re.sub(r"\*+", "", block).strip()
        for source in re.split(r"\s+e\s+|,|;", block):
            source = source.strip().rstrip(".")
            if len(source) < 8 or "não informada" in source.lower():
                continue
            if any(source.lower() in t.lower() or t.lower() in source.lower() for t in expected_titles):
                notices.append(f"título citado como fonte: '{source[:60]}'")
                continue
            if expected_sources and not any(
                source.lower() in s.lower() or s.lower() in source.lower() for s in expected_sources
            ):
                alerts.append(f"fonte não corresponde aos documentos recuperados: '{source[:60]}'")

    return alerts, notices


# CHAT ORCHESTRATION


NO_CONTEXT_MESSAGE = (
    "Não encontrei informação relevante na minha base para responder com segurança. "
    "Posso responder sobre a Covid-19 (definição, sintomas, o que fazer, grupos de risco) "
    "e sobre as unidades de saúde de Itabira por bairro."
)
EMPTY_ANSWER_MESSAGE = "Não foi possível gerar uma resposta para essa pergunta."
ERROR_MESSAGE = "⚠️ O serviço está indisponível no momento. Tente novamente em alguns minutos."


def chat_fn(message, history):
    question = extract_text(message).strip()
    if not question:
        yield "Digite uma pergunta."
        return

    try:
        memory = convert_history(history)
        query = rewrite_query(question, memory)

        results = retrieve_context(query)  # 🧠 RAG, retrieval: finds the most relevant chunks
        if not results:
            yield NO_CONTEXT_MESSAGE  # without grounding the LLM is not called, which prevents hallucination
            return

        context = "\n\n".join(  # 🧠 RAG, augmentation: each chunk goes in with its real sources and score
            f"[Documento: {r['title']} | Fontes: {', '.join(r['sources']) or 'não informada'} | Score: {r['score']:.3f}]\n{r['content']}"
            for r in results
        )
        prompt = f"PERGUNTA ATUAL: {question}\n\nTEXTO BASE:\n{context}"

        answer = ""
        for answer in stream_answer(memory + [types.Content(role="user", parts=[types.Part(text=prompt)])]):  # 🧠 RAG, generation: memory + grounded prompt
            yield answer
        if not answer:
            yield EMPTY_ANSWER_MESSAGE  # 🧠 Empty text means the safety filters blocked the answer
            return

        alerts, notices = check_faithfulness(
            answer, context,
            expected_sources=sorted({s for r in results for s in r["sources"]}),
            expected_titles=list({r["title"] for r in results}),
        )
        if alerts:
            answer += "\n\n> ⚠️ **Alerta de fidelidade:** " + "; ".join(alerts)
        if notices:
            answer += "\n\n> ℹ️ " + "; ".join(notices)
        yield answer

    except Exception as error:
        print(f"[Error] {type(error).__name__}: {error}")
        yield ERROR_MESSAGE


# USER INTERFACE


demo = gr.ChatInterface(  # 🧠 Conversational interface: keeps the history and feeds it to the RAG pipeline on every turn
    fn=chat_fn,
    title="🩺 Assistente RAG — Saúde Pública (Itabira/MG)",
    description=(
        "Protótipo de chatbot com RAG sobre a Covid-19 e as unidades de saúde de Itabira. "
        "As respostas são fundamentadas apenas no texto base recuperado de fontes oficiais.<br>"
        "⚠️ <b>Não utilize para decisões clínicas.</b> Em caso de sintomas, procure uma unidade de saúde."
    ),
    examples=[
        "O que é a COVID-19?",
        "Quais são os sintomas mais comuns?",
        "Estou no Centro, quais UBS ficam perto?",
        "Quem são os grupos de risco?",
        "Quais UBS ficam no bairro São Bento?",
        "Qual o telefone da UBS Bethânia?",
    ],
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
