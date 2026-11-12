import os
import re
import random
import time

import gradio as gr
import numpy as np
import pandas as pd
from datasets import load_dataset

from google import genai
from google.genai import types
from google.genai import errors

# RAG flow: indexação → retrieval (com query rewriting) → augmentation →
# geração → faithfulness check. Etapas marcadas como "RAG ·" abaixo.

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise gr.Error(
        "GOOGLE_API_KEY não configurada. Adicione como Secret no Space "
        "(Settings → Repository Secrets) e reinicie o Space."
    )

# Dataset público no HF Hub — não precisa de token para leitura.
HF_DATASET = os.getenv("HF_DATASET", "lukinhapdm/corpus-rag-saude-itabira")

client = genai.Client(api_key=GOOGLE_API_KEY)


# MODELOS
# Blocklists separadas: "embedding" só bloqueia geradores, nunca embedders.
BLOQUEIO_OBSOLETO_COMUM = ["gemini-2.0", "gemini-2.5", "gemini-1.5", "gemini-1.0"]
BLOQUEIO_GERACAO   = BLOQUEIO_OBSOLETO_COMUM + ["embedding", "aqa", "imagen", "veo", "tts"]
BLOQUEIO_EMBEDDING = BLOQUEIO_OBSOLETO_COMUM + ["aqa", "imagen", "veo", "tts"]

PREFERENCIAS_GERACAO   = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite",
                          "gemini-3.5-flash", "gemini-3.8-flash",
                          "gemini-3.6-flash", "gemini-3.7-flash"]
PREFERENCIAS_EMBEDDING = ["gemini-embedding-001", "embedding-001", "embedding"]

def _bloqueado(nome, bloquear):
    n = nome.lower()
    return any(b in n for b in bloquear)

def listar_candidatos(preferencias, disponiveis, bloquear):
    # Preferências primeiro, demais não bloqueados depois.
    candidatos, vistos = [], set()
    for pref in preferencias:
        for nome in disponiveis:
            if pref in nome and not _bloqueado(nome, bloquear) and nome not in vistos:
                candidatos.append(nome); vistos.add(nome)
    for nome in disponiveis:
        if not _bloqueado(nome, bloquear) and nome not in vistos:
            candidatos.append(nome); vistos.add(nome)
    return candidatos

# Separa o catálogo por capability.
todos_geracao, todos_embedding = [], []
for m in client.models.list():
    acoes = getattr(m, "supported_actions", []) or []
    if "generateContent" in acoes: todos_geracao.append(m.name)
    if "embedContent"   in acoes: todos_embedding.append(m.name)

if not todos_embedding:
    raise RuntimeError(
        "Nenhum modelo de embedding na API. Verifique a chave em "
        "https://ai.google.dev/gemini-api/docs/models"
    )

candidatos_geracao   = listar_candidatos(PREFERENCIAS_GERACAO,   todos_geracao,   BLOQUEIO_GERACAO)
candidatos_embedding = listar_candidatos(PREFERENCIAS_EMBEDDING, todos_embedding, BLOQUEIO_EMBEDDING)

if not candidatos_geracao:
    raise RuntimeError(f"Nenhum gerador ativo. API retornou: {todos_geracao}")

# Valida em runtime e faz failover em 404 (modelos "zumbis" da API).
def validar_geracao(nome):
    try:
        client.models.generate_content(
            model=nome, contents="ping",
            config=types.GenerateContentConfig(max_output_tokens=1, temperature=0.0),
        )
        return True
    except errors.ClientError as e:
        if getattr(e, "status_code", None) == 404 or "404" in str(e):
            return False
        raise
    except Exception:
        return False

model_name_geracao = next((c for c in candidatos_geracao if validar_geracao(c)), None)
if not model_name_geracao:
    raise RuntimeError("Nenhum gerador respondeu ao teste de validação.")

model_embeddings = candidatos_embedding[0]
print(f"[Gerador]    {model_name_geracao}")
print(f"[Embedder]   {model_embeddings}")


# CORPUS
# RAG · indexação — o dataset já vem chunked (id, titulo, texto, fonte,
# documento_origem). Como é público, load_dataset não exige autenticação.
corpus = load_dataset(HF_DATASET, split="train")

dataframe = corpus.to_pandas().rename(columns={
    "id":     "Chunk_ID",
    "titulo": "Titulo",
    "texto":  "Conteudo",
})

# A coluna `fonte` é uma string separada por ", " — reconstruímos a lista
# para o faithfulness check e para o cabeçalho de augmentação.
dataframe["Fontes"] = dataframe["fonte"].apply(
    lambda s: [f.strip() for f in str(s).split(", ") if f.strip()]
)

print(f"[Corpus] {len(dataframe)} chunks carregados de {HF_DATASET}")


# INDEXAÇÃO VETORIAL
# RAG · indexação vetorial — task_type DOCUMENT; `title` ancora semanticamente
# cada chunk com o título do documento-pai.
def funcao_embed(title, text):
    resp = client.models.embed_content(
        model=model_embeddings, contents=text,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT", title=title),
    )
    return resp.embeddings[0].values

dataframe["Embeddings"] = dataframe.apply(
    lambda row: funcao_embed(row["Titulo"], row["Conteudo"]), axis=1
)


# RETRIEVAL
# RAG · retrieval — task_type QUERY + top-k + threshold. Sem contexto válido,
# o loop do chat NÃO chama o LLM (evita alucinação por falta de grounding).
def recuperar_contexto(consulta, top_k=3, threshold=0.30):
    resp = client.models.embed_content(
        model=model_embeddings, contents=consulta,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    q_emb = np.array(resp.embeddings[0].values)
    scores = np.dot(np.stack(dataframe["Embeddings"]), q_emb)
    ordem = np.argsort(scores)[::-1][:top_k]

    resultados = []
    for i in ordem:
        if scores[i] < threshold:
            continue
        resultados.append({
            "Titulo":   dataframe.iloc[i]["Titulo"],
            "Chunk_ID": dataframe.iloc[i]["Chunk_ID"],
            "Conteudo": dataframe.iloc[i]["Conteudo"],
            "Fontes":   list(dataframe.iloc[i]["Fontes"]),
            "Score":    float(scores[i]),
        })
    return resultados


# RESILIÊNCIA
# Backoff exponencial + jitter para ServerError (503).
def chamar_api_com_retentativa(funcao, *args, tentativas=5, base=2.0, cap=30.0, **kwargs):
    for n in range(tentativas):
        try:
            return funcao(*args, **kwargs)
        except errors.ServerError:
            if n == tentativas - 1:
                raise
            delay = min(cap, base * (2 ** n)) + random.uniform(0, 1)
            time.sleep(delay)


# SAFETY E PROMPT DO SISTEMA
seguranca_config = [
    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                        threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE),
    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE),
    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE),
    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE),
]

# RAG · generation — few-shot + grounding restritivo + regra de citação (só copia
# do campo "Fontes:" do cabeçalho; nunca infere a partir do conteúdo).
instrucao_sistema = """
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

config_geracao = types.GenerateContentConfig(
    candidate_count=1, temperature=0.5, top_p=0.9, max_output_tokens=2048,
    safety_settings=seguranca_config, system_instruction=instrucao_sistema,
)


# QUERY REWRITING
# RAG · retrieval — reescreve a pergunta como consulta autocontida a partir do
# histórico; fallback silencioso para a pergunta original em qualquer falha.
def reformular_consulta(pergunta, historico_txt):
    if not historico_txt:
        return pergunta
    contexto = "\n".join(historico_txt[-6:])
    prompt = (
        "Reescreva a NOVA PERGUNTA como uma consulta AUTOCONTIDA, "
        "substituindo pronomes e referências implícitas por termos explícitos. "
        "Responda APENAS a consulta reescrita.\n\n"
        f"HISTÓRICO:\n{contexto}\n\nNOVA PERGUNTA: {pergunta}\n\nCONSULTA REESCRITA:"
    )
    try:
        resp = chamar_api_com_retentativa(
            client.models.generate_content, model=model_name_geracao, contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0, max_output_tokens=128, safety_settings=seguranca_config,
            ),
        )
        return (resp.text or "").strip() or pergunta
    except Exception:
        return pergunta


# FAITHFULNESS
# RAG · pós-geração — checagens ortogonais:
#   (a) telefones: contra os chunks recuperados (o que o modelo viu);
#   (b) fontes: contra o CONJUNTO de fontes dos documentos recuperados;
#   (c) título citado como fonte: vira aviso, não erro.
def verificar_faithfulness(resposta, contexto_chunks, fontes_esperadas, titulos_esperados):
    alertas, avisos = [], []

    tel_resp = set(re.findall(r"\b\d{4}-\d{4}\b", resposta))
    tel_ctx  = set(re.findall(r"\b\d{4}-\d{4}\b", contexto_chunks))
    orfaos = tel_resp - tel_ctx
    if orfaos:
        alertas.append(f"telefone(s) sem lastro nos chunks: {sorted(orfaos)}")

    # Decompõe o bloco "A e B e C" em fontes individuais antes de comparar.
    blocos = re.findall(r"Fontes?:?\s*\**\s*([^.\n]+)", resposta, re.I)
    for bloco in blocos:
        bloco = re.sub(r"\*+", "", bloco).strip()
        for f in re.split(r"\s+e\s+|,|;", bloco):
            f = f.strip().rstrip(".")
            if len(f) < 8:
                continue

            if any(f.lower() in t.lower() or t.lower() in f.lower()
                   for t in titulos_esperados):
                avisos.append(f"citou título como fonte: '{f[:60]}...'")
                continue

            if fontes_esperadas and not any(
                f.lower() in fe.lower() or fe.lower() in f.lower()
                for fe in fontes_esperadas
            ):
                alertas.append(f"fonte não corresponde aos documentos recuperados: '{f[:60]}...'")

    return alertas, avisos


# INTERFACE GRADIO
def chat_fn(message: str, history: list[dict]):
    # RAG · retrieval — (0) reformula com base no histórico textual.
    historico_txt = [f"{m['role']}: {m['content']}" for m in history[-6:]]
    consulta = reformular_consulta(message, historico_txt)

    # RAG · retrieval — (1) top-k + threshold; sem resultado, não chama o LLM.
    resultados = recuperar_contexto(consulta)
    if not resultados:
        yield "Não encontrei informação relevante na base para responder com segurança."
        return

    # RAG · augmentation — (2) injeta fontes reais ao lado de cada documento.
    contexto = "\n\n".join(
        f"[Documento: {r['Titulo']} | Fontes: {', '.join(r['Fontes']) or 'não informada'} | Score: {r['Score']:.3f}]\n{r['Conteudo']}"
        for r in resultados
    )
    prompt = f"PERGUNTA ATUAL: {message}\n\nTEXTO BASE:\n{contexto}"

    # RAG · generation — (3) chama o modelo com o prompt aumentado.
    resposta = chamar_api_com_retentativa(
        client.models.generate_content,
        model=model_name_geracao, contents=prompt, config=config_geracao,
    )
    texto = resposta.text or ""

    # RAG · pós-geração — (4) checagens de faithfulness.
    titulos_esperados = list({r["Titulo"] for r in resultados})
    fontes_esperadas  = sorted({f for r in resultados for f in r["Fontes"]})
    alertas, avisos = verificar_faithfulness(
        texto, contexto, fontes_esperadas, titulos_esperados,
    )
    if alertas:
        texto += "\n\n> ⚠️ **Alerta de faithfulness:** " + "; ".join(alertas)
    if avisos:
        texto += "\n\n> ℹ️ " + "; ".join(avisos)

    # Streaming token a token via yield.
    parcial = ""
    for token in texto.split():
        parcial += token + " "
        yield parcial


demo = gr.ChatInterface(
    fn=chat_fn,
    title="Assistente RAG — Saúde Pública (Itabira)",
    description=(
        "Protótipo de chatbot com RAG sobre dados de COVID-19 e unidades de "
        "saúde de Itabira. As respostas são fundamentadas apenas no texto base "
        "recuperado. **Não utilizar para decisões clínicas.**"
    ),
    examples=[
        "O que é a COVID-19?",
        "Quais são os sintomas mais comuns?",
        "Estou no Centro, quais UBS ficam perto?",
        "Quem são os grupos de risco?",
        "Quais UBS ficam no bairro São Bento?",
        "Qual o telefone da UBS Bethânia?",
    ],
    theme="soft",
    api_name="chat",
)

if __name__ == "__main__":
    demo.launch()
