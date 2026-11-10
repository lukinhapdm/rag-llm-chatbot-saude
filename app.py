import os
import re
import random
import time

import gradio as gr
import numpy as np
import pandas as pd

from google import genai
from google.genai import types
from google.genai import errors

# Fluxo RAG: indexação → retrieval (com query rewriting) → augmentation →
# geração → faithfulness check. Etapas marcadas como "RAG ·" abaixo.

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise gr.Error(
        "GOOGLE_API_KEY não configurada. Adicione como Secret no Space "
        "(Settings → Repository Secrets) e reinicie o Space."
    )

client = genai.Client(api_key=GOOGLE_API_KEY)


# Modelos
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


# Corpus
DOCUMENTO1 = {"Título": "O que é o novo coronavírus, a Covid-19?",
              "Conteúdo": "A COVID-19 é uma doença causada pelo coronavírus, denominado SARS-CoV-2, que apresenta um espectro clínico variando de infecções assintomáticas a quadros graves. De acordo com a Organização Mundial de Saúde, a maioria (cerca de 80%) dos pacientes com COVID-19 podem ser assintomáticos ou oligossintomáticos (poucos sintomas), e aproximadamente 20% dos casos detectados requer atendimento hospitalar por apresentarem dificuldade respiratória, dos quais aproximadamente 5% podem necessitar de suporte ventilatório. Fontes: Secretaria de Estado de Saúde de Minas Gerais e Organização Mundial de Saúde (OMS)."}
DOCUMENTO2 = {"Título": "Quais são os sintomas mais comuns da Covid-19?",
              "Conteúdo": "A COVID-19 (novo coronavírus) possui múltiplos sintomas, em que esses sintomas podem variar, sendo os mais comuns deles: resfriado, febre, sensação febril, tosse, coriza, dor de garganta, dor de cabeça, dificuldade para respirar, perda de olfato, alteração do paladar, náuseas, vômitos, diarreia, cansaço, diminuição do apetite e falta de ar. Fonte: Secretaria de Estado de Saúde de Minas Gerais."}
DOCUMENTO3 = {"Título": "O que fazer caso esteja com sintomas?",
              "Conteúdo": "Se você estiver doente, com sintomas compatíveis com a COVID-19, tais como febre, tosse, dor de garganta e/ou coriza, com ou sem falta de ar, evite contato físico com outras pessoas, incluindo os familiares, principalmente, idosos e doentes crônicos, procure imediatamente os postos de triagem nas Unidades Básica de Saúde (UBS) ou outras unidades de saúde. Fonte: Secretaria de Estado de Saúde de Minas Gerais."}
DOCUMENTO4 = {"Título": "Grupos de risco da Covid-19",
              "Conteúdo": "Alguns grupos têm maior chance de desenvolver as formas mais graves da doença covid-19, como: idosos (idade igual ou superior a 60 anos), pessoas com doenças cardíacas, pessoas com doenças pulmonares, como asma e Doença Pulmonar Obstrutiva Crônica, pessoas com problemas de baixa imunidade, como pessoas transplantadas ou em quimioterapia, pessoas com doenças renais ou em diálise, diabéticos, gestantes de alto risco, pessoas com doenças do fígado e obesos (IMC>=40). Fonte: Secretaria de Estado de Saúde de Minas Gerais."}
DOCUMENTO5 = {"Título": "Bairros e Unidades de Saúde da Cidade de Itabira Separados por Região",
              "Conteúdo": "Bairros da Região 1: Fênix, João XXIII, Valença, Monsenhor José Lopes, Santa Marta, Santa Ruth, Ribeira de Cima, Pedras do Vale, Jardim Belvedere, Bálsamos, Conceição, Vila Técnica Conceição, Abóboras, Barreiro, Condomínio Retiro da Serra, Cedro, Jardim Universitário, CDI I e CDI II. Unidades de Saúde da Região 1: UBS Fênix (3839-2489), UBS João XXIII (3839-2400), UBS Santa Marta (3839-2485), UBS Santa Ruth (3839-2485) e UBS Barreiro (3839-2788). Bairros da Região 2: Gabiroba, Ribeira de Baixo, Santa Tereza, Jardins dos Ipês, Bethânia, Fazenda do Lago, Flamboyant, Vale do Sol, Loteamento da Cohapi e Candidópolis. Unidades de Saúde da Região 2: UBS Bethânia (3839-2935), UBS Gabiroba de Baixo (3839-2491) e UBS Gabiroba de Cima (3839-2492). Bairros da Região 3: Amazonas, Novo Amazonas, Clóvis Alvim I, Clóvis Alvim II, Hamilton, Gianetti, Major Lage (Areão), Caminho Novo, São Tomé, São Bento, Cônego Guilhermino, Vila Dom Prudêncio, São Marcos, Panorama, Água Fresca, Vila Santa Bárbara, Vila Salica, Madre Maria de Jesus, Machado, Juca Batista e Loteamento José Leandro de Araújo. Unidades de Saúde da Região 3: UBS Amazonas (3839-2225), UBS Clóvis Alvim (3839-2165), UBS Major Lage (3839-2324), UBS Areão (3839-2482), UBS Água Fresca (3839-2222), UBS Machado (3839-2497) e UBS Juca Batista (3839-2495). Bairros da Região 4: Colina da Praia, Praia, Eldorado, São Francisco, Cidade Nova, Santa Inês, Santa Matilde, João Fortunato, Alvorada, Juca Rosa, Nossa Senhora das Oliveiras, São Cristóvão, Alta Vista, Belo Monte, Jardim Buritis, Jardim Belvedere, Jardim Mangabeiras, Condomínio Beleville, Condomínio Morada da Brisa e Condomínio Village da Lagoa. Unidades de Saúde da Região 4: UBS Praia I (3831-3819), UBS Praia II (3839-2224), UBS Eldorado (3839-2833), UBS Juca Rosa (3839-2496) e UBS Nossa Senhora das Oliveiras (3839-2574). Bairros da Região 5: Pedreira. Unidades de Saúde da Região 5: UBS Pedreira I (3839-2484) e UBS Pedreira II (3839-2583). Bairros da Região 6: Centro, Penha, Campestre, Bela Vista, Nova Vista, São Pedro, Vila Santa Izabel, Vila Santa Rosa, Vila Piedade, Esplanada da Estação, 14 de Fevereiro, Vila São Joaquim, Alto Pereira, Vila Amélia, Moinho Velho, Chacrinha, Vila Paciência, Vila das Nações Unidas, Santo Antônio e Pará. Unidades de Saúde da Região 6: Hospital Nossa Senhora das Dores (3839-1405), UBS Centro (3839-2488), UBS Campestre (3839-2487), UBS Bela Vista (3839-2486), UBS Nova Vista (3839-2499), UBS Vila Santa Rosa (3839-2494), UBS Vila Piedade (3839-2493) e UBS Pará (3839-2483). Bairros da Região 7: Chapada, Boa Esperança e Barro Branco. Unidades de Saúde da Região 7: UBS Chapada (3839-2927). Bairros da Região 8: Senhora do Carmo. Unidades de Saúde da Região 8: UBS Senhora do Carmo (3839-2766). Bairros da Região 9: Ipoema. Unidades de Saúde da Região 9: UBS Ipoema (3839-2780). Fontes: Secretaria Municipal de Saúde de Itabira e Prefeitura de Itabira."}
documentos = [DOCUMENTO1, DOCUMENTO2, DOCUMENTO3, DOCUMENTO4, DOCUMENTO5]


# Indexação
# RAG · indexação — extrai rodapé "Fonte:/Fontes:" para citation grounding.
def extrair_fontes(texto):
    m = re.search(r"Fontes?:?\s*(.+?)(?:\.\s*$|$)", texto, re.I | re.S)
    if not m: return []
    bloco = m.group(1).strip().rstrip(".")
    return [s.strip() for s in re.split(r"\s+e\s+", bloco) if len(s.strip()) > 8]

# RAG · indexação — chunking com quebra em pontuação e overlap.
def chunkar_texto(texto, max_chars=450, overlap_chars=80):
    if len(texto) <= max_chars: return [texto]
    chunks, inicio = [], 0
    while inicio < len(texto):
        fim = min(inicio + max_chars, len(texto))
        if fim < len(texto):
            janela = texto[max(inicio, fim - 100):fim]
            corte = max(janela.rfind(". "), janela.rfind("! "), janela.rfind("? "))
            if corte != -1:
                fim = max(inicio, fim - 100) + corte + 1
        chunks.append(texto[inicio:fim].strip())
        if fim >= len(texto): break
        inicio = fim - overlap_chars
    return chunks

def montar_base_chunks(documentos, max_chars=450, overlap=80):
    registros = []
    for doc_id, doc in enumerate(documentos):
        fontes = extrair_fontes(doc["Conteúdo"])
        for ch_id, ch in enumerate(chunkar_texto(doc["Conteúdo"], max_chars, overlap)):
            registros.append({
                "Titulo":   doc["Título"],
                "Chunk_ID": f"d{doc_id}_c{ch_id}",
                "Conteudo": ch,
                "Fontes":   fontes,
            })
    return pd.DataFrame(registros)

dataframe = montar_base_chunks(documentos)
print(f"[Corpus] {len(dataframe)} chunks de {len(documentos)} documentos.")


# RAG · indexação vetorial — task_type DOCUMENT para documentos.
def funcao_embed(title, text):
    resp = client.models.embed_content(
        model=model_embeddings, contents=text,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT", title=title),
    )
    return resp.embeddings[0].values

dataframe["Embeddings"] = dataframe.apply(
    lambda row: funcao_embed(row["Titulo"], row["Conteudo"]), axis=1
)


# Retrieval
# RAG · retrieval — task_type QUERY + top-k + threshold.
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
        if scores[i] < threshold: continue
        resultados.append({
            "Titulo":   dataframe.iloc[i]["Titulo"],
            "Chunk_ID": dataframe.iloc[i]["Chunk_ID"],
            "Conteudo": dataframe.iloc[i]["Conteudo"],
            "Fontes":   list(dataframe.iloc[i]["Fontes"]),
            "Score":    float(scores[i]),
        })
    return resultados


# Resiliência
# Backoff exponencial + jitter para ServerError (503).
def chamar_api_com_retentativa(funcao, *args, tentativas=5, base=2.0, cap=30.0, **kwargs):
    for n in range(tentativas):
        try:
            return funcao(*args, **kwargs)
        except errors.ServerError:
            if n == tentativas - 1: raise
            delay = min(cap, base * (2 ** n)) + random.uniform(0, 1)
            time.sleep(delay)


# Safety e prompt de sistema
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

# RAG · generation — few-shot + grounding + regra de citação (só de "Fontes:").
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
* **Fontes:** Secretaria de Estado de Saúde de Minas Gerais e Organização Mundial de Saúde (OMS).

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
* **Fontes:** Secretaria Municipal de Saúde de Itabira e Prefeitura de Itabira.
"""

config_geracao = types.GenerateContentConfig(
    candidate_count=1, temperature=0.5, top_p=0.9, max_output_tokens=2048,
    safety_settings=seguranca_config, system_instruction=instrucao_sistema,
)


# Query rewriting
# RAG · retrieval — consulta autocontida a partir do histórico.
def reformular_consulta(pergunta, historico_txt):
    if not historico_txt: return pergunta
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


# Faithfulness (pós-geração)
# RAG · pós-geração — telefones (contra chunks), fontes e títulos (contra metadado).
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
        for f in re.split(r"\s+e\s+|;", bloco):
            f = f.strip().rstrip(".")
            if len(f) < 8: continue

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


# Interface Gradio
def chat_fn(message: str, history: list[dict]):
    # RAG · retrieval — (0) reformula a partir do histórico textual.
    historico_txt = [f"{m['role']}: {m['content']}" for m in history[-6:]]
    consulta = reformular_consulta(message, historico_txt)

    # RAG · retrieval — (1) top-k + threshold.
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
    ],
    theme="soft",
    api_name="chat",
)

if __name__ == "__main__":
    demo.launch()