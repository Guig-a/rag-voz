<div align="center">

# 🎙️ RAG Voz

**Assistente de voz que responde perguntas com base nos seus próprios documentos.**

Busca semântica com PostgreSQL + pgvector, raciocínio em grafo com LangGraph e conversa por voz em tempo real com LiveKit, construído em etapas para entender cada camada por dentro.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL_16-pgvector-4169E1?logo=postgresql&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C?logo=langchain&logoColor=white)
![LiveKit](https://img.shields.io/badge/LiveKit_Agents-1FD5F9?logo=webrtc&logoColor=black)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)

</div>

---

## Sumário

- [Visão geral](#visão-geral)
- [Arquitetura](#arquitetura)
- [Stack](#stack)
- [Começando](#começando)
- [Usando a API](#usando-a-api)
- [Conversando por voz](#conversando-por-voz)
- [Avaliação](#avaliação)
- [Experimentos com PyTorch](#experimentos-com-pytorch)
- [Configuração](#configuração)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Solução de problemas](#solução-de-problemas)
- [Roadmap](#roadmap)

---

## Visão geral

Um LLM não conhece os seus documentos. Este projeto resolve isso com **RAG** (*Retrieval-Augmented Generation*): para cada pergunta, busca os trechos mais relevantes nos documentos e pede ao modelo que responda **somente** com base neles. Se a resposta não estiver nos documentos, ele diz `nao encontrei isso nos documentos` em vez de inventar.

**Destaques**

- 🔎 **Busca semântica** por significado, não por palavra-chave: embeddings multilíngues locais e índice HNSW no pgvector.
- 🧠 **Grafo de decisão** com LangGraph: reescreve a pergunta, avalia se o contexto basta e tenta de novo antes de desistir.
- 🛡️ **Respostas fundamentadas**: prompt de grounding, limiar mínimo de score e recusa explícita quando falta contexto.
- 🎙️ **Voz em tempo real**: fala → texto → grafo RAG → fala, com detecção de fim de turno e interrupção.
- 🔁 **Resiliência**: retry com backoff para erros transitórios do LLM (429/503) e fallbacks em cada nó do grafo.
- 📏 **Avaliação reproduzível**: conjunto de perguntas com chunk e resposta esperados, incluindo casos que **devem** ser recusados.
- 🔬 **Experimentos em PyTorch** para entender o que acontece por baixo: cosseno na mão, PCA, reranker e loop de treino.

---

## Arquitetura

### Pipeline de voz

```mermaid
flowchart LR
    MIC[🎤 Microfone] --> LK[LiveKit<br/>WebRTC]
    LK --> TD[Detecção<br/>de turno]
    TD --> STT[STT<br/>Deepgram Nova-3]
    STT --> G[[Grafo RAG<br/>LangGraph]]
    G --> TTS[TTS<br/>Gradium]
    TTS --> LK2[LiveKit]
    LK2 --> SPK[🔊 Alto-falante]

    G <--> PG[(PostgreSQL<br/>pgvector)]
    G <--> LLM[LLM<br/>Gemini / Claude]
```

A mesma lógica de RAG atende a **API HTTP** (`POST /ask`) e o **agente de voz**. O agente substitui o nó de LLM do LiveKit (`llm_node`) pelo grafo, então o LiveKit cuida só do áudio.

### Grafo RAG

```mermaid
flowchart TD
    START([Início]) --> RW[rewrite_question<br/>reescreve a pergunta]
    RW --> RT[retrieve<br/>busca top-k no pgvector]
    RT --> GR{grade_context<br/>o contexto basta?}
    GR -- sim --> GEN[generate<br/>resposta fundamentada]
    GR -- não, ainda há tentativas --> RW
    GR -- não, tentativas esgotadas --> DK[say_dont_know<br/>recusa explícita]
    GEN --> END([Fim])
    DK --> END
```

| Nó | O que faz | Se o LLM falhar |
|---|---|---|
| `rewrite_question` | Reescreve a pergunta para melhorar a busca, preservando os termos-chave | Usa a pergunta original |
| `retrieve` | Busca os `k` chunks mais próximos por cosseno | — |
| `grade_context` | Aplica o `RAG_MIN_SCORE` e pede ao LLM um veredito SIM/NÃO | Decide só pelo score |
| `generate` | Gera a resposta usando apenas o contexto | Resposta de erro amigável |
| `say_dont_know` | Devolve a recusa padrão | — |

**Modo voz.** Para reduzir a latência e o consumo de cota, o agente de voz pula a reescrita na 1ª tentativa e avalia o contexto só pelo score. Fica **1 chamada ao LLM por pergunta**, em vez de 3. A API mantém o grafo completo.

---

## Stack

| Camada | Tecnologia | Por quê |
|---|---|---|
| API | FastAPI + Uvicorn | Rotas finas, validação com Pydantic |
| Banco vetorial | PostgreSQL 16 + pgvector | SQL direto com `psycopg` v3, sem ORM, para ver cada query |
| Embeddings | `intfloat/multilingual-e5-small` (local) | Multilíngue, 384 dimensões, roda em CPU |
| LLM | Google Gemini ou Anthropic Claude | Provedor trocável por variável de ambiente |
| Orquestração | LangGraph | Estado compartilhado, arestas condicionais, condição de parada |
| Voz | LiveKit Agents + LiveKit Inference | STT, TTS e detecção de turno com suporte a português |
| Experimentos | PyTorch + matplotlib | Entender embeddings, reranking e treino |

---

## Começando

### Pré-requisitos

- Python **3.11+**
- Docker Desktop
- Chave de API de um LLM ([Google AI Studio](https://aistudio.google.com/apikey) ou [Anthropic](https://console.anthropic.com/))
- Para voz: um projeto gratuito no [LiveKit Cloud](https://cloud.livekit.io)

### Instalação

```powershell
git clone git@github.com:Guig-a/rag-voz.git
cd rag-voz

python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[experiments]"

Copy-Item .env.example .env   # depois preencha LLM_API_KEY (e as chaves do LiveKit, para voz)
```

> No macOS/Linux, use `source .venv/bin/activate` e `cp .env.example .env`.

### Banco de dados e documentos

```powershell
docker compose up -d                       # sobe o Postgres com pgvector
.\.venv\Scripts\python.exe data\ingest.py  # indexa os documentos de exemplo em data/
```

O script de ingestão é idempotente: documentos que já existem (pelo nome) são pulados.

### Subir a API

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

A documentação interativa fica em <http://localhost:8000/docs>.

---

## Usando a API

| Método | Rota | Descrição |
|---|---|---|
| `GET` | `/health` | Verificação de saúde |
| `POST` | `/ingest` | Divide um texto em chunks, gera embeddings e salva |
| `POST` | `/search` | Busca semântica pura, sem LLM |
| `POST` | `/ask` | Pergunta completa pelo grafo RAG |

**Ingerir um documento**

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/ingest -ContentType "application/json" `
  -Body '{"name": "notas.md", "content": "O pgvector adiciona o tipo vector ao PostgreSQL..."}'
```

**Perguntar**

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/ask -ContentType "application/json" `
  -Body '{"question": "Qual índice o pgvector usa para acelerar a busca?", "k": 5}'
```

Formato da resposta (valores ilustrativos):

```json
{
  "question": "Qual índice o pgvector usa para acelerar a busca?",
  "answer": "O pgvector usa o índice HNSW, um grafo de vizinhança em camadas...",
  "used_chunk_ids": [4, 3],
  "best_score": 0.8795,
  "rewritten_question": "Qual índice o pgvector utiliza para acelerar buscas?",
  "attempts": 1,
  "context_is_enough": true,
  "sources": [{ "chunk_id": 4, "document_id": 3, "position": 1, "content": "...", "score": 0.8795 }]
}
```

Os campos `rewritten_question`, `attempts` e `context_is_enough` expõem o caminho que a pergunta fez no grafo, o que facilita a depuração.

---

## Conversando por voz

1. Preencha `LIVEKIT_URL`, `LIVEKIT_API_KEY` e `LIVEKIT_API_SECRET` no `.env`. Todos vêm de **Settings → API Keys** no LiveKit Cloud; crie uma chave nova e copie os três valores da mesma tela.
2. Baixe os modelos locais usados pelo agente (só na primeira vez):

   ```powershell
   .\.venv\Scripts\python.exe voice\agent.py download-files
   ```

3. Converse pelo terminal:

   ```powershell
   .\.venv\Scripts\python.exe voice\agent.py console                  # voz, com o microfone padrão
   .\.venv\Scripts\python.exe voice\agent.py console --input-device 21 # escolhendo o microfone
   .\.venv\Scripts\python.exe voice\agent.py console --text            # digitando (sem áudio)
   .\.venv\Scripts\python.exe voice\agent.py dev                       # conectar pelo Agents Playground
   ```

Use `voice\agent.py console --list-devices` para ver os IDs dos microfones.

**Como o agente se comporta**

- Diz *"Um momento, vou consultar os documentos."* assim que você termina de falar, enquanto o grafo trabalha.
- Responde em no máximo duas frases curtas, sem listas nem markdown, porque um prompt específico para voz controla o estilo.
- Pode ser interrompido: comece a falar por cima e ele para.

**Latência medida** (fim da fala → início do áudio)

| Etapa | Tempo |
|---|---|
| Detecção de fim de turno | ~0,5 s |
| Frase de espera começa a tocar | ~0,9 s |
| Busca + avaliação por score | ~10 ms |
| Geração no LLM (sem erros) | ~1,7 s |
| **Resposta real (sem erros)** | **~2,5 s** |

O gargalo é o LLM. No plano gratuito do Gemini, erros 503 e 429 somam vários segundos de retentativas.

> ⚠️ A API do LiveKit Agents muda com frequência. Este projeto usa a versão **1.8.x**; em caso de quebra após atualizar, consulte a [documentação oficial](https://docs.livekit.io/agents/).

---

## Avaliação

```powershell
.\.venv\Scripts\python.exe -m eval.run_eval
```

O arquivo `eval/questions.jsonl` define cada caso com a resposta esperada e o trecho que deve ser recuperado. Casos com `"expected_answer": "__NO_ANSWER__"` exigem que o sistema **recuse** a pergunta.

| Métrica | O que mede |
|---|---|
| Chunk esperado no top-k | A busca encontrou o trecho certo? |
| Resposta esperada | A resposta contém a informação correta? |
| Recusas corretas | O sistema admitiu não saber quando devia? |
| Falhas de infraestrutura | Erros do provedor, contados à parte para não distorcer as métricas |

---

## Experimentos com PyTorch

Experimentos independentes em `experiments/`, que **não fazem parte do app**, para entender as peças por dentro. Os resultados detalhados e as conclusões estão em [`NOTES.md`](NOTES.md).

```powershell
.\.venv\Scripts\python.exe -m experiments.01_cosine_by_hand "Qual é o maior planeta?"
.\.venv\Scripts\python.exe -m experiments.02_visualize_embeddings
.\.venv\Scripts\python.exe -m experiments.03_reranker
.\.venv\Scripts\python.exe -m experiments.04_train_intent
```

| # | Experimento | Principal resultado |
|---|---|---|
| 1 | Cosseno calculado na mão com tensores | Bate com o pgvector (diferença de ~1e-8) e mantém a mesma ordem |
| 2 | PCA dos embeddings em 2D | Os documentos formam grupos nítidos por tema |
| 3 | Reranker cross-encoder no top-10 | Não melhora a ordem neste corpus (MRR 0.944 → 0.926), mas separa muito melhor perguntas **sem resposta** |
| 4 | Rede neural treinada do zero | Classifica intenção com 100% nas frases de teste; loss de 1.09 para 0.0007 |

<div align="center">
  <img src="experiments/output/embeddings_pca.png" alt="Chunks projetados em 2D com PCA, agrupados por documento" width="640">
  <br>
  <sub>Chunks projetados em 2D. Cada cor é um documento; temas técnicos ficam vizinhos, café e astronomia ficam isolados.</sub>
</div>

---

## Configuração

Toda a configuração vem do `.env` (veja `.env.example`). Nenhuma chave fica no código.

| Variável | Padrão | Descrição |
|---|---|---|
| `DATABASE_URL` | `postgresql://postgres:postgres@localhost:5432/rag` | Conexão com o Postgres |
| `LLM_PROVIDER` | `gemini` | `gemini` ou `anthropic` |
| `LLM_API_KEY` | — | Chave do provedor de LLM |
| `LLM_MODEL` | `gemini-3.1-flash-lite` | Modelo usado em todas as chamadas |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-small` | Modelo de embeddings (local) |
| `EMBEDDING_DIM` | `384` | Precisa bater com o modelo e com a coluna do banco |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `500` / `100` | Tamanho e sobreposição dos chunks, em caracteres |
| `SEARCH_DEFAULT_K` | `5` | Chunks recuperados por pergunta |
| `RAG_MIN_SCORE` | `0.75` | Score de cosseno mínimo para considerar o contexto |
| `GRAPH_MAX_ATTEMPTS` | `2` | Máximo de voltas do grafo antes de recusar |
| `LLM_RETRY_ATTEMPTS` / `LLM_RETRY_DELAY_SECONDS` | `3` / `2.0` | Retentativas para 429/503, com espera crescente |
| `LIVEKIT_URL` / `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` | — | Credenciais do LiveKit Cloud |
| `VOICE_LANGUAGE` | `pt-BR` | Idioma do STT e do TTS |
| `VOICE_STT_MODEL` / `VOICE_TTS_MODEL` | `deepgram/nova-3` / `gradium/default` | Modelos do LiveKit Inference |
| `VOICE_TTS_VOICE` | `uCqxlQCKi8sPHwG2` | Voz do TTS |

> Se trocar o modelo de embedding, ajuste `EMBEDDING_DIM`, recrie a coluna `vector` e reindexe todos os documentos.

---

## Estrutura do projeto

```
rag-voz/
├── app/
│   ├── main.py          # FastAPI: rotas finas, a lógica fica nos módulos
│   ├── config.py        # Settings lidas do .env
│   ├── chunking.py      # divisão em chunks com overlap
│   ├── embeddings.py    # sentence-transformers com prefixos query:/passage:
│   ├── db.py            # SQL direto: schema, inserção e busca por cosseno
│   ├── retrieval.py     # ingestão e busca semântica
│   ├── llm.py           # provedores de LLM, retry e prompt de grounding
│   ├── graph.py         # grafo LangGraph
│   └── rag.py           # ponto de entrada do RAG usado pela API
├── voice/
│   └── agent.py         # agente LiveKit que usa o grafo como "cérebro"
├── data/
│   ├── *.md             # corpus de exemplo
│   └── ingest.py        # indexa o corpus
├── eval/
│   ├── questions.jsonl  # casos de avaliação
│   └── run_eval.py
├── experiments/         # PyTorch: cosseno, PCA, reranker, treino
├── NOTES.md             # diário de experimentos
├── docker-compose.yml   # Postgres + pgvector
└── pyproject.toml
```

---

## Solução de problemas

| Sintoma | Causa provável |
|---|---|
| `psycopg ... ConnectionTimeout` | Docker Desktop ou o container `rag-voz-postgres` não está rodando |
| `503 UNAVAILABLE` / `429 RESOURCE_EXHAUSTED` | Sobrecarga ou cota do plano gratuito do LLM; aguarde ou troque de modelo |
| `401 Unauthorized` no agente de voz | `LIVEKIT_API_KEY` e `LIVEKIT_API_SECRET` de projetos diferentes ou copiados errados |
| Agente não reage à voz | Microfone padrão errado; use `--list-devices` e `--input-device` |
| `console --text` fecha ao enviar | Sessão sem LLM configurado. O agente usa um `RagGraphLLM` marcador para isso |
| Busca devolve trechos sem relação | Prefixo de embedding errado, chunk grande demais ou modelo não multilíngue |
| Erro de dimensão no pgvector | Modelo de embedding trocado sem recriar a coluna e reindexar |

---

## Roadmap

- [x] **Etapa 0**: ambiente, FastAPI e Postgres com pgvector
- [x] **Etapa 1**: chunking, embeddings e busca semântica
- [x] **Etapa 2**: RAG completo com grounding e avaliação
- [x] **Etapa 3**: fluxo reescrito como grafo LangGraph com retentativa
- [x] **Etapa 4**: voz em tempo real com LiveKit
- [x] **Etapa 5**: experimentos com PyTorch

**Ideias futuras**: busca híbrida (vetorial + full-text), reranker como filtro de relevância no `grade_context`, streaming da resposta, memória de conversa e avaliação com LLM-as-judge.

---

> 🔒 **Privacidade:** no plano gratuito do Gemini, as entradas e saídas podem ser usadas pelo Google e lidas por revisores humanos. Use apenas documentos sem dados sensíveis.
