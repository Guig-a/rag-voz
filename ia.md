# Guia do Projeto: Assistente de Voz com RAG

> Este arquivo serve para duas coisas: **guia de estudo** para mim e **contexto/instruções** para qualquer IA que me ajude no projeto (cole no início da conversa ou salve como `AGENTS.md` / `CLAUDE.md` na raiz do repositório).

---

## 1. Objetivo

Construir, em etapas pequenas e funcionais, um **assistente que responde perguntas sobre meus próprios documentos**, primeiro por texto (API) e depois por voz.

O objetivo principal **não é o produto, é entender como cada peça funciona**: LLM, embeddings, chunking, RAG, LangGraph, LiveKit, PyTorch e FastAPI.

Critério de sucesso: ao final de cada etapa eu consigo **explicar com minhas palavras** o que cada parte do código faz e por que existe.

---

## 2. Instruções para a IA (leia com atenção)

### Perfil de quem está aprendendo
- Desenvolvedor full stack (TypeScript, NestJS, Next.js, PostgreSQL, Prisma), ~2 anos de experiência.
- **Python e o ecossistema de IA são novos para mim.** Python eu aprendo no caminho; os conceitos de IA são o foco.
- Inglês B2. Pode escrever código e termos técnicos em inglês, mas **explique em português**.

### Como me ajudar
1. **Passos pequenos.** Uma etapa por vez. Não implemente o projeto inteiro de uma vez.
2. **Explique o "porquê" antes do "como".** Antes de cada trecho de código relevante, diga em 2 a 3 linhas qual problema ele resolve.
3. **Use analogias com o que eu já conheço** quando ajudar (ex.: Pydantic ≈ DTOs do NestJS, `Depends` ≈ injeção de dependência, Alembic ≈ migrations do Prisma).
4. **Sem frameworks cedo demais.** Nas etapas 1 e 2, **não use LangChain** nem abstrações que escondam o fluxo. Use o SDK do provedor de LLM, `psycopg` e funções simples. Frameworks só quando eu pedir ou quando a etapa mandar (LangGraph na etapa 3).
5. **Código mínimo e legível.** Prefira clareza a "elegância". Comentários só onde o conceito não é óbvio.
6. **Aponte trade-offs reais** (tamanho de chunk, valor de `k`, modelo de embedding) e diga o que eu devo **experimentar e medir**, em vez de dar uma resposta "certa" sem evidência.
7. **Seja honesto sobre o que é necessário e o que é só didático.** Ex.: PyTorch não é necessário para o app funcionar, é para eu entender o que acontece por baixo.
8. **Se algo depender de versão de biblioteca ou API que muda rápido** (LiveKit Agents, LangGraph, SDKs de LLM), avise e me mande conferir a documentação oficial em vez de chutar.
9. **No fim de cada etapa**, me faça 3 perguntas para eu checar se entendi, e liste os erros comuns que eu posso ter cometido.

### O que evitar
- Gerar o projeto inteiro de uma vez.
- Esconder lógica atrás de helpers genéricos.
- Adicionar autenticação, filas, Docker complexo, front-end etc. sem eu pedir. **Fora do escopo.**
- Afirmar que algo "funciona" sem eu poder rodar e ver o resultado.

---

## 3. Stack

| Camada | Escolha | Observação |
|---|---|---|
| Linguagem | Python 3.11+ | Gerenciar com `uv` (ou `venv` + `pip`) |
| API | FastAPI + Uvicorn | Pydantic para validação |
| Banco | PostgreSQL + `pgvector` | Imagem `pgvector/pgvector:pg16` |
| Acesso ao banco | `psycopg` (v3) | SQL direto, sem ORM, para ver o que acontece |
| Embeddings | `sentence-transformers` (local) | Usar um modelo **multilíngue** (documentos em português) |
| LLM | API de um provedor (ex.: Anthropic) | Chave e modelo via variáveis de ambiente |
| Orquestração | LangGraph | A partir da etapa 3 |
| Voz | LiveKit Agents (Python) | A partir da etapa 4 |
| Deep learning | PyTorch | Etapa 5 (experimentos) |

Variáveis de ambiente (`.env`):

```env
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/rag
LLM_API_KEY=
LLM_MODEL=
EMBEDDING_MODEL=intfloat/multilingual-e5-small
EMBEDDING_DIM=384
```

> Se trocar o modelo de embedding, a dimensão do vetor no banco **precisa** mudar junto (e os documentos precisam ser reindexados).

---

## 4. Estrutura sugerida do repositório

```
rag-voz/
├── AGENTS.md              # este guia
├── docker-compose.yml     # só o Postgres com pgvector
├── pyproject.toml
├── .env.example
├── data/                  # documentos de exemplo (corpus)
├── eval/
│   ├── questions.jsonl    # perguntas + resposta esperada + chunk esperado
│   └── run_eval.py
├── app/
│   ├── main.py            # FastAPI (rotas)
│   ├── config.py          # leitura do .env
│   ├── db.py              # conexão e queries
│   ├── chunking.py        # estratégias de chunking
│   ├── embeddings.py      # geração de embeddings
│   ├── retrieval.py       # busca por similaridade
│   ├── llm.py             # chamada ao LLM
│   ├── rag.py             # pipeline simples (etapa 1)
│   └── graph.py           # grafo LangGraph (etapa 3)
├── voice/
│   └── agent.py           # agente LiveKit (etapa 4)
└── experiments/
    └── *.ipynb | *.py     # PyTorch e afins (etapa 5)
```

---

## 5. Conceitos-chave (cola rápida)

- **LLM**: modelo que prevê o próximo token. Não "sabe" dos meus documentos; só do que foi treinado e do que eu coloco no prompt.
- **Embedding**: vetor numérico que representa o significado de um texto. Textos parecidos ficam **próximos** no espaço vetorial.
- **Similaridade de cosseno**: mede o ângulo entre dois vetores. Quanto mais perto de 1, mais parecidos.
- **Chunking**: dividir documentos em pedaços para indexar. Pedaço grande demais dilui o significado; pequeno demais perde contexto.
- **Overlap**: sobreposição entre chunks vizinhos, para não cortar uma ideia no meio.
- **RAG**: *Retrieval-Augmented Generation*. Buscar trechos relevantes e **colocá-los no prompt** para o LLM responder com base neles.
- **top-k**: quantos chunks recuperar por pergunta.
- **Grounding / alucinação**: o quanto a resposta é sustentada pelo contexto recuperado, e o que o modelo inventa quando não é.
- **Reranker**: segundo modelo (cross-encoder) que reordena os top-k do jeito mais preciso (e mais lento).
- **LangGraph**: biblioteca para descrever um fluxo como **grafo** (nós, estado compartilhado, arestas condicionais, loops).
- **VAD / STT / TTS**: detecção de voz / fala→texto / texto→fala. Compõem o pipeline de voz.
- **LiveKit**: infraestrutura de áudio/vídeo em tempo real (WebRTC). O *Agents* é o framework para um agente participar de uma sala.

---

## 6. Roadmap por etapas

Cada etapa tem **entrega**, **o que aprender** e **critério de pronto**. Só avançar quando o critério for atendido.

### Etapa 0: Ambiente e "hello world"
**Entrega:** Postgres com `pgvector` rodando; FastAPI com `GET /health`.

**Aprender:** estrutura de projeto Python, ambiente virtual, `uvicorn`, Pydantic.

**Pronto quando:**
- [ ] `docker compose up` sobe o Postgres.
- [ ] `CREATE EXTENSION vector;` executa sem erro.
- [ ] `GET /health` responde `{"status": "ok"}`.

---

### Etapa 1: Embeddings e busca semântica (sem LLM ainda)
**Entrega:** `POST /ingest` e `POST /search`.

**Aprender:** chunking, embeddings, similaridade, índice vetorial.

Schema sugerido:

```sql
CREATE TABLE documents (
  id SERIAL PRIMARY KEY,
  name TEXT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE chunks (
  id SERIAL PRIMARY KEY,
  document_id INT REFERENCES documents(id) ON DELETE CASCADE,
  position INT NOT NULL,
  content TEXT NOT NULL,
  embedding vector(384) NOT NULL
);

CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops);
```

Busca (o operador `<=>` é a distância de cosseno):

```sql
SELECT id, content, 1 - (embedding <=> %s) AS score
FROM chunks
ORDER BY embedding <=> %s
LIMIT %s;
```

**Fluxo de `/ingest`:** ler texto → chunking → embedding de cada chunk → inserir no banco.

**Fluxo de `/search`:** embedding da pergunta → query acima → devolver chunks + score.

> Alguns modelos (como a família `e5`) esperam prefixos diferentes para documento e pergunta (`passage:` / `query:`). **Conferir a documentação do modelo escolhido.**

**Experimentos obrigatórios (anotar o que observei):**
1. Chunk de 200 vs 500 vs 1000 caracteres/tokens. Quais resultados mudam?
2. Com e sem overlap.
3. Perguntar com palavras diferentes das do texto (paráfrase). A busca ainda acerta?
4. Perguntar algo que **não está** nos documentos. Qual o score do melhor resultado?

**Pronto quando:**
- [ ] Ingiro 1 documento e `/search` devolve trechos coerentes.
- [ ] Sei explicar por que o `<=>` ordena do mais próximo ao mais distante.
- [ ] Tenho anotado qual configuração de chunking funcionou melhor no meu corpus, e por quê.

---

### Etapa 2: RAG completo
**Entrega:** `POST /ask` que devolve resposta **e as fontes usadas**.

**Aprender:** montagem de prompt, grounding, alucinação.

Fluxo:

```
pergunta → embedding → top-k chunks → prompt (contexto + pergunta) → LLM → resposta + fontes
```

Pontos de atenção no prompt:
- Instruir o modelo a responder **somente com base no contexto**.
- Instruir a dizer **"não encontrei isso nos documentos"** quando o contexto não bastar.
- Separar claramente contexto e pergunta (delimitadores).
- Devolver os `chunk_id`s usados, para eu auditar.

**Montar o conjunto de avaliação** (`eval/questions.jsonl`), com ~10 perguntas:

```json
{"question": "...", "expected_answer": "...", "expected_chunk_hint": "trecho que deveria ser recuperado"}
```

Incluir 2 ou 3 perguntas **sem resposta** nos documentos. O script `run_eval.py` roda todas e mostra: o chunk esperado entrou no top-k? A resposta bate? O modelo recusou quando devia?

**Pronto quando:**
- [ ] `/ask` responde com fontes.
- [ ] Para perguntas fora do corpus, o sistema diz que não sabe (em vez de inventar).
- [ ] `run_eval.py` roda e eu comparo **antes/depois** ao mudar `k` ou o chunking.

---

### Etapa 3: LangGraph
**Entrega:** o fluxo da etapa 2 reescrito como grafo, com decisão e retentativa.

**Aprender:** estado, nós, arestas condicionais, loops.

Estado:

```python
class RAGState(TypedDict):
    question: str
    rewritten_question: str
    chunks: list[dict]
    context_is_enough: bool
    attempts: int
    answer: str
```

Grafo:

```
START → rewrite_question → retrieve → grade_context
grade_context ──(suficiente)──→ generate → END
grade_context ──(insuficiente e attempts < 2)──→ rewrite_question
grade_context ──(insuficiente e attempts >= 2)──→ say_dont_know → END
```

- `rewrite_question`: reformula a pergunta para melhorar a busca (e, na 2ª tentativa, tenta outro ângulo).
- `grade_context`: o LLM decide, com saída estruturada (sim/não), se os chunks bastam para responder.
- O `attempts` evita loop infinito. **Todo loop precisa de condição de parada.**

**Pronto quando:**
- [ ] `/ask` agora roda pelo grafo.
- [ ] Consigo desenhar o grafo de memória e explicar cada aresta.
- [ ] Vi, em pelo menos um caso, o grafo reescrever a pergunta e acertar na segunda tentativa.
- [ ] Rodei o `run_eval.py` de novo e comparei com a etapa 2.

---

### Etapa 4: Voz com LiveKit
**Entrega:** falar uma pergunta e ouvir a resposta, usando o mesmo grafo.

**Aprender:** pipeline de voz em tempo real, latência, interrupção.

Arquitetura:

```
microfone → LiveKit (sala) → VAD → STT → [grafo RAG] → TTS → LiveKit → alto-falante
```

Passos:
1. Conta/projeto no LiveKit (Cloud ou servidor local) e chaves de API.
2. Agente em `voice/agent.py` que entra numa sala.
3. Plugar STT e TTS (escolher provedores com bom suporte a **português**).
4. O "cérebro" do agente chama a lógica de RAG (via função direta ou HTTP para o FastAPI).
5. Testar pelo playground/cliente de exemplo do LiveKit.

**Atenção:**
- A API do LiveKit Agents muda com frequência. **Seguir a documentação oficial atual**, não exemplos antigos.
- Respostas faladas devem ser **curtas**. Considerar um prompt específico para voz (sem listas, sem markdown).
- Medir a latência (fim da fala → início da resposta) e anotar onde está o gargalo (STT, busca, LLM ou TTS).

**Pronto quando:**
- [ ] Faço uma pergunta falada e recebo resposta falada baseada nos documentos.
- [ ] Sei dizer quanto tempo cada estágio leva.
- [ ] Testei interromper o agente no meio da resposta.

---

### Etapa 5: PyTorch (para entender por baixo)
**Entrega:** experimentos independentes na pasta `experiments/`. Não fazem parte do app.

**Aprender:** tensores, o que é um embedding de verdade, loop de treino.

Em ordem de dificuldade:
1. **Tensores e similaridade na mão:** gerar embeddings com `sentence-transformers` e calcular cosseno com `torch` (sem usar o pgvector). Conferir que bate com o resultado do banco.
2. **Visualização:** reduzir os embeddings dos meus chunks para 2D (PCA/UMAP) e plotar. Os temas se agrupam?
3. **Reranker:** colocar um cross-encoder depois do top-k e medir, com o `run_eval.py`, se melhora.
4. **Treinar do zero:** uma rede pequena (ex.: classificar a intenção de uma pergunta) para entender `forward → loss → backward → optimizer.step()`.

**Pronto quando:**
- [ ] Consigo explicar o que é um tensor, o que faz `loss.backward()` e para que serve o otimizador.
- [ ] Tenho ao menos o experimento 1 e o 3 funcionando.

---

## 7. Convenções de código

- Tipagem com *type hints* em todas as funções públicas.
- Rotas finas: a lógica fica em módulos (`rag.py`, `retrieval.py`), não dentro do handler.
- Configuração só via `.env` (nunca chave no código).
- Funções pequenas e puras sempre que possível (fáceis de testar e de eu entender).
- Logs simples mostrando o que foi recuperado (ids e scores). Facilita muito depurar RAG.
- Commits por etapa, com mensagem dizendo o que aprendi, não só o que mudei.

---

## 8. Diário de experimentos

Anotar aqui (ou em `NOTES.md`) cada experimento. Sem isso, não aprendo com o que testei.

```
### [data] Experimento: <nome>
- Hipótese:
- O que mudei:
- Resultado (use o eval quando possível):
- O que aprendi:
```

---

## 9. Problemas comuns (checklist de depuração)

| Sintoma | Causas prováveis |
|---|---|
| Busca devolve resultados sem relação | Chunking ruim, prefixo de embedding errado, modelo não multilíngue, dimensão/vetor desalinhados |
| Resposta ignora o contexto | Prompt fraco, contexto longo demais, instrução de grounding ausente |
| Resposta inventa quando não há contexto | Faltou instrução "diga que não sabe" e/ou limiar mínimo de score |
| Boa resposta, mas fonte errada | Retornando ids errados / chunks reordenados depois da busca |
| Loop infinito no LangGraph | Sem contador de tentativas ou condição de parada |
| Voz com muito atraso | STT/TTS lentos, resposta longa demais, busca/LLM sem streaming |
| Erro de dimensão no pgvector | Mudou o modelo de embedding sem recriar a coluna e reindexar |

---

## 10. Fora do escopo (por enquanto)

Autenticação, multiusuário, front-end bonito, filas, deploy em nuvem, multi-tenant, observabilidade avançada, fine-tuning de LLM. Se bater a vontade de adicionar, anotar em "Ideias futuras" e voltar ao roadmap.

### Ideias futuras
- Busca híbrida (vetorial + texto completo do Postgres).
- Streaming da resposta na API.
- Memória de conversa (histórico) no grafo.
- Ferramentas (tool calling) além da busca nos documentos.
- Avaliação automática com LLM-as-judge.