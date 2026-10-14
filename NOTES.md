# Diário de experimentos

Corpus usado nos experimentos: 14 chunks (4 documentos de `data/` + 2 chunks de teste da etapa 1), modelo `intfloat/multilingual-e5-small` (384 dimensões).

### [2026-10-14] Experimento 1: cosseno na mão com torch
- Hipótese: o score do pgvector (`1 - (embedding <=> query)`) é exatamente a similaridade de cosseno que eu consigo calcular com tensores.
- O que mudei: nada no app. Li os embeddings direto do banco e calculei `(A @ q) / (|A| * |q|)` com `torch`.
- Resultado: mesma ordem de ids no top-5 e diferença máxima de ~1e-8 (arredondamento de float32). Como os vetores são normalizados (norma 1), o produto escalar puro já é o cosseno.
- O que aprendi: o pgvector não faz mágica; é álgebra linear. Normalizar os embeddings na ingestão permite trocar cosseno por produto escalar, que é mais barato.

### [2026-10-14] Experimento 2: embeddings em 2D (PCA)
- Hipótese: chunks do mesmo documento ficam próximos no espaço vetorial.
- O que mudei: PCA feito na mão com `torch.linalg.svd`, gráfico em `experiments/output/embeddings_pca.png`.
- Resultado: quatro grupos nítidos. Café e Sistema Solar ficam isolados; busca vetorial, RAG e os chunks de teste ficam vizinhos (assuntos técnicos parecidos). Cosseno médio: 0.888 dentro do mesmo documento contra 0.806 entre documentos diferentes. PC1 + PC2 explicam só 41% da variância.
- O que aprendi: os temas se agrupam, mas os scores do e5 são comprimidos numa faixa alta (quase tudo acima de 0.75). Um gráfico 2D esconde mais da metade da informação.

### [2026-10-14] Experimento 3: reranker cross-encoder
- Hipótese: reordenar o top-10 com um cross-encoder (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`) coloca o chunk certo em 1º com mais frequência.
- O que mudei: experimento isolado; mede só a recuperação dos 9 casos com resposta do `eval/questions.jsonl` (sem LLM, para não depender da cota do Gemini).
- Resultado:

  | | hit@1 | hit@3 | MRR |
  |---|---|---|---|
  | busca vetorial | 8/9 | 9/9 | 0.944 |
  | com reranker | 8/9 | 9/9 | 0.926 |

  Custo: ~160 ms por pergunta em CPU. Na separação entre "tem resposta" e "não tem", o reranker foi muito melhor: perguntas sem resposta ficaram entre -6 e -7, contra -4.5 a 8 das com resposta. No cosseno, "O corpus menciona LangGraph?" (sem resposta) teve 0.85, acima de "Qual banco aparece no corpus?" (com resposta, 0.81).
- O que aprendi: num corpus pequeno e de temas bem separados, a busca vetorial já ordena bem e o reranker não ajuda na ordem. O ganho real dele aqui seria como **filtro de relevância**: um limiar no score do cross-encoder separa "não sei" muito melhor que o `RAG_MIN_SCORE` no cosseno, e poderia substituir a chamada de LLM do `grade_context`.

### [2026-10-14] Experimento 4: treinar uma rede do zero
- Hipótese: uma rede pequena aprende a classificar intenção (pergunta, saudação, despedida) com 20 exemplos, usando embeddings congelados como entrada.
- O que mudei: MLP 384 → 32 → 3 (12.419 parâmetros), `CrossEntropyLoss`, `Adam` com lr 0.01, 200 épocas.
- Resultado: loss de 1.09 para 0.0007; acurácia de 40% (pesos aleatórios) para 100% no treino e 100% nas 6 frases de teste nunca vistas. Menor confiança: "Opa, tudo certo?" → saudação com 71%.
- O que aprendi: `forward` calcula a previsão, a loss mede o erro, `loss.backward()` calcula o gradiente de cada peso e `optimizer.step()` anda os pesos na direção que reduz o erro. Com embeddings bons, pouca rede e poucos dados bastam; o conjunto de teste é pequeno demais para confiar nos 100%.
