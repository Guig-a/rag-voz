# Busca vetorial com PostgreSQL

O PostgreSQL ganha busca vetorial com a extensão pgvector. Ela adiciona o tipo vector, que guarda um embedding como uma lista de números de ponto flutuante com dimensão fixa, por exemplo vector(384).

Para comparar vetores, o pgvector oferece operadores de distância. O operador <=> calcula a distância de cosseno, o operador <-> calcula a distância euclidiana e o operador <#> calcula o produto interno negativo. Como <=> devolve uma distância, a similaridade de cosseno é obtida com a conta 1 menos a distância.

Sem índice, o banco compara a consulta com todas as linhas da tabela, o que é exato mas fica lento com milhões de vetores. Para acelerar, o pgvector oferece o índice HNSW, um grafo de vizinhança em camadas que encontra vizinhos aproximados muito rápido. Outra opção é o índice IVFFlat, que agrupa os vetores em listas e exige treinar os grupos depois de carregar os dados.

O índice precisa usar a mesma métrica da consulta. Quem busca com <=> deve criar o índice com vector_cosine_ops. Se o modelo de embedding mudar, a dimensão da coluna também muda, e todos os documentos precisam ser reindexados.
