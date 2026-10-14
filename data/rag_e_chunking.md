# RAG e chunking

RAG significa geração aumentada por recuperação. Em vez de confiar apenas no que o modelo de linguagem aprendeu no treino, o sistema primeiro busca trechos relevantes nos documentos e depois coloca esses trechos no prompt, pedindo que a resposta se baseie somente neles.

Antes de indexar, cada documento é dividido em pedaços chamados chunks. Um chunk grande demais mistura vários assuntos e dilui o significado do embedding. Um chunk pequeno demais perde o contexto necessário para responder. Tamanhos entre trezentos e oitocentos caracteres costumam ser um bom ponto de partida.

A sobreposição entre chunks vizinhos, chamada de overlap, repete o final de um pedaço no começo do próximo. Ela serve para evitar cortar uma ideia no meio, já que uma frase importante pode cair exatamente na fronteira entre dois chunks.

O parâmetro top-k define quantos chunks são recuperados por pergunta. Um valor baixo pode deixar de fora a informação certa, e um valor alto enche o prompt de ruído. Para reduzir alucinações, o prompt deve mandar o modelo dizer que não sabe quando o contexto não for suficiente, e o sistema pode exigir um score mínimo de similaridade.
