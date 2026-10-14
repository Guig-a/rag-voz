"""Experimento 4: treinar do zero uma rede pequena que classifica a intencao de uma frase.

Entrada: embedding congelado (384 numeros) do multilingual-e5-small.
Saida: uma de tres intencoes. Os pesos da rede comecam aleatorios e sao ajustados
pelo loop forward -> loss -> backward -> optimizer.step().

Uso: python -m experiments.04_train_intent
"""

import torch
from torch import nn

from app.config import get_settings
from app.embeddings import get_embedding_model

INTENTS = ["pergunta", "saudacao", "despedida"]

TRAIN = [
    ("O que e busca vetorial?", "pergunta"),
    ("Como funciona o indice HNSW?", "pergunta"),
    ("Qual a temperatura da agua para o cafe?", "pergunta"),
    ("Para que serve o overlap?", "pergunta"),
    ("Qual o maior planeta?", "pergunta"),
    ("Me explica o que e RAG", "pergunta"),
    ("Quanto cafe uso por litro?", "pergunta"),
    ("Por que Marte e vermelho?", "pergunta"),
    ("Oi, tudo bem?", "saudacao"),
    ("Ola!", "saudacao"),
    ("Bom dia", "saudacao"),
    ("E ai, beleza?", "saudacao"),
    ("Boa tarde, assistente", "saudacao"),
    ("Oi, voce esta ai?", "saudacao"),
    ("Tchau", "despedida"),
    ("Ate logo", "despedida"),
    ("Obrigado, era so isso", "despedida"),
    ("Pode encerrar", "despedida"),
    ("Valeu, ate mais", "despedida"),
    ("Falou, tchau tchau", "despedida"),
]

TEST = [
    ("Qual operador calcula a distancia de cosseno?", "pergunta"),
    ("O que e um chunk?", "pergunta"),
    ("Boa noite!", "saudacao"),
    ("Opa, tudo certo?", "saudacao"),
    ("Brigado, pode desligar", "despedida"),
    ("Ate a proxima", "despedida"),
]

EPOCHS = 200
LEARNING_RATE = 0.01


class IntentClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def encode(texts: list[str]) -> torch.Tensor:
    settings = get_settings()
    model = get_embedding_model(settings.embedding_model)
    prefixed = [f"{settings.embedding_query_prefix} {text}" for text in texts]
    return model.encode(prefixed, convert_to_tensor=True, normalize_embeddings=True).cpu()


def to_dataset(examples: list[tuple[str, str]]) -> tuple[torch.Tensor, torch.Tensor]:
    features = encode([text for text, _ in examples])
    labels = torch.tensor([INTENTS.index(label) for _, label in examples])
    return features, labels


def accuracy(model: nn.Module, features: torch.Tensor, labels: torch.Tensor) -> float:
    with torch.no_grad():
        predictions = model(features).argmax(dim=1)
    return float((predictions == labels).float().mean())


def main() -> None:
    torch.manual_seed(0)

    train_x, train_y = to_dataset(TRAIN)
    test_x, test_y = to_dataset(TEST)
    print(f"treino: x={tuple(train_x.shape)} y={tuple(train_y.shape)} | teste: x={tuple(test_x.shape)}")

    model = IntentClassifier(input_dim=train_x.shape[1], hidden_dim=32, num_classes=len(INTENTS))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print(f"parametros treinaveis: {sum(p.numel() for p in model.parameters())}")
    print(f"acuracia antes de treinar (pesos aleatorios): treino={accuracy(model, train_x, train_y):.0%}")

    for epoch in range(1, EPOCHS + 1):
        logits = model(train_x)
        loss = loss_fn(logits, train_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 25 == 0:
            print(
                f"epoca {epoch:>3}  loss={loss.item():.4f}  "
                f"acuracia treino={accuracy(model, train_x, train_y):.0%}  "
                f"teste={accuracy(model, test_x, test_y):.0%}"
            )

    print()
    print("previsoes no conjunto de teste:")
    with torch.no_grad():
        probabilities = torch.softmax(model(test_x), dim=1)
    for (text, expected), probs in zip(TEST, probabilities):
        predicted = INTENTS[int(probs.argmax())]
        status = "OK" if predicted == expected else "ERRO"
        print(f"  [{status}] {text!r} -> {predicted} ({float(probs.max()):.0%})")


if __name__ == "__main__":
    main()
