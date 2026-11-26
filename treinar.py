
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

from tensorflow.keras import layers
from dados import carregar_tudo, IMG_SIZE, CLASSES

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score
)

# ==========================================
# 1. CARREGAMENTO DOS DADOS
# ==========================================

print("\n========== CARREGANDO OS DADOS ==========")

treino, validacao, teste = carregar_tudo()

print("Dados carregados com sucesso!")
print(f"Classes: {CLASSES}")
print(f"Imagem: {IMG_SIZE}")
print(f"Total de classes: {len(CLASSES)}")


# ==========================================
# 2. CONTAGEM DE IMAGENS POR CLASSE
# ==========================================

def contar_classes(dataset):
    contagem = np.zeros(len(CLASSES), dtype=int)

    for _, rotulos in dataset:
        rotulos = rotulos.numpy().astype(int).ravel()
        contagem += np.bincount(
            rotulos,
            minlength=len(CLASSES)
        )[:len(CLASSES)]

    return contagem


print("\n========== DISTRIBUIÇÃO DO DATASET ==========")

cont_treino = contar_classes(treino)
cont_validacao = contar_classes(validacao)
cont_teste = contar_classes(teste)

print(f"{'Classe':<15} {'Treino':>10} {'Validação':>12} {'Teste':>10}")

for i, classe in enumerate(CLASSES):
    print(
        f"{classe:<15} "
        f"{cont_treino[i]:>10} "
        f"{cont_validacao[i]:>12} "
        f"{cont_teste[i]:>10}"
    )

print("-" * 50)
print(
    f"{'TOTAL':<15} "
    f"{sum(cont_treino):>10} "
    f"{sum(cont_validacao):>12} "
    f"{sum(cont_teste):>10}"
)


# ==========================================
# 3. CONSTRUÇÃO DA MOBILENETV2
# ==========================================

print("\n========== CONSTRUINDO O MODELO ==========")

base = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)

base.trainable = False

entrada = tf.keras.Input(shape=IMG_SIZE + (3,))

# Aumento de dados
x = layers.RandomFlip("horizontal")(entrada)
x = layers.RandomRotation(0.1)(x)
x = layers.RandomZoom(0.1)(x)

# Normalização para [-1, 1]
x = layers.Rescaling(1 / 127.5, offset=-1)(x)

# Extração de características
x = base(x, training=False)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)

# Classificação das 6 classes
saida = layers.Dense(
    len(CLASSES),
    activation="softmax"
)(x)

modelo = tf.keras.Model(entrada, saida)

print("\n========== RESUMO DA ARQUITETURA ==========")

modelo.summary()

print(f"\nParâmetros totais: {modelo.count_params():,}")
print(
    f"Parâmetros treináveis: "
    f"{sum(np.prod(p.shape) for p in modelo.trainable_weights):,}"
)


# ==========================================
# 4. ETAPA 1 - TREINAMENTO INICIAL
# ==========================================

print("\n==========================================")
print(" ETAPA 1: TREINAMENTO DA CAMADA FINAL")
print("==========================================")

print("Épocas: 8")
print("Learning rate: 0.001")
print("MobileNetV2 congelada.")

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

inicio1 = time.time()

historico1 = modelo.fit(
    treino,
    validation_data=validacao,
    epochs=8,
    verbose=1
)

tempo1 = time.time() - inicio1

print(f"\nEtapa 1 concluída em {tempo1 / 60:.2f} minutos.")


# ==========================================
# 5. ETAPA 2 - FINE-TUNING
# ==========================================

print("\n==========================================")
print(" ETAPA 2: AJUSTE FINO (FINE-TUNING)")
print("==========================================")

base.trainable = True

for camada in base.layers[:-30]:
    camada.trainable = False

print("Últimas 30 camadas liberadas.")
print("Épocas: 5")
print("Learning rate: 0.00001")

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

inicio2 = time.time()

historico2 = modelo.fit(
    treino,
    validation_data=validacao,
    epochs=5,
    verbose=1
)

tempo2 = time.time() - inicio2

print(f"\nEtapa 2 concluída em {tempo2 / 60:.2f} minutos.")


# ==========================================
# 6. GRÁFICOS DE TREINAMENTO
# ==========================================

print("\n========== GERANDO GRÁFICOS ==========")

acuracia = (
    historico1.history["accuracy"] +
    historico2.history["accuracy"]
)

val_acuracia = (
    historico1.history["val_accuracy"] +
    historico2.history["val_accuracy"]
)

perda = (
    historico1.history["loss"] +
    historico2.history["loss"]
)

val_perda = (
    historico1.history["val_loss"] +
    historico2.history["val_loss"]
)

epocas = range(1, len(acuracia) + 1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epocas, acuracia, label="Treinamento")
plt.plot(epocas, val_acuracia, label="Validação")
plt.title("Acurácia por época")
plt.xlabel("Épocas")
plt.ylabel("Acurácia")
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(epocas, perda, label="Treinamento")
plt.plot(epocas, val_perda, label="Validação")
plt.title("Perda por época")
plt.xlabel("Épocas")
plt.ylabel("Perda")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("curvas_mobilenet.png")
plt.show()


# ==========================================
# 7. AVALIAÇÃO DO CONJUNTO DE TESTE
# ==========================================

print("\n========== AVALIAÇÃO DO MODELO ==========")

perda_teste, acc_teste = modelo.evaluate(teste, verbose=1)

print(f"\nAcurácia no teste: {acc_teste:.2%}")
print(f"Perda no teste: {perda_teste:.4f}")


# ==========================================
# 8. F1-SCORE E TEMPO DE INFERÊNCIA
# ==========================================

print("\n========== MÉTRICAS DE CLASSIFICAÇÃO ==========")

y_real = []
y_previsto = []
tempos_inferencia = []

# Aquecimento do modelo para evitar medir
# o custo inicial de execução
primeiro_lote = next(iter(teste))[0]

for i in range(min(10, len(primeiro_lote))):
    modelo(
        primeiro_lote[i:i+1],
        training=False
    ).numpy()

# Classificação e medição individual
for imagens, rotulos in teste:

    imagens_np = imagens.numpy()
    rotulos_np = rotulos.numpy().astype(int).ravel()

    for imagem, rotulo in zip(imagens_np, rotulos_np):

        inicio = time.perf_counter()

        previsao = modelo(
            imagem[np.newaxis, ...],
            training=False
        ).numpy()[0]

        fim = time.perf_counter()

        tempos_inferencia.append((fim - inicio) * 1000)

        y_real.append(rotulo)
        y_previsto.append(np.argmax(previsao))


# F1-score macro: média entre as seis classes
f1 = f1_score(
    y_real,
    y_previsto,
    average="macro",
    zero_division=0
)

tempo_medio = np.mean(tempos_inferencia)

print(f"F1-score (média macro): {f1:.4f}")
print(f"F1-score em porcentagem: {f1:.2%}")
print(f"Tempo médio de inferência: {tempo_medio:.2f} ms/imagem")

print("\nRelatório detalhado por classe:")

print(classification_report(
    y_real,
    y_previsto,
    labels=list(range(len(CLASSES))),
    target_names=CLASSES,
    zero_division=0
))


# ==========================================
# 9. MATRIZ DE CONFUSÃO
# ==========================================

print("\n========== MATRIZ DE CONFUSÃO ==========")

matriz = confusion_matrix(
    y_real,
    y_previsto,
    labels=list(range(len(CLASSES)))
)

plt.figure(figsize=(9, 7))

plt.imshow(matriz, interpolation="nearest", cmap="Blues")
plt.title("Matriz de Confusão - MobileNetV2")
plt.colorbar()

posicoes = np.arange(len(CLASSES))

plt.xticks(posicoes, CLASSES, rotation=45)
plt.yticks(posicoes, CLASSES)

plt.xlabel("Classe prevista")
plt.ylabel("Classe real")

for i in range(len(CLASSES)):
    for j in range(len(CLASSES)):
        plt.text(
            j, i, str(matriz[i, j]),
            ha="center", va="center"
        )

plt.tight_layout()
plt.savefig("matriz_confusao_mobilenet.png")
plt.show()


# ==========================================
# 10. RESUMO FINAL
# ==========================================

print("\n==========================================")
print(" RESUMO FINAL DO TREINAMENTO")
print("==========================================")

print(f"Acurácia no teste: {acc_teste:.2%}")
print(f"F1-score médio: {f1:.2%}")
print(f"Perda no teste: {perda_teste:.4f}")
print(f"Inferência por imagem: {tempo_medio:.2f} ms")
print(f"Tempo de treinamento: {(tempo1 + tempo2) / 60:.2f} minutos")


# ==========================================
# 11. SALVAMENTO
# ==========================================

print("\n========== SALVANDO O MODELO ==========")

modelo.save("modelo_mobilenet.keras")

print("Modelo salvo com sucesso!")
print("Arquivo: modelo_mobilenet.keras")

print("\n========== PROCESSO FINALIZADO ==========")
