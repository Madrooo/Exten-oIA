
import tensorflow as tf
import numpy as np
import time

from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix
)

from dados import carregar_tudo, CLASSES


# ==========================================
# 1. CARREGAMENTO DO MODELO E DOS DADOS
# ==========================================

print("\n==========================================")
print(" AVALIAÇÃO FINAL - CNN DO ZERO")
print("==========================================")

print("\nCarregando modelo...")

modelo = tf.keras.models.load_model("modelo.keras")

print("Modelo carregado com sucesso!")

print("\nCarregando conjunto de teste...")

treino, validacao, teste = carregar_tudo()

print(f"Classes: {CLASSES}")
print(f"Quantidade de classes: {len(CLASSES)}")


# ==========================================
# 2. AVALIAÇÃO DE ACURÁCIA E PERDA
# ==========================================

print("\n========== AVALIAÇÃO DO TESTE ==========")

perda, acuracia = modelo.evaluate(
    teste,
    verbose=1
)

print(f"\nAcurácia no teste: {acuracia:.2%}")
print(f"Perda no teste: {perda:.4f}")


# ==========================================
# 3. PREPARAÇÃO DAS PREVISÕES
# ==========================================

print("\n========== REALIZANDO CLASSIFICAÇÕES ==========")

y_real = []
y_previsto = []
tempos = []

# Aquecimento do modelo

primeiro_lote = next(iter(teste))[0]

for i in range(min(10, len(primeiro_lote))):
    modelo(
        primeiro_lote[i:i+1],
        training=False
    ).numpy()

print("Aquecimento concluído!")
print("Classificando imagens do conjunto de teste...")


# ==========================================
# 4. TEMPO DE INFERÊNCIA POR IMAGEM
# ==========================================

total_imagens = 0

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

        tempo_ms = (fim - inicio) * 1000

        tempos.append(tempo_ms)

        y_real.append(rotulo)
        y_previsto.append(np.argmax(previsao))

        total_imagens += 1

        if total_imagens % 50 == 0:
            print(f"Imagens processadas: {total_imagens}")


# ==========================================
# 5. CÁLCULO DO F1-SCORE
# ==========================================

print("\n========== RESULTADO DO F1-SCORE ==========")

f1_macro = f1_score(
    y_real,
    y_previsto,
    average="macro",
    zero_division=0
)

print(f"F1-score macro: {f1_macro:.4f}")
print(f"F1-score percentual: {f1_macro:.2%}")


# ==========================================
# 6. TEMPO MÉDIO DE INFERÊNCIA
# ==========================================

print("\n========== TEMPO DE INFERÊNCIA ==========")

tempo_medio = np.mean(tempos)
tempo_minimo = np.min(tempos)
tempo_maximo = np.max(tempos)

print(f"Imagens classificadas: {total_imagens}")
print(f"Tempo médio: {tempo_medio:.3f} ms/imagem")
print(f"Tempo mínimo: {tempo_minimo:.3f} ms")
print(f"Tempo máximo: {tempo_maximo:.3f} ms")


# ==========================================
# 7. RELATÓRIO POR CLASSE
# ==========================================

print("\n========== RELATÓRIO POR CLASSE ==========")

print(
    classification_report(
        y_real,
        y_previsto,
        labels=list(range(len(CLASSES))),
        target_names=CLASSES,
        zero_division=0
    )
)


# ==========================================
# 8. MATRIZ DE CONFUSÃO
# ==========================================

print("\n========== MATRIZ DE CONFUSÃO ==========")

matriz = confusion_matrix(
    y_real,
    y_previsto,
    labels=list(range(len(CLASSES)))
)

print("Linhas: classes reais")
print("Colunas: classes previstas")
print(matriz)


# ==========================================
# 9. RESUMO FINAL
# ==========================================

print("\n==========================================")
print(" RESULTADOS FINAIS - CNN DO ZERO")
print("==========================================")

print(f"Acurácia no teste: {acuracia:.2%}")
print(f"F1-score macro: {f1_macro:.2%}")
print(f"Tempo médio de inferência: {tempo_medio:.3f} ms/imagem")
print(f"Total de imagens avaliadas: {total_imagens}")

print("\n========== AVALIAÇÃO CONCLUÍDA ==========")
