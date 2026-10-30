import tensorflow as tf
from tensorflow.keras import layers
from dados import carregar_tudo, IMG_SIZE, CLASSES

# Divisão oficial do Kaggle: treino / validação / teste
treino, validacao, teste = carregar_tudo()

# CNN construída do zero: 4 blocos Conv2D + MaxPooling, depois camadas densas
modelo = tf.keras.Sequential([
    tf.keras.Input(shape=IMG_SIZE + (3,)),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.Rescaling(1 / 255),

    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dropout(0.4),
    layers.Dense(128, activation="relu"),
    layers.Dense(len(CLASSES), activation="softmax"),
])

modelo.compile(optimizer="adam",
               loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Para de treinar quando a validação para de melhorar e guarda o melhor ponto
parada = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy", patience=6, restore_best_weights=True)
modelo.fit(treino, validation_data=validacao, epochs=40, callbacks=[parada])

# Avaliação final no conjunto de teste (imagens que o modelo nunca viu)
perda, acc = modelo.evaluate(teste)
print(f"Acurácia no teste: {acc:.2%}")

modelo.save("modelo.keras")
print("Modelo salvo em modelo.keras")