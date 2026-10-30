import tensorflow as tf
from tensorflow.keras import layers
from dados import carregar_tudo, IMG_SIZE, CLASSES

# Divisão oficial do Kaggle: treino / validação / teste
treino, validacao, teste = carregar_tudo()

# Modelo: aumento de dados + MobileNetV2 pré-treinada (uma CNN) + camada final de 6 classes
base = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
base.trainable = False

entrada = tf.keras.Input(shape=IMG_SIZE + (3,))
x = layers.RandomFlip("horizontal")(entrada)
x = layers.RandomRotation(0.1)(x)
x = layers.RandomZoom(0.1)(x)
x = layers.Rescaling(1 / 127.5, offset=-1)(x)  # normaliza para [-1, 1]
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
saida = layers.Dense(len(CLASSES), activation="softmax")(x)
modelo = tf.keras.Model(entrada, saida)

# Etapa 1: treina só a camada final
modelo.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
               loss="sparse_categorical_crossentropy", metrics=["accuracy"])
modelo.fit(treino, validation_data=validacao, epochs=8)

# Etapa 2: ajuste fino das últimas camadas da MobileNetV2
base.trainable = True
for camada in base.layers[:-30]:
    camada.trainable = False
modelo.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
               loss="sparse_categorical_crossentropy", metrics=["accuracy"])
modelo.fit(treino, validation_data=validacao, epochs=5)

# Avaliação final no conjunto de teste (imagens que o modelo nunca viu)
perda, acc = modelo.evaluate(teste)
print(f"Acurácia no teste: {acc:.2%}")

modelo.save("modelo.keras")
print("Modelo salvo em modelo.keras")