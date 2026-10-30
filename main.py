import tensorflow as tf
from tensorflow.keras import layers

# Pasta que contém as 6 subpastas (cardboard, glass, metal, paper, plastic, trash).
# No dataset do Kaggle costuma ser: "Garbage classification/Garbage classification"
DATASET_DIR = "Garbage classification/Garbage classification"
IMG_SIZE = (224, 224)
BATCH = 32

# 80% treino / 20% validação
treino = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR, validation_split=0.2, subset="training",
    seed=123, image_size=IMG_SIZE, batch_size=BATCH)
validacao = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR, validation_split=0.2, subset="validation",
    seed=123, image_size=IMG_SIZE, batch_size=BATCH)

classes = treino.class_names
print("Classes:", classes)
with open("classes.txt", "w") as f:
    f.write("\n".join(classes))

treino = treino.prefetch(tf.data.AUTOTUNE)
validacao = validacao.prefetch(tf.data.AUTOTUNE)

# Modelo: aumento de dados + MobileNetV2 pré-treinada + camada final de 6 classes
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
saida = layers.Dense(len(classes), activation="softmax")(x)
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

modelo.save("modelo.keras")
print("Modelo salvo em modelo.keras")