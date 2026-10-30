import re
import tensorflow as tf

# Pasta com as imagens (contém as subpastas cardboard, glass, metal, paper, plastic, trash)
DATASET_DIR = "Garbage classification/"

# Arquivos .txt do Kaggle com a divisão oficial (coloque-os na mesma pasta dos scripts)
ARQ_TREINO = "one-indexed-files-notrash_train.txt"
ARQ_VALIDACAO = "one-indexed-files-notrash_val.txt"
ARQ_TESTE = "one-indexed-files-notrash_test.txt"

IMG_SIZE = (224, 224)
BATCH = 32
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


def ler_imagem(caminho, rotulo):
    img = tf.io.read_file(caminho)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    return img, rotulo


def carregar(arquivo, embaralhar=False):
    caminhos, rotulos = [], []
    for linha in open(arquivo):
        if not linha.strip():
            continue
        nome = linha.split()[0]                    # ex.: "paper70.jpg"
        classe = re.match(r"[a-z]+", nome).group()  # ex.: "paper"
        caminhos.append(f"{DATASET_DIR}/{classe}/{nome}")
        rotulos.append(CLASSES.index(classe))

    ds = tf.data.Dataset.from_tensor_slices((caminhos, rotulos))
    if embaralhar:
        ds = ds.shuffle(len(caminhos))
    ds = ds.map(ler_imagem, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)


def carregar_tudo():
    with open("classes.txt", "w") as f:
        f.write("\n".join(CLASSES))
    treino = carregar(ARQ_TREINO, embaralhar=True)
    validacao = carregar(ARQ_VALIDACAO)
    teste = carregar(ARQ_TESTE)
    return treino, validacao, teste