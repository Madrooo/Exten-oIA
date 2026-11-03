
import os
import re
import tensorflow as tf

# ==========================================
# CONFIGURAÇÕES DO DATASET
# ==========================================

DATASET_DIR = "Garbage classification"

ARQ_TREINO = "one-indexed-files-notrash_train.txt"
ARQ_VALIDACAO = "one-indexed-files-notrash_val.txt"
ARQ_TESTE = "one-indexed-files-notrash_test.txt"

IMG_SIZE = (224, 224)
BATCH = 32

# Cinco categorias de materiais recicláveis
CLASSES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic"
]


# ==========================================
# LEITURA E PROCESSAMENTO DAS IMAGENS
# ==========================================

def ler_imagem(caminho, rotulo):
    img = tf.io.read_file(caminho)

    img = tf.io.decode_jpeg(
        img,
        channels=3
    )

    img = tf.image.resize(
        img,
        IMG_SIZE
    )

    return img, rotulo


# ==========================================
# CARREGAMENTO DOS DADOS
# ==========================================

def carregar(arquivo, embaralhar=False):

    caminhos = []
    rotulos = []

    if not os.path.isfile(arquivo):
        raise FileNotFoundError(
            f"Arquivo de divisão não encontrado: {arquivo}"
        )

    with open(arquivo, "r") as f:

        for linha in f:

            if not linha.strip():
                continue

            nome = linha.split()[0]

            resultado = re.match(r"^[a-z]+", nome)

            if resultado is None:
                continue

            classe = resultado.group()

            # Ignora categorias que não fazem parte do projeto
            if classe not in CLASSES:
                continue

            caminho = os.path.join(
                DATASET_DIR,
                classe,
                nome
            )

            # Verifica se a imagem realmente existe
            if not os.path.isfile(caminho):
                print(f"Imagem não encontrada: {caminho}")
                continue

            caminhos.append(caminho)

            # Converte a categoria em um número
            rotulos.append(CLASSES.index(classe))

    if not caminhos:
        raise ValueError(
            f"Nenhuma imagem válida encontrada em {arquivo}"
        )

    # Cria o dataset TensorFlow
    dataset = tf.data.Dataset.from_tensor_slices(
        (caminhos, rotulos)
    )

    # Embaralha somente os dados de treinamento
    if embaralhar:
        dataset = dataset.shuffle(
            buffer_size=len(caminhos),
            reshuffle_each_iteration=True
        )

    # Lê e processa as imagens
    dataset = dataset.map(
        ler_imagem,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Agrupa as imagens em lotes
    dataset = dataset.batch(BATCH)

    # Otimiza o carregamento
    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    print(f"\nArquivo: {arquivo}")
    print(f"Imagens carregadas: {len(caminhos)}")

    return dataset


# ==========================================
# CARREGAMENTO GERAL
# ==========================================

def carregar_tudo():

    # Verifica se a estrutura do dataset existe
    if not os.path.isdir(
        os.path.join(DATASET_DIR, "cardboard")
    ):
        raise SystemExit(
            f"Não encontrei '{DATASET_DIR}/cardboard' "
            f"a partir de {os.getcwd()}. "
            "Verifique o DATASET_DIR."
        )

    # Salva os nomes das classes
    with open("classes.txt", "w") as f:
        f.write("\n".join(CLASSES))

    # Carrega os três conjuntos
    treino = carregar(
        ARQ_TREINO,
        embaralhar=True
    )

    validacao = carregar(
        ARQ_VALIDACAO
    )

    teste = carregar(
        ARQ_TESTE
    )

    return treino, validacao, teste
