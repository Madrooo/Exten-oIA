
import cv2
import numpy as np
import tensorflow as tf
import serial
import time
import uuid

from collections import deque, Counter

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# modelo e classes
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

modelo = tf.keras.models.load_model("modelo_mobilenet.keras")

with open("classes.txt") as f:
    classes = [linha.strip().lower() for linha in f if linha.strip()]

print("Classes:", classes)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# configurações
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

CAMERA = 5
MODO = cv2.CAP_ANY

PORTA_ESP = "COM6"
BAUD = 115200

CONFIANCA_MINIMA = 0.80

JANELA_VOTACAO = 15
VOTOS_NECESSARIOS = 12

LIMITE_LARGURA = 900
LIMITE_ALTURA = 600

NOME_JANELA = "Classificador de lixo"

#identificador para cada execução do programa
sessao = uuid.uuid4().hex[:8]
contador = 0

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# conexão com esp8266
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

print("Conectando ao ESP8266...")

esp = serial.Serial(
    PORTA_ESP,
    BAUD,
    timeout=0.1
)

#espera o esp8266 iniciar depois de abrir a conexão
time.sleep(2.5)

esp.reset_input_buffer()

print("ESP8266 conectado!")

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# envio dos resultados pro esp8266
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

def enviar_e_aguardar(id_mensagem, categoria):

    mensagem = f"{id_mensagem}|{categoria}\n"

    for tentativa in range(3):

        print(
            f"Enviando {categoria} "
            f"(tentativa {tentativa + 1}/3)"
        )

        esp.write(mensagem.encode("utf-8"))
        esp.flush()

        inicio = time.time()

        while time.time() - inicio < 5:

            resposta = esp.readline().decode(
                "utf-8",
                errors="replace"
            ).strip()

            if not resposta:
                continue

            print("Resposta ESP8266:", resposta)

            partes = resposta.split("|")

            if (
                len(partes) == 3
                and partes[0] == "DONE"
                and partes[1] == id_mensagem
                and partes[2] == categoria
            ):
                return True

            if resposta.startswith("NACK|"):
                return False

        print("Sem confirmacao. Tentando novamente...")

    return False


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# liberar os leds para a próxima leitura
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

def liberar_esp():

    identificador = f"{sessao}-CLEAR-{contador}"

    mensagem = f"CLEAR|{identificador}\n"

    esp.write(mensagem.encode("utf-8"))
    esp.flush()

    inicio = time.time()

    while time.time() - inicio < 5:

        resposta = esp.readline().decode(
            "utf-8",
            errors="replace"
        ).strip()

        if not resposta:
            continue

        print("Resposta ESP8266:", resposta)

        if resposta == f"CLEARED|{identificador}":
            return True

    return False


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# abrir a camera
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

camera = cv2.VideoCapture(CAMERA, MODO)

if not camera.isOpened():

    print("Nao foi possivel abrir a camera.")

    esp.close()

    raise SystemExit

print("Camera iniciada!")

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# controle da identificação
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

historico = deque(maxlen=JANELA_VOTACAO)

armado = True

#guarda o resultado caso o esp não confirme o recebimento
pendente = None

status = "Aguardando objeto..."

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# processamento principal
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

try:

    while True:

        ok, frame = camera.read()

        if not ok:
            print("Erro ao capturar imagem.")
            break

        #preparação da imagem

        img = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        img = cv2.resize(
            img,
            (224, 224)
        )

        img = np.expand_dims(
            img.astype("float32"),
            axis=0
        )

        #classificação do material

        pred = modelo(
            img,
            training=False
        ).numpy()[0]

        i = int(np.argmax(pred))

        confianca = float(pred[i])

        categoria = classes[i]

        #verifica a votação das 15 imagens

        if confianca >= CONFIANCA_MINIMA:

            historico.append(categoria)

            texto = f"{categoria} ({confianca:.0%})"

        else:

            historico.append(None)

            texto = f"Baixa confianca ({confianca:.0%})"

        #verifica se a classificação é o suficiente para confirmar o material

        if armado and len(historico) == JANELA_VOTACAO:

            votos = Counter(
                item for item in historico
                if item is not None
            )

            if votos:

                melhor_categoria, quantidade = votos.most_common(1)[0]

                if quantidade >= VOTOS_NECESSARIOS:

                    print("\nCLASSIFICACAO CONFIRMADA PELO MODELO!")

                    print("Categoria:", melhor_categoria)

                    print("Votos:", quantidade, "/", JANELA_VOTACAO)

                    #trava para reconhecer um objeto por vez, necessario apertar R para liberar a leitura
                    armado = False

                    contador += 1

                    id_mensagem = f"{sessao}-{contador}"

                    pendente = (
                        id_mensagem,
                        melhor_categoria
                    )

                    status = "Enviando ao ESP8266..."

                    #envia o resultado pro esp8266 e espera a confirmação

                    sucesso = enviar_e_aguardar(
                        id_mensagem,
                        melhor_categoria
                    )

                    if sucesso:

                        print("OPERACAO CONCLUIDA!")

                        status = (
                            "CONFIRMADO PELO ESP8266: "
                            + melhor_categoria
                        )

                        pendente = None

                    else:

                        print("ESP8266 NAO CONFIRMOU!")

                        status = (
                            "SEM CONFIRMACAO - pressione T"
                        )

                    historico.clear()

        #mostra as informações na tela da camera

        cv2.putText(
            frame,
            texto,
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            status,
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 255),
            2
        )

        if armado:

            orientacao = "Aguardando identificacao..."

        elif pendente is not None:

            orientacao = "T: tentar novamente"

        else:

            orientacao = "R: liberar proximo objeto"

        cv2.putText(
            frame,
            orientacao,
            (10, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

        #ajusta o tamanho da imagem da camera

        altura, largura = frame.shape[:2]

        escala = min(
            LIMITE_LARGURA / largura,
            LIMITE_ALTURA / altura,
            1.0
        )

        preview = cv2.resize(
            frame,
            (
                int(largura * escala),
                int(altura * escala)
            ),
            interpolation=cv2.INTER_AREA
        )

        cv2.imshow(NOME_JANELA, preview)

        #controles para liberar a leitura ou encerrar o programa

        tecla = cv2.waitKey(1) & 0xFF

        if tecla == ord("q"):

            break

        #libera a leitura para reconhecer o próximo objeto
        elif tecla == ord("r"):

            if pendente is None:

                status = "Apagando LED anterior..."

                cv2.waitKey(1)

                sucesso = liberar_esp()

                if sucesso:

                    armado = True

                    historico.clear()

                    status = "Novo objeto: analisando..."

                    print("Sistema liberado para novo objeto.")

                else:

                    status = "Nao foi possivel apagar os LEDs!"

                    print("ESP8266 nao confirmou a liberacao.")

            else:

                status = "Aguardando confirmacao do ESP!"

        #tenta enviar novamente o resultado caso não tenha recebido confirmação
        elif tecla == ord("t"):

            if pendente is not None:

                id_mensagem, categoria_pendente = pendente

                sucesso = enviar_e_aguardar(
                    id_mensagem,
                    categoria_pendente
                )

                if sucesso:

                    status = (
                        "CONFIRMADO PELO ESP8266: "
                        + categoria_pendente
                    )

                    pendente = None

                    print("Operacao confirmada!")

                else:

                    status = "SEM CONFIRMACAO - pressione T"

finally:

    camera.release()

    cv2.destroyAllWindows()

    esp.close()

    print("Sistema encerrado.")
