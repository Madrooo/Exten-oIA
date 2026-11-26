
import cv2
import numpy as np
import tensorflow as tf
import serial
import time

# ==========================================
# 1. CARREGAR MODELO E CLASSES
# ==========================================

modelo = tf.keras.models.load_model("modelo.keras")

with open("classes.txt") as f:
    classes = f.read().splitlines()

# ==========================================
# 2. CONFIGURACOES
# ==========================================

CAMERA = 5
MODO = cv2.CAP_ANY

PORTA_ESP = "COM6"
BAUD = 115200

CONFIANCA_MINIMA = 0.60

# Limites da janela da camera
LIMITE_LARGURA = 900
LIMITE_ALTURA = 600

NOME_JANELA = "Classificador de lixo (q para sair)"

# ==========================================
# 3. CONECTAR AO ESP8266
# ==========================================

print("Conectando ao ESP8266...")

esp = serial.Serial(PORTA_ESP, BAUD, timeout=0.1)

# Aguarda a inicializacao do ESP8266
time.sleep(2)

esp.reset_input_buffer()

print("ESP8266 conectado!")

# ==========================================
# 4. INICIAR CAMERA
# ==========================================

camera = cv2.VideoCapture(CAMERA, MODO)

if not camera.isOpened():
    print("Nao consegui abrir a camera:", CAMERA)
    esp.close()
    raise SystemExit

print("Camera iniciada!")
print("Sistema de identificacao iniciado.")

# Evita enviar repetidamente a mesma classificacao
ultimo_estado = None

# ==========================================
# 5. PROCESSAMENTO
# ==========================================

try:

    while True:

        ok, frame = camera.read()

        if not ok:
            print("Erro ao capturar imagem.")
            break

        # ----------------------------------
        # PREPARAR IMAGEM PARA O MODELO
        # ----------------------------------

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (224, 224))

        img = np.expand_dims(
            img.astype("float32"),
            axis=0
        )

        # ----------------------------------
        # IDENTIFICACAO DO LIXO
        # ----------------------------------

        pred = modelo(img, training=False).numpy()[0]

        i = int(np.argmax(pred))

        confianca = float(pred[i])

        if confianca >= CONFIANCA_MINIMA:

            estado = classes[i]

            texto = f"{estado} ({confianca:.0%})"

        else:

            estado = "incerto"

            texto = "Incerto"

        # ----------------------------------
        # COMUNICACAO COM ESP8266
        # ----------------------------------

        if estado != ultimo_estado:

            mensagem = estado + "\n"

            esp.write(mensagem.encode("utf-8"))

            print("Enviado ao ESP8266:", estado)

            ultimo_estado = estado

        # ----------------------------------
        # MOSTRAR RESULTADO NA CAMERA
        # ----------------------------------

        cv2.putText(
            frame,
            texto,
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # ----------------------------------
        # REDIMENSIONAR SEM DISTORCER
        # ----------------------------------

        altura, largura = frame.shape[:2]

        escala = min(
            LIMITE_LARGURA / largura,
            LIMITE_ALTURA / altura,
            1.0
        )

        nova_largura = int(largura * escala)
        nova_altura = int(altura * escala)

        preview = cv2.resize(
            frame,
            (nova_largura, nova_altura),
            interpolation=cv2.INTER_AREA
        )

        cv2.imshow(NOME_JANELA, preview)

        # ----------------------------------
        # SAIR COM A TECLA Q
        # ----------------------------------

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:

    # ======================================
    # 6. ENCERRAR SISTEMA
    # ======================================

    camera.release()

    cv2.destroyAllWindows()

    esp.close()

    print("Sistema encerrado.")
