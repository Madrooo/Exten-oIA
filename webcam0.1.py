import os
import time
import cv2
import numpy as np
import tensorflow as tf

modelo = tf.keras.models.load_model("modelo.keras")
with open("classes.txt") as f:
    classes = f.read().splitlines()


# Compilar a previsão com tf.function deixa cada classificação muito mais rápida
@tf.function
def prever(x):
    return modelo(x, training=False)


# Primeira chamada: prepara o modelo antes de abrir a câmera (evita um travamento no início)
prever(np.zeros((1, 224, 224, 3), dtype="float32"))

# Número da câmera: 0 costuma ser a webcam do notebook; a GoPro geralmente é 1 ou 2.
# Rode testar_camera.py para descobrir qual número é a GoPro.
CAMERA = 0
# Se a imagem ficar preta no Windows, troque por cv2.CAP_DSHOW
MODO = cv2.CAP_ANY
# Classifica 1 a cada PULAR quadros: o vídeo fica mais fluido (use 1 para classificar todos)
PULAR = 5

# ---------- Arduino ----------
# Porta serial do Arduino (ex.: "COM3"; veja no Arduino IDE ou no Gerenciador de Dispositivos).
# Com None, o programa só mostra o que enviaria (útil para testar sem o Arduino).
PORTA = None
# Caractere enviado ao Arduino para cada classe (o sketch do Arduino interpreta estes códigos)
CODIGOS = {"cardboard": "0", "glass": "1", "metal": "2", "paper": "3", "plastic": "4", "trash": "5"}
LIMIAR = 0.85      # confiança mínima para a previsão contar
CONFIRMACOES = 6   # previsões seguidas iguais antes de enviar
ESPERA = 6         # segundos sem enviar depois de um envio (tempo do servo abrir e fechar)

arduino = None
if PORTA:
    import serial  # pip install pyserial
    try:
        arduino = serial.serial_for_url(PORTA, 9600, timeout=1)
    except Exception as erro:
        print("Não consegui abrir a porta", PORTA, "->", erro)
        raise SystemExit
    time.sleep(2)  # o Arduino reinicia quando a porta é aberta


def enviar(classe):
    codigo = CODIGOS[classe]
    if arduino:
        arduino.write(codigo.encode())
    print("Enviado ao Arduino:" if arduino else "Simulação, enviaria:", codigo, f"({classe})")


camera = cv2.VideoCapture(CAMERA, MODO)
if not camera.isOpened():
    print("Não consegui abrir a câmera", CAMERA, "- rode testar_camera.py e confira o número.")
    raise SystemExit

contador = 0
texto = "..."
classe_atual = None      # classe mostrada na tela (None quando está "incerto")
ultima = None            # última classe confiante
repeticoes = 0           # quantas previsões seguidas iguais
ultimo_envio = 0
enviado = ""
anterior = time.time()

while True:
    ok, frame = camera.read()
    if not ok:
        break

    if contador % PULAR == 0:
        # OpenCV entrega BGR; o modelo espera RGB 224x224
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = np.expand_dims(img.astype("float32"), axis=0)

        pred = prever(img).numpy()[0]
        i = int(np.argmax(pred))
        confianca = float(pred[i])
        classe_atual = classes[i] if confianca >= 0.6 else None
        texto = f"{classes[i]} ({confianca:.0%})" if confianca >= 0.6 else "incerto"

        # Envio automático: só depois de várias previsões confiantes e iguais
        if confianca >= LIMIAR:
            repeticoes = repeticoes + 1 if classes[i] == ultima else 1
            ultima = classes[i]
        else:
            ultima, repeticoes = None, 0

        if repeticoes >= CONFIRMACOES and time.time() - ultimo_envio > ESPERA:
            enviar(ultima)
            enviado = f"enviado: {ultima}"
            ultimo_envio = time.time()
            repeticoes = 0
    contador += 1

    # FPS do vídeo (se ficar baixo mesmo assim, o gargalo é a câmera)
    agora = time.time()
    fps = 1 / max(agora - anterior, 1e-6)
    anterior = agora

    cv2.putText(frame, texto, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)
    captura = frame.copy()  # versão sem o FPS, para salvar com a tecla s
    cv2.putText(frame, f"FPS: {fps:.0f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2)
    cv2.putText(frame, enviado, (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 200, 0), 2)
    cv2.imshow("Classificador de lixo (q sair, s salvar, e enviar agora)", frame)

    tecla = cv2.waitKey(1) & 0xFF
    if tecla == ord("q"):
        break
    if tecla == ord("s"):
        os.makedirs("imagens", exist_ok=True)
        nome = f"imagens/captura_{int(time.time())}.png"
        cv2.imwrite(nome, captura)
        print("Imagem salva em", nome)
    if tecla == ord("e") and classe_atual:
        # Envio manual da classe mostrada na tela (bom para demonstrações)
        enviar(classe_atual)
        enviado = f"enviado: {classe_atual}"
        ultimo_envio = time.time()
        repeticoes = 0

camera.release()
cv2.destroyAllWindows()
if arduino:
    arduino.close()