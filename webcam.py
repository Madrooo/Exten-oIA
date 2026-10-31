import cv2
import numpy as np
import tensorflow as tf

modelo = tf.keras.models.load_model("modelo.keras")
with open("classes.txt") as f:
    classes = f.read().splitlines()

# Número da câmera: 0 costuma ser a webcam do notebook; a GoPro geralmente é 1 ou 2.
# Rode testar_camera.py para descobrir qual número é a GoPro.
CAMERA = 5
# Se a imagem ficar preta no Windows, troque por cv2.CAP_DSHOW
MODO = cv2.CAP_ANY

camera = cv2.VideoCapture(CAMERA, MODO)
if not camera.isOpened():
    print("Não consegui abrir a câmera", CAMERA, "- rode testar_camera.py e confira o número.")
    raise SystemExit

while True:
    ok, frame = camera.read()
    if not ok:
        break

    # OpenCV entrega BGR; o modelo espera RGB 224x224
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img.astype("float32"), axis=0)

    pred = modelo(img, training=False).numpy()[0]
    i = int(np.argmax(pred))
    confianca = float(pred[i])

    texto = f"{classes[i]} ({confianca:.0%})" if confianca >= 0.6 else "incerto"
    cv2.putText(frame, texto, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)
    cv2.imshow("Classificador de lixo (q para sair)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()