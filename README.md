# Lixeira Inteligente: Classificação de Lixo Reciclável com Visão Computacional

![Python](https://img.shields.io/badge/Python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-câmera-green)
![Status](https://img.shields.io/badge/status-em%20desenvolvimento-yellow)

Projeto de extensão da disciplina de **Inteligência Artificial** (IFCE). O sistema reconhece, em tempo real e pela câmera, o tipo de material de um resíduo (papelão, vidro, metal, papel, plástico ou lixo comum) usando uma **rede neural convolucional (CNN)** treinada com o dataset *Garbage Classification* do Kaggle.

Esta etapa entrega o **reconhecimento visual**. A parte física, com lixeiras impressas em 3D, Arduino e servo motores, é a continuação planejada (veja [Protótipo físico](#protótipo-físico-e-próximos-passos)).

<!-- COLOQUE AQUI (01_capa.png): uma foto ou montagem de capa do projeto, por exemplo a GoPro apontada para um objeto com a janela do webcam.py mostrando a classe prevista. -->
<p align="center">
  <img src="imagens/01_capa.png" alt="Capa do projeto: GoPro e classificação em tempo real" width="750">
</p>

---

## Sumário

1. [Contexto e motivação](#contexto-e-motivação)
2. [Objetivos](#objetivos)
3. [Visão geral da solução](#visão-geral-da-solução)
4. [Dataset](#dataset)
5. [Metodologia](#metodologia)
6. [Reconhecimento em tempo real](#reconhecimento-em-tempo-real)
7. [Resultados](#resultados)
8. [Limitações](#limitações)
9. [Protótipo físico e próximos passos](#protótipo-físico-e-próximos-passos)
10. [Estrutura do projeto](#estrutura-do-projeto)
11. [Requisitos e instalação](#requisitos-e-instalação)
12. [Como usar](#como-usar)
13. [Usando a GoPro como câmera](#usando-a-gopro-como-câmera)
14. [Solução de problemas](#solução-de-problemas)
15. [Créditos](#créditos)

---

## Contexto e motivação

A reciclagem depende de uma etapa simples e muitas vezes feita errado: **separar corretamente cada material**. Misturar papel, plástico, vidro e metal reduz o aproveitamento dos resíduos e encarece o processo. Uma lixeira capaz de identificar o material de um objeto e indicar (ou, no futuro, direcionar) o compartimento correto pode apoiar esse hábito e servir como ferramenta de educação ambiental.

Este trabalho é um **projeto de extensão** da disciplina de Inteligência Artificial: aplica técnicas de aprendizado profundo a um problema real e prepara a base para um protótipo físico.

<!-- COLOQUE AQUI (texto): se quiser, descreva o público ou a comunidade a que o projeto se destina e como ele será apresentado ou testado (por exemplo, escola, bairro, evento da instituição). -->

---

## Objetivos

**Objetivo geral:** desenvolver um sistema de visão computacional que classifique resíduos recicláveis pela câmera, como base para uma lixeira inteligente.

**Objetivos específicos:**

- Treinar uma CNN para reconhecer 6 classes de resíduos com o dataset *Garbage Classification*.
- Comparar duas abordagens: **transfer learning** (MobileNetV2) e **CNN construída do zero**.
- Avaliar os modelos em um conjunto de teste separado, com a divisão oficial do dataset.
- Executar a classificação em tempo real com uma câmera (GoPro).
- Preparar a integração com Arduino e servo motores para o protótipo físico.

---

## Visão geral da solução

<!-- COLOQUE AQUI (04_pipeline.png): um diagrama de fluxo simples (pode ser feito no draw.io, Canva ou PowerPoint): Câmera (GoPro) → Captura do quadro (OpenCV) → Redimensiona 224×224 e converte para RGB → CNN → Classe + confiança → (futuro) Arduino → Servo → Lixeira correta. -->
<p align="center">
  <img src="imagens/04_pipeline.png" alt="Diagrama do fluxo do sistema, da câmera à classe prevista" width="800">
</p>
<p align="center"><em>Figura 1. Fluxo do sistema: da captura da imagem à classe prevista (a parte do Arduino é a etapa futura).</em></p>

Foram implementadas **duas abordagens** para comparação:

| Script | Abordagem |
|---|---|
| `treinar.py` | **Transfer learning** com MobileNetV2 (CNN pré-treinada na ImageNet, ajustada para as 6 classes) |
| `treinarCNN.py` | **CNN construída do zero**, treinada apenas com as imagens do dataset |

---

## Dataset

- **Fonte:** [Garbage Classification (Kaggle)](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification/data), derivado do TrashNet.
- **Total:** 2.527 imagens em 6 classes: `cardboard` (papelão), `glass` (vidro), `metal`, `paper` (papel), `plastic` (plástico) e `trash` (lixo comum).
- **Divisão oficial** pelos arquivos `.txt` do Kaggle, sem imagens repetidas entre os conjuntos (cerca de 70% treino, 13% validação e 17% teste).

**Imagens por classe e conjunto:**

| Classe | Treino | Validação | Teste | Total |
|---|---:|---:|---:|---:|
| Papelão (`cardboard`) | 287 | 46 | 70 | 403 |
| Vidro (`glass`) | 354 | 65 | 82 | 501 |
| Metal (`metal`) | 286 | 56 | 68 | 410 |
| Papel (`paper`) | 403 | 83 | 108 | 594 |
| Plástico (`plastic`) | 347 | 61 | 74 | 482 |
| Lixo comum (`trash`) | 91 | 17 | 29 | 137 |
| **Total** | **1.768** | **328** | **431** | **2.527** |

A classe `trash` tem bem menos imagens que as outras (o dataset é **desbalanceado**), o que tende a torná-la a mais difícil de reconhecer.

<!-- COLOQUE AQUI (02_exemplos_dataset.png): uma grade com 1 ou 2 imagens de cada uma das 6 classes, tiradas do próprio dataset, com o nome da classe embaixo de cada uma. -->
<p align="center">
  <img src="imagens/02_exemplos_dataset.png" alt="Exemplos de imagens de cada classe do dataset" width="750">
</p>
<p align="center"><em>Figura 2. Exemplos de cada classe do dataset.</em></p>

<!-- COLOQUE AQUI (03_distribuicao_classes.png): um gráfico de barras com o número de imagens por classe (use os valores da coluna Total da tabela acima). -->
<p align="center">
  <img src="imagens/03_distribuicao_classes.png" alt="Gráfico de barras com a quantidade de imagens por classe" width="600">
</p>
<p align="center"><em>Figura 3. Distribuição de imagens por classe (dataset desbalanceado).</em></p>

> As imagens do dataset mostram objetos isolados sobre fundo claro e com boa iluminação. Isso influencia o desempenho com a câmera (veja [Limitações](#limitações)).

---

## Metodologia

### Pré-processamento e aumento de dados

- Todas as imagens são redimensionadas para **224 × 224** pixels.
- Aumento de dados aplicado somente durante o treino: espelhamento horizontal, rotação (até 10%) e zoom (até 10%).
- A normalização é feita dentro do próprio modelo, então a câmera envia a imagem com valores de 0 a 255.

### Modelo 1: transfer learning (`treinar.py`)

- Base **MobileNetV2** com pesos da ImageNet, seguida de `GlobalAveragePooling2D`, `Dropout(0.2)` e uma camada densa com 6 saídas (softmax).
- **Etapa 1:** treina só a camada final (8 épocas, taxa de aprendizado 1e-3).
- **Etapa 2:** ajuste fino das últimas 30 camadas da MobileNetV2 (5 épocas, taxa de aprendizado 1e-5).

> A base foi pré-treinada com imagens externas (ImageNet), mas o treino para as 6 classes e toda a avaliação usam **somente as imagens do dataset do projeto**. Os pesos da ImageNet são baixados como um arquivo, não como fotos.

### Modelo 2: CNN do zero (`treinarCNN.py`)

- 4 blocos `Conv2D` (32, 64, 128 e 128 filtros, kernel 3, ReLU) com `MaxPooling2D`.
- Depois: `Flatten`, `Dropout(0.4)`, `Dense(128)` e saída softmax com 6 classes.
- Até 40 épocas, com `EarlyStopping`: para quando a acurácia de validação não melhora por 6 épocas e mantém os melhores pesos.

<!-- COLOQUE AQUI (05_arquiteturas.png): um desenho simples das duas arquiteturas lado a lado, ou o print do resultado de modelo.summary() de cada uma. -->
<p align="center">
  <img src="imagens/05_arquiteturas.png" alt="Arquiteturas da MobileNetV2 com transfer learning e da CNN do zero" width="800">
</p>
<p align="center"><em>Figura 4. Arquiteturas dos dois modelos.</em></p>

### Avaliação

- A **validação** acompanha o treino a cada época.
- A **acurácia no conjunto de teste** (431 imagens que o modelo nunca viu) é calculada no final de cada treino e é o número principal de desempenho.
- Complementos recomendados: matriz de confusão e acurácia por classe, para ver onde o modelo erra.

---

## Reconhecimento em tempo real

O `webcam.py` repete, para cada quadro da câmera:

1. Captura o quadro com o OpenCV.
2. Converte de BGR para RGB e redimensiona para 224 × 224.
3. Passa a imagem pelo modelo e obtém a probabilidade de cada classe.
4. Mostra a classe de maior probabilidade e a confiança na tela. Abaixo de **60% de confiança**, mostra "incerto".

O classificador analisa o **quadro inteiro** (não recorta o objeto), então o objeto deve ocupar a parte central da imagem, sobre um fundo limpo.

<!-- COLOQUE AQUI (13_setup_gopro.png): uma foto do setup real: GoPro fixa, superfície clara e lisa, objeto no centro da imagem. -->
<p align="center">
  <img src="imagens/13_setup_gopro.png" alt="Setup de teste com a GoPro e fundo claro" width="600">
</p>
<p align="center"><em>Figura 5. Setup usado nos testes com a câmera.</em></p>

<!-- COLOQUE AQUI (11_webcam_exemplos.png): uma montagem com 6 prints da janela do webcam.py, um por classe (papelão, vidro, metal, papel, plástico e lixo comum), cada um mostrando o texto com a classe prevista e a confiança. -->
<p align="center">
  <img src="imagens/11_webcam_exemplos.png" alt="Exemplos da classificação em tempo real, um por classe" width="800">
</p>
<p align="center"><em>Figura 6. Classificação em tempo real pela GoPro: um exemplo para cada classe.</em></p>

---

## Resultados

### Comparação dos modelos

Preencha com os valores impressos no final de cada treino (acurácia no **conjunto de teste**):

| Modelo | Acurácia no teste | Acurácia na validação | Observações |
|---|---|---|---|
| MobileNetV2 (transfer learning) | _preencher_ | _preencher_ | |
| CNN do zero | _preencher_ | _preencher_ | |

<!-- COLOQUE AQUI (10_comparacao_modelos.png), opcional: um gráfico de barras com a acurácia no teste dos dois modelos. -->
<p align="center">
  <img src="imagens/10_comparacao_modelos.png" alt="Gráfico de barras comparando a acurácia dos dois modelos no teste" width="500">
</p>
<p align="center"><em>Figura 7. Acurácia no conjunto de teste dos dois modelos.</em></p>

### Curvas de treinamento

<!-- COLOQUE AQUI (06_curvas_mobilenet.png): dois gráficos lado a lado (acurácia e perda) por época, com as linhas de treino e de validação, do treino com MobileNetV2. -->
<p align="center">
  <img src="imagens/06_curvas_mobilenet.png" alt="Curvas de acurácia e perda da MobileNetV2" width="700">
</p>
<p align="center"><em>Figura 8. Curvas de acurácia e perda por época: MobileNetV2.</em></p>

<!-- COLOQUE AQUI (07_curvas_cnn.png): os mesmos dois gráficos (acurácia e perda por época, treino e validação), do treino da CNN do zero. -->
<p align="center">
  <img src="imagens/07_curvas_cnn.png" alt="Curvas de acurácia e perda da CNN do zero" width="700">
</p>
<p align="center"><em>Figura 9. Curvas de acurácia e perda por época: CNN do zero.</em></p>

### Matrizes de confusão (conjunto de teste)

<!-- COLOQUE AQUI (08_matriz_confusao_mobilenet.png): a matriz de confusão da MobileNetV2 calculada no conjunto de teste (431 imagens), com os nomes das 6 classes nos dois eixos. -->
<p align="center">
  <img src="imagens/08_matriz_confusao_mobilenet.png" alt="Matriz de confusão da MobileNetV2 no conjunto de teste" width="550">
</p>
<p align="center"><em>Figura 10. Matriz de confusão: MobileNetV2.</em></p>

<!-- COLOQUE AQUI (09_matriz_confusao_cnn.png): a matriz de confusão da CNN do zero no mesmo conjunto de teste. -->
<p align="center">
  <img src="imagens/09_matriz_confusao_cnn.png" alt="Matriz de confusão da CNN do zero no conjunto de teste" width="550">
</p>
<p align="center"><em>Figura 11. Matriz de confusão: CNN do zero.</em></p>

### Exemplos de erros

<!-- COLOQUE AQUI (12_exemplos_de_erros.png): 3 a 6 imagens em que o modelo errou (por exemplo, vidro previsto como plástico), cada uma com a classe real e a classe prevista na legenda. Podem ser imagens do teste ou prints da câmera. -->
<p align="center">
  <img src="imagens/12_exemplos_de_erros.png" alt="Exemplos de imagens classificadas incorretamente" width="750">
</p>
<p align="center"><em>Figura 12. Exemplos de classificações incorretas, com a classe real e a prevista.</em></p>

### Discussão

<!-- COLOQUE AQUI (texto): escreva 1 ou 2 parágrafos interpretando os resultados: qual modelo teve melhor acurácia no teste, quais classes foram mais confundidas, como foi o desempenho na câmera em comparação com o teste e o que explica as diferenças. -->

---

## Limitações

- **Diferença entre o dataset e a câmera:** as fotos de treino têm fundo claro e iluminação controlada. Na câmera, a acurácia tende a cair com fundos diferentes, sombras ou vários objetos na imagem. Para melhorar, use uma superfície lisa e clara, boa luz e **um objeto por vez**.
- **Dataset desbalanceado:** a classe `trash` tem poucas imagens e tende a ser a mais difícil.
- **Confusões comuns:** vidro e plástico podem ser confundidos.
- **Sem detecção de objetos:** o modelo classifica o quadro inteiro, sem localizar ou recortar o objeto.
- **Apenas 6 categorias:** objetos fora dessas classes serão forçados a uma delas (a confiança baixa ajuda a sinalizar "incerto").
- Para um desempenho melhor na demonstração, vale acrescentar ao treino fotos dos objetos reais que serão usados, tiradas no mesmo ambiente.

---

## Protótipo físico e próximos passos

Etapas planejadas para a lixeira inteligente:

1. Enviar a classe prevista pelo computador ao **Arduino** por comunicação serial.
2. Acionar **servo motores** para abrir ou direcionar o resíduo ao compartimento correto.
3. Imprimir em **3D** as lixeiras e definir o mapeamento entre as 6 classes e os compartimentos.
4. Reavaliar o modelo com fotos próprias, tiradas no ambiente da demonstração.

<!-- COLOQUE AQUI (14_prototipo_3d.png): quando existir, uma foto da lixeira impressa em 3D ou o modelo 3D (print do software de modelagem). Enquanto não existir, pode usar um esboço. -->
<p align="center">
  <img src="imagens/14_prototipo_3d.png" alt="Protótipo da lixeira impressa em 3D" width="600">
</p>
<p align="center"><em>Figura 13. Protótipo físico da lixeira (em desenvolvimento).</em></p>

<!-- COLOQUE AQUI (15_esquema_arduino.png): quando existir, o esquema de ligação do Arduino com os servo motores (pode ser feito no Tinkercad ou no Fritzing). -->
<p align="center">
  <img src="imagens/15_esquema_arduino.png" alt="Esquema de ligação do Arduino com os servo motores" width="600">
</p>
<p align="center"><em>Figura 14. Esquema de ligação do Arduino e dos servo motores (em desenvolvimento).</em></p>

---

## Estrutura do projeto

```
.
├── Garbage classification/                 # imagens (subpastas: cardboard, glass, metal, paper, plastic, trash)
├── imagens/                                # figuras usadas neste README
├── one-indexed-files-notrash_train.txt     # lista oficial de treino
├── one-indexed-files-notrash_val.txt       # lista oficial de validação
├── one-indexed-files-notrash_test.txt      # lista oficial de teste
├── dados.py                                # carrega as imagens seguindo as listas
├── treinar.py                              # treino com MobileNetV2 (transfer learning)
├── treinarCNN.py                           # treino com CNN própria (do zero)
├── testar_camera.py                        # descobre qual número é a câmera (ex.: GoPro)
├── webcam.py                               # classificação em tempo real pela câmera
├── classes.txt                             # gerado automaticamente (nomes das classes)
└── modelo.keras                            # gerado pelo treino (modelo treinado)
```

Os arquivos `one-indexed-files.txt` e `zero-indexed-files.txt` do Kaggle (lista completa com os rótulos) não são usados pelos scripts.

---

## Requisitos e instalação

- **Python 3.12** e **TensorFlow 2.21** (versões em que o fluxo de treino foi testado)
- OpenCV e NumPy

```bash
pip install tensorflow opencv-python numpy
```

Na primeira execução do `treinar.py`, o TensorFlow baixa os pesos pré-treinados da MobileNetV2, então é necessário ter **internet** nessa vez.

---

## Como usar

Execute os comandos **dentro da pasta do projeto** (a mesma que contém `dados.py`).

### 1. Preparar os arquivos

- Baixe o dataset no Kaggle e deixe as 6 pastas de classes dentro de `Garbage classification/`.
- Coloque os três arquivos `.txt` de divisão (`..._train.txt`, `..._val.txt`, `..._test.txt`) na mesma pasta dos scripts.
- Se a sua estrutura for diferente, ajuste `DATASET_DIR` no início do `dados.py`.

### 2. Treinar o modelo

```bash
python treinar.py        # MobileNetV2 (transfer learning)
# ou
python treinarCNN.py     # CNN do zero
```

Ao final, o script mostra a **acurácia no conjunto de teste** e salva `modelo.keras` e `classes.txt`.

> Os dois scripts salvam com o mesmo nome. Para comparar os modelos, renomeie o arquivo depois do primeiro treino (por exemplo, `modelo_mobilenet.keras`) e volte para `modelo.keras` o que quiser usar na câmera.

### 3. Descobrir o número da câmera

```bash
python testar_camera.py
```

O script testa as câmeras de 0 a 5 em dois modos (padrão e DirectShow, do Windows) e informa quais abrem e se a imagem está preta. Aperte uma tecla para passar de uma câmera à próxima.

### 4. Classificar pela câmera

No `webcam.py`, ajuste as duas variáveis no início:

```python
CAMERA = 0              # número encontrado no testar_camera.py
MODO = cv2.CAP_ANY      # use cv2.CAP_DSHOW se a imagem ficar preta (Windows)
```

Depois execute:

```bash
python webcam.py
```

A janela mostra a classe prevista e a confiança. Aperte **`q`** para sair.

---

## Usando a GoPro como câmera

- **HERO8 Black ou mais nova:** conecte por USB-C e abra o **GoPro Webcam Desktop Utility** no computador. A GoPro passa a aparecer como uma webcam. Confira o modelo na lista de compatibilidade no site da GoPro.
- **Modelos mais antigos (HERO4 a 7 Black):** é necessário um adaptador de captura HDMI-USB.
- Mantenha o utilitário aberto e feche outros programas que usem a câmera (Zoom, OBS, navegador).
- Nas configurações do utilitário, use a lente **Linear**: a lente padrão (Wide) distorce a imagem e afasta o formato dos objetos do que o modelo viu no treino.

---

## Solução de problemas

| Problema | O que fazer |
|---|---|
| `Nao encontrei 'Garbage classification/cardboard'...` | Confira se a pasta com as 6 classes está ao lado dos scripts ou ajuste `DATASET_DIR` no `dados.py`. |
| `FileNotFoundError` com um arquivo `.txt` | Coloque os arquivos de divisão (`train`, `val`, `test`) na mesma pasta dos scripts. |
| A câmera não abre | Rode `testar_camera.py` e use o número que mostrar `OK` na variável `CAMERA`. |
| Imagem preta | Troque `MODO` para `cv2.CAP_DSHOW` (Windows), abra o GoPro Webcam Utility e feche outros apps que usem a câmera. |
| Caracteres estranhos no terminal (`�`) | É só a codificação do console do Windows; não afeta o funcionamento. |
| Treino muito lento | Use o Google Colab ou um notebook do Kaggle com GPU e depois baixe `modelo.keras` e `classes.txt`. |
| Download dos pesos falha no `treinar.py` | A primeira execução precisa de internet. |

---

## Créditos

- **Autor:** Matheus Oliveira Silva, estudante de Ciência da Computação no IFCE ([GitHub](https://github.com/Madrooo))
- **Dataset:** [Garbage Classification no Kaggle](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification/data), derivado do TrashNet (Gary Thung e Mindy Yang).
- **MobileNetV2:** pesos pré-treinados na ImageNet, distribuídos com o Keras/TensorFlow.
- Projeto acadêmico de extensão da disciplina de Inteligência Artificial.
