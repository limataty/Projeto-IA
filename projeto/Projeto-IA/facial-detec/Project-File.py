import cv2
import matplotlib.pyplot as plt

# caminho da imagem
imagePath = 'input_image.jpg'

# ler a imagem
img = cv2.imread(imagePath)

# printar a img
img.shape
# valores da img - width, height, rgb
(4000, 2667, 3)

# passar a img pra escala de cinza
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_image.shape
(4000, 2667)

# inicializa
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
face = face_classifier.detectMultiScale(
    gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
)

# contorno em volta do rosto
for (x, y, w, h) in face:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

# formatar BGR para RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# pacote Matplotlib pra mostrar a img
plt.figure(figsize=(20,10))
plt.imshow(img_rgb)
plt.axis('off')
plt.show()







