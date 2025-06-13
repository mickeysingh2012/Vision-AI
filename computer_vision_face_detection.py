# -*- coding: utf-8 -*-
"""Computer VisionFace detection

# Computer Vision - Face detection

## OpenCV

### Loading the image
"""

import cv2

from google.colab import drive
drive.mount('/content/drive')

image = cv2.imread('/content/drive/MyDrive/Cursos - recursos/Computer Vision/Images/people1.jpg')

image.shape

#cv2.imshow(image)
from google.colab.patches import cv2_imshow
cv2_imshow(image)

image = cv2.resize(image, (800, 600))
image.shape

cv2_imshow(image)

600 * 800 * 3, 600 * 800, 1440000 - 480000

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2_imshow(image_gray)

image.shape

image_gray.shape

"""### Detecting faces"""

face_detector = cv2.CascadeClassifier('/content/drive/MyDrive/Cursos - recursos/Computer Vision/Cascades/haarcascade_frontalface_default.xml')

detections = face_detector.detectMultiScale(image_gray)

detections

len(detections)

for (x, y, w, h) in detections:
  #print(x, y, w, h)
  cv2.rectangle(image, (x, y), (x + w, y + h), (0,255,255), 5)
cv2_imshow(image)

"""### Haarcascade parameters"""

image = cv2.imread('/content/drive/MyDrive/Cursos - recursos/Computer Vision/Images/people1.jpg')
image = cv2.resize(image, (800, 600))
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
detections = face_detector.detectMultiScale(image_gray, scaleFactor = 1.09)
for (x, y, w, h) in detections:
  cv2.rectangle(image, (x, y), (x + w, y + h), (0,255,0), 5)
cv2_imshow(image)

image = cv2.imread('/content/drive/MyDrive/Cursos - recursos/Computer Vision/Images/people2.jpg')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
detections = face_detector.detectMultiScale(image_gray, scaleFactor=1.2, minNeighbors=7,
                                            minSize=(20,20), maxSize=(100,100))
for (x, y, w, h) in detections:
  print(w, h)
  cv2.rectangle(image, (x, y), (x + w, y + h), (0,255,0), 2)
cv2_imshow(image)

"""### Eye detection"""

eye_detector = cv2.CascadeClassifier('/content/drive/MyDrive/Cursos - recursos/Computer Vision/Cascades/haarcascade_eye.xml')

image = cv2.imread('/content/drive/MyDrive/Cursos - recursos/Computer Vision/Images/people1.jpg')
#image = cv2.resize(image, (800, 600))
print(image.shape)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

face_detections = face_detector.detectMultiScale(image_gray, scaleFactor = 1.3, minSize = (30,30))
for (x, y, w, h) in face_detections:
  cv2.rectangle(image, (x, y), (x + w, y + h), (0,255,0), 2)

eye_detections = eye_detector.detectMultiScale(image_gray, scaleFactor = 1.1, minNeighbors=10, maxSize=(70,70))
for (x, y, w, h) in eye_detections:
  print(w, h)
  cv2.rectangle(image, (x, y), (x + w, y + h), (0,0,255), 2)

cv2_imshow(image)

"""### Other objects

#### Cars
"""

car_detector = cv2.CascadeClassifier('/content/drive/MyDrive/Cursos - recursos/Computer Vision/Cascades/cars.xml')
image = cv2.imread('/content/drive/MyDrive/Cursos - recursos/Computer Vision/Images/car.jpg')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
detections = car_detector.detectMultiScale(image_gray, scaleFactor = 1.03, minNeighbors=5)
for (x, y, w, h) in detections:
  cv2.rectangle(image, (x, y), (x + w, y + h), (0,255,0), 2)
cv2_imshow(image)

"""#### Clocks"""

clock_detector = cv2.CascadeClassifier('/content/drive/MyDrive/Cursos - recursos/Computer Vision/Cascades/clocks.xml')
image = cv2.imread('/content/drive/MyDrive/Cursos - recursos/Computer Vision/Images/clock.jpg')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
detections = clock_detector.detectMultiScale(image_gray, scaleFactor = 1.03, minNeighbors=1)
for (x, y, w, h) in detections:
  cv2.rectangle(image, (x, y), (x + w, y + h), (0,255,0), 2)
cv2_imshow(image)

"""#### Full body"""

full_detector = cv2.CascadeClassifier('/content/drive/MyDrive/Cursos - recursos/Computer Vision/Cascades/fullbody.xml')
image = cv2.imread('/content/drive/MyDrive/Cursos - recursos/Computer Vision/Images/people3.jpg')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
detections = full_detector.detectMultiScale(image_gray, scaleFactor = 1.05, minNeighbors=5,
                                              minSize = (50,50))
for (x, y, w, h) in detections:
  cv2.rectangle(image, (x, y), (x + w, y + h), (0,255,0), 2)
cv2_imshow(image)

"""## Dlib"""

import dlib

"""### Detecting faces with HOG"""

image = cv2.imread('/content/drive/MyDrive/Cursos - recursos/Computer Vision/Images/people2.jpg')
cv2_imshow(image)

face_detector_hog = dlib.get_frontal_face_detector()

detections = face_detector_hog(image, 1)

detections, len(detections)

for face in detections:
  #print(face)
  #print(face.left())
  #print(face.top())
  #print(face.right())
  #print(face.bottom())
  l, t, r, b = face.left(), face.top(), face.right(), face.bottom()
  cv2.rectangle(image, (l, t), (r, b), (0, 255, 255), 2)
cv2_imshow(image)

"""### Detecting faces with CNN (Convolutional Neural Networks)"""

image = cv2.imread('/content/drive/MyDrive/Cursos - recursos/Computer Vision/Images/people2.jpg')
cnn_detector = dlib.cnn_face_detection_model_v1('/content/drive/MyDrive/Cursos - recursos/Computer Vision/Weights/mmod_human_face_detector.dat')

detections = cnn_detector(image, 1)
for face in detections:
  l, t, r, b, c = face.rect.left(), face.rect.top(), face.rect.right(), face.rect.bottom(), face.confidence
  print(c)
  cv2.rectangle(image, (l, t), (r, b), (255, 255, 0), 2)
cv2_imshow(image)

"""### Haarcascade x HOG x CNN

#### Haarcascade
"""

image.shape

image = cv2.imread('/content/drive/MyDrive/Cursos - recursos/Computer Vision/Images/people3.jpg')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
haarcascade_detector = cv2.CascadeClassifier('/content/drive/MyDrive/Cursos - recursos/Computer Vision/Cascades/haarcascade_frontalface_default.xml')
detections = haarcascade_detector.detectMultiScale(image_gray, scaleFactor = 1.001, minNeighbors=5, minSize = (5,5))
for (x, y, w, h) in detections:
  cv2.rectangle(image, (x, y), (x + w, y + h), (0,255,0), 2)
cv2_imshow(image)

"""#### HOG"""

image = cv2.imread('/content/drive/MyDrive/Cursos - recursos/Computer Vision/Images/people3.jpg')
face_detector_hog = dlib.get_frontal_face_detector()
detections = face_detector_hog(image, 4)
for face in detections:
    l, t, r, b = (face.left(), face.top(), face.right(), face.bottom())
    cv2.rectangle(image, (l, t), (r, b), (0, 255, 255), 2)
cv2_imshow(image)

"""#### CNN"""

image = cv2.imread('/content/drive/MyDrive/Cursos - recursos/Computer Vision/Images/people3.jpg')
cnn_detector = dlib.cnn_face_detection_model_v1('/content/drive/MyDrive/Cursos - recursos/Computer Vision/Weights/mmod_human_face_detector.dat')
detections = cnn_detector(image, 4)
for face in detections:
  l, t, r, b, c = face.rect.left(), face.rect.top(), face.rect.right(), face.rect.bottom(), face.confidence
  print(c)
  cv2.rectangle(image, (l, t), (r, b), (255, 255, 0), 2)
cv2_imshow(image)