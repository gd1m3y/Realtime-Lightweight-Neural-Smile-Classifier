from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from keras.preprocessing.image import img_to_array

from Lenet import Lenet

from keras.utils import np_utils

import matplotlib.pyplot as plt

import numpy as np

import argparse
from imutils import paths

import cv2
import imutils
import os

# ap = argparse.ArgumentParser()
# ap.add_argument('-d', '--dataset', required=True, help='path to the data set')
#
# ap.add_argument('-m', '--model', required=True, help='path to the model')
#
# args = vars(ap.parse_args())

data = []

labels = []

for imgpath in sorted(list(paths.list_images('D:\data\dataset\SMILEsmileD\SMILEs'))):
    image = cv2.imread(imgpath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = imutils.resize(image, width=28)
    image = img_to_array(image)
    data.append(image)

    label = imgpath.split(os.path.sep)[-2]
    # print(label)
    label = 'smiling' if label == 'positive' else 'negative'

    labels.append(label)

data = np.array(data) / 255.0

labels = np.array(labels)

le = LabelEncoder().fit(labels)

labels = np_utils.to_categorical(le.transform(labels), 2)

class_total = labels.sum(axis=0)

class_weight = class_total.max() / class_total

train_x, test_x, train_y, test_y = train_test_split(data, labels, stratify=labels, random_state=42, test_size=0.20)

print('Compiling the model')

model = Lenet.build(width=28, height=28, depth=1, classes=2)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

h = model.fit(train_x, train_y, validation_data=(test_x, test_y), class_weight=class_weight, batch_size=64, epochs=15,
              verbose=1)

print('Evaluating the model')

predictions = model.predict(test_x, batch_size=64)

print(classification_report(test_y.argmax(axis=1), predictions.argmax(axis=1), target_names=le.classes_))

model.save('my_model.hdf5')

plt.style.use('ggplot')

plt.figure()

plt.plot(np.arange(0, 15), h.history['loss'], label='train_loss')
plt.plot(np.arange(0, 15), h.history['val_loss'], label='val_loss')

plt.plot(np.arange(0,15), h.history['accuracy'],label='accuracy')

plt.plot(np.arange(0,15),h.history['val_accuracy'],label = 'val_accuracy')

plt.xlabel('Epoch')

plt.ylabel('loss')

plt.legend()
plt.show()
