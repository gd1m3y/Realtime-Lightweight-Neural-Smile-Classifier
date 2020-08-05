import cv2

from keras.models import load_model

import imutils

from keras.preprocessing.image import img_to_array

import numpy as np

detector = cv2.CascadeClassifier(
    'C:/Users/naray/PycharmProjects/opencv_projects/venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')


model = load_model('my_model.hdf5')

camera = cv2.VideoCapture(0)

while True:
    grabbed, frame = camera.read()

    frame = imutils.resize(frame, width=300)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameClone = frame.copy()
    rects = detector.detectMultiScale(gray, scaleFactor=1.1,

                                      minNeighbors=5, minSize=(30, 30),

                                      flags=cv2.CASCADE_SCALE_IMAGE)
    for (fx, fy, fw, fh) in rects:
        roi = gray[fy:fy + fh, fx:fx + fw]
        roi = cv2.resize(roi, (28,28))
        roi = roi.astype('float') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        (notsmile, smile) = model.predict(roi)[0]
        label = 'smile' if smile > notsmile else 'no smile'
        print(smile)
        cv2.putText(frameClone, label, (fx, fy - 10), cv2.FONT_HERSHEY_COMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frameClone, (fx, fy), (fw + fx, fh + fy), (0, 0, 255), 2)

        cv2.imshow('face', frameClone)

        if cv2.waitKey(1) and 0xFF == ord('q'):
            break
camera.release()
cv2.destroyAllWindows()
