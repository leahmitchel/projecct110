import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.model.load_model('keras_model.h5')

video = cv2.VideoCapture(0)

while True:
    check, frame = video.read()

    img = cv2.resize(frame, (224,224))

    test_image = np.array(img, dtype=np.float32)
    test__image = np.expand_dims(test_image, axis=0)

    normalised_image = test_image/255.0

    prediction = model.predict(normalised_image)
    print("Prediction : ", prediction)
    cv2.imshow("Result", frame)
    cv2.waitKey()

cv2.destroyAllWindows()