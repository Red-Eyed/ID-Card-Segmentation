import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from model.iou_loss import IoU

if __name__ == '__main__':
    model = load_model('unet_model_whole_100epochs.h5', compile=False)
    model.compile(optimizer=Adam(1e-4), loss=IoU, metrics=['binary_accuracy'])

    image_name = './images/test.jpg'

    img = cv2.imread(image_name)
    h, w = img.shape[:2]
    img = cv2.resize(img, (256, 256))
    img = img / 255.0
    predict = model.predict(img.reshape(1, 256, 256, 3))

    output = predict[0]
    output = cv2.resize(output, (w, h))
    plt.imsave('output.jpg', output, cmap='gray')
