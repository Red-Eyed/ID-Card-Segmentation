#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Vadym Stupakov"
__email__ = "vadim.stupakov@gmail.com"

import tensorflow.compat.v1 as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import onnxmltools
from model.iou_loss import IoU

if __name__ == '__main__':
    model = load_model('unet_model_whole_100epochs.h5', compile=False)
    model.compile(optimizer=Adam(1e-4), loss=IoU, metrics=['binary_accuracy'])
    tf.keras.backend.set_learning_phase(0)

    converted_model = onnxmltools.convert_keras(model)
    onnxmltools.save_model(converted_model, "unet_model_whole_100epochs.onnx")
