import tensorflow as tf
import tf2onnx
import onnx
from tensorflow import keras
from keras.models import load_model

model = load_model('model.h5')

tf.saved_model.save(model, "tmp_model")

