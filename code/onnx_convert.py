import tensorflow as tf
import tf2onnx
import onnx
from tensorflow import keras
from keras.models import load_model

model = load_model('code/models/model_28.h5')

tf.saved_model.save(model, "tmp_model2")

