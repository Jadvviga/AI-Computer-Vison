import tensorflow as tf
import os


MODELS_PATH = "models"
model_filename = 'VGG16_Updated_model_224_224.h5'
model = tf.keras.models.load_model(filepath=os.path.join(MODELS_PATH, model_filename))

model.summary()
