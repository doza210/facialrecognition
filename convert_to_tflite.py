import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import save_model

# Load MobileNetV2 from Keras Applications
model = MobileNetV2(weights='imagenet', input_shape=(224, 224, 3))

# Save the Keras model
save_model(model, 'mobilenetv2.h5')

# Convert the Keras model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()

# Save the TensorFlow Lite model to a file
with open("mobilenetv2.tflite", "wb") as f:
    f.write(tflite_model)
