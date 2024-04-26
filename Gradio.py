import gradio as gr
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import LabelEncoder


@keras.utils.register_keras_serializable()
class NGL(tf.keras.losses.Loss):
    def __init__(
    	self,
    	scaling=False,
    	name="ngl_loss",
        reduction=tf.keras.losses.Reduction.AUTO,):
        super().__init__(name=name)
        self.name = name
        self.scaling = scaling

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        if self.scaling == True:
	 	        y_pred = tf.math.sigmoid(y_pred)
        part_1 = tf.math.exp(2.4092 - y_pred - y_pred*y_true)
        part_2 = tf.math.cos(tf.math.cos(tf.math.sin(y_pred)))
        elements = part_1 - part_2
        loss = tf.reduce_mean(elements)
        return loss


custom_objects = {"NGL": NGL}
with keras.utils.custom_object_scope(custom_objects):
    reconstructed_model = keras.models.load_model("model.keras")



data = pd.read_csv("input/train.csv")
filtered_data = data['Id']
# Создание объекта LabelEncoder
label_encoder = LabelEncoder()

# Преобразование строковых идентификаторов классов в числовые метки
classes_id = label_encoder.fit_transform(data["Id"])

# Создание словаря для соответствия числовых меток и строковых идентификаторов
class_dict = {class_id: class_name for class_id, class_name in zip(classes_id, data["Id"])}

# Функция для идентификации кита по загруженному изображению
def identify_whale(img):
    x = image.img_to_array(img)
    x = tf.image.resize(x, [100, 100])
    x = x / 225

    x = np.expand_dims(x, axis=0)

    prediction = reconstructed_model.predict(x)
    whale_id = np.argmax(prediction)

    class_name = class_dict[whale_id]

    return class_name

# Создание интерфейса Gradio
iface = gr.Interface(
    fn=identify_whale,
    inputs="image",
    outputs="text",
    title="Humpback Whale Identification",
    description="Upload an image of a whale to identify its ID."
)

# Запуск сервиса Gradio
iface.launch()