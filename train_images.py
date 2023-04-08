import sys
import numpy as np
import os
from PIL import Image
import tensorflow as tf
from tensorflow.python.keras import models, layers
from tensorflow.python.keras import backend as K
from keras.utils import img_to_array, load_img

K.clear_session()

data_entrenamiento = './entrenamiento'
data_validacion = './validacion'

datos_entrenamiento = tf.keras.utils.image_dataset_from_directory(data_entrenamiento, validation_split=0.2, subset="training", seed=123, batch_size= 100)
datos_validacion = tf.keras.utils.image_dataset_from_directory(data_validacion, validation_split=0.2, subset="validation", seed=123, batch_size= 100)

class_names = datos_entrenamiento.class_names

#parametros
epocas = 3 #numero de veces que se va a iterar
altura, longitud = 255, 255
batch_size = 100 #numero de imagenes a procesar
pasos = 1000 #numero de veces que se va a procesar la informacion en cada epoca
pasos_validacion = 200 
filtrosConv1 = 32 #profundidad de 32
filtrosConv2 = 64
tamano_filtro1 = (3,3)
tamano_pool = (2,2)

#pre procesamiento de imagenes

#crear la red CNN

# cnn = Sequential()

# cnn.add(Convolution2D(filtrosConv1, tamano_filtro1, padding='same', input_shape = (altura, longitud), activation='relu'))

# cnn.add(MaxPooling2D(pool_size=tamano_pool))

# cnn.add(Convolution2D(filtrosConv2, tamano_filtro2, padding='same', activation='relu'))

# cnn.add(MaxPooling2D(pool_size=tamano_pool))

# cnn.add(Flatten())
# cnn.add(Dense(256,activation='relu'))
# cnn.add(Dropout(0.5))
# cnn.add(Dense(clases, activation='softmax'))

# cnn.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=lr), metrics=['accuracy'])

# cnn.fit()

model = models.Sequential()
model.add(layers.Conv2D(filtrosConv1, tamano_filtro1, activation='relu', input_shape=(altura, longitud, 3)))
model.add(layers.MaxPooling2D((tamano_pool)))
model.add(layers.Conv2D(filtrosConv2, tamano_filtro1, activation='relu'))
model.add(layers.MaxPooling2D((tamano_pool)))
model.add(layers.Conv2D(filtrosConv2, tamano_filtro1, activation='relu'))

model.add(layers.Flatten())
#model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
#model.add(layers.Dense(10))
model.add(layers.Dense(3, activation='softmax'))

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])


history = model.fit(datos_entrenamiento, validation_data=datos_validacion, epochs=epocas)

test_loss, test_acc = model.evaluate(datos_validacion, verbose=2)
print(test_acc)

modelo_entrenado = './modelo'

if not os.path.exists(modelo_entrenado):
    os.mkdir(modelo_entrenado)

model.save('./modelo/modelo.h5')
model.save_weights('./modelo/pesos.h5')

img = Image.open('./green_bananas_pru/imagen_red487.jpg')
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
