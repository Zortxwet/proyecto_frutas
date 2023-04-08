import numpy as np
from keras.utils import load_img, img_to_array
from keras.models import load_model

longitud, altura = 255, 255
modelo = './modelo/modelo.h5'
pesos_modelo = './modelo/pesos.h5'
model = load_model(modelo)
model.load_weights(pesos_modelo)

def prediccion(file):
    x = load_img(file, target_size=(longitud, altura))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    array = model.predict(x)
    result = array[0]
    answer = np.argmax(result)
    if answer == 0:
        print("pred: Perro")
    elif answer == 1:
        print("pred: Gato")
    elif answer == 2:
        print("pred: Gorila")

    return answer

prediccion('./putamadre')