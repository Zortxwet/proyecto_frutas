import cv2
import os

ent_imagen = "C:/Users/Jose/Downloads/JOSE/jose/Electronica/VIII Semestre/Seminario modalidad de grado/proyecto_frutas/green_bananas"

ent_imagen_red = "C:/Users/Jose/Downloads/JOSE/jose/Electronica/VIII Semestre/Seminario modalidad de grado/proyecto_frutas/green_bananas_pru"

if not os.path.exists(ent_imagen_red):
    os.makedirs(ent_imagen_red) 

nombre_imagenes = os.listdir(ent_imagen)

cont = 0
for nombre_imagen in nombre_imagenes:
    direccion_imagen = ent_imagen + "/" + nombre_imagen
    imagen = cv2.imread(direccion_imagen)

    imagen = cv2.resize(imagen, (255, 255), interpolation=cv2.INTER_CUBIC)
    
    cv2.imwrite(ent_imagen_red + "/imagen_red" + str(cont) + ".jpg", imagen)
    cont += 1