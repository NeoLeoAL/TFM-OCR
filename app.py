from tkinter import *
from tkinter import messagebox
from tkinterdnd2 import *
from PIL import Image, ImageTk
from utils import browseFiles
from googletrans import *
import pyttsx3
from keras.models import load_model

import cv2
import imutils
from imutils.contours import sort_contours
import numpy as np

import pytesseract
from hermetrics.levenshtein import Levenshtein

class Application():
    __WINDOW_WIDTH = 1024
    __WINDOW_HEIGHT = 768
    
    __DEFAULT_IMAGE = 'background.png'
    __IMAGE_WIDTH = 700
    __IMAGE_HEIGHT = 300
    
    __DEFAULT_LANG = 'es'
    
    __root = Tk()
    __translatedText = ''
    __imagePath = ''

    def __init__(self):
        self.__root.title("Optical character recognition")
        #__root.iconbitmap('./assets/pythontutorial.ico')
        #__root.config(bg='#fcb103') # BACKGROUND COLOR

        center_x = int(self.__root.winfo_screenwidth() / 2 - self.__WINDOW_WIDTH / 2)
        center_y = int(self.__root.winfo_screenheight() / 2 - self.__WINDOW_HEIGHT / 2)

        self.__root.geometry(f'{self.__WINDOW_WIDTH}x{self.__WINDOW_HEIGHT}+{center_x}+{center_y}')
        self.__root.resizable(False, False)

        self.__createImageField(self.__DEFAULT_IMAGE)
        self.__createButton(20, "TRANSLATE", self.__translateText, ((self.__WINDOW_WIDTH / 2) - 80), (self.__IMAGE_HEIGHT + 80))
        self.__createButton(5, "VOICE", self.__reproduceTranslation, (self.__WINDOW_WIDTH - 100), (self.__IMAGE_HEIGHT + 120))
        self.__createTextarea(self.__translatedText)

        self.__root.mainloop()

    def __createImageField(self, imagePath):
        self.__imagePath = imagePath
        image = Image.open(imagePath)
        resizedImage = image.resize((self.__IMAGE_WIDTH, self.__IMAGE_HEIGHT))

        image = ImageTk.PhotoImage(resizedImage)

        labelImage = Label(image = image)
        labelImage.image = image

        labelImage.drop_target_register(DND_FILES)
        labelImage.dnd_bind('<<Drop>>', self.__showImage)

        center_x = (self.__WINDOW_WIDTH / 2) - (resizedImage.width / 2)
        labelImage.place(x = center_x, y = 50)
        
        labelImage.bind("<Button-1>", self.__labelClick)

    def __showImage(self, event):    
        if event.data.endswith('.png') or event.data.endswith('.jpg'):
            self.__createImageField(event.data)
            
    def __labelClick(self, event):
        imagePath = browseFiles()
        
        if imagePath != '':
            self.__createImageField(imagePath)
            
    def __createButton(self, width, text, command, place_x, place_y):
        button = Button(self.__root,  width = width, text = text, command = command)
        button.place(x = place_x, y = place_y)

    def __translateText(self):
        model = load_model('ocrModel.h5')
        imgText, predictText = self.__convertImage(model)

        translator = Translator() 
        detectedLang = translator.detect(imgText)
        translated = translator.translate(imgText, src = detectedLang.lang, dest = self.__DEFAULT_LANG)
        self.__translatedText = translated.text
        finalText = f'Library predit Text ({detectedLang.lang}): {imgText}Model predict Text: {predictText} \nTranslated Text ({self.__DEFAULT_LANG}): {self.__translatedText} '
        
        self.__createTextarea(finalText)

    def __convertImage(self, model):
        image = cv2.imread(self.__imagePath)

        imgText = pytesseract.image_to_string(image, config = r'--oem 3 --psm 6')

        imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        imgBlurred = cv2.GaussianBlur(imgGray, (5, 5), 0)

        # Detecta los bordes, encuentra los contornos y los ordena de izquierda a derecha
        imgEdged = cv2.Canny(imgBlurred, 30, 150)

        contours = cv2.findContours(imgEdged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sort_contours(contours, method="left-to-right")[0]

        chars = []

        for contour in contours:
            # Devuelve la posición y medida de la caja contorno de cada caracter
            (x, y, width, height) = cv2.boundingRect(contour)

            if (width >= 5 and width <= 150) and (height >= 15 and height <= 120):
                """
                Se extrae el carácter y se umbraliza para que el carácter aparezca como blanco sobre un fondo negro
                y se toma el ancho y el alto de la imagen umbralizada
                """
                roi = imgGray[y:y + height, x:x + width]
                thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                (tH, tW) = thresh.shape

                """
                Se vuelve a tomar las dimensiones de la imagen (ahora que ha sido redimensionada) 
                y se determina cuánto tenemos que rellenar de ancho y alto para que la imagen sea de 32x32
                """
                dX = int(max(0, 28 - tW) / 2.0)
                dY = int(max(0, 28 - tH) / 2.0)

                # Se rellena la imagen y  se fuerzan las dimensiones 32x32
                padded = cv2.copyMakeBorder(thresh, top = dY, bottom = dY, left = dX, right = dX, borderType = cv2.BORDER_CONSTANT, value = (0, 0, 0))
                padded = cv2.resize(padded, (28, 28))

                # Se prepara la imagen para la clasificación mediante nuestro modelo OCR
                padded = padded.astype("float32") / 255.0
                padded = np.expand_dims(padded, axis = -1)

                chars.append((padded, (x, y, width, height)))

        # Se extraen las ubicaciones de los recuadros delimitadores y los caracteres
        boxes = [box[1] for box in chars]
        chars = np.array([char[0] for char in chars], dtype="float32")

        preds = model.predict(chars)

        # Se define la lista de letras
        labelNames = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        labelNames = [label for label in labelNames]
        
        predictText = ''

        for (pred, (x, y, width, height)) in zip(preds, boxes):
            # Busca la letra en la lista según el indice predicho
            index = np.argmax(pred)
            label = labelNames[index]

            predictText += label
            print("[INFO] {}".format(label))
            
            # Dibuja las predicciones en la imagen
            """ cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
            cv2.putText(image, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2) """
            
        # Muestra la imagen
        """ cv2.imshow("Image", image)
        cv2.waitKey(0) """

        print(predictText)
        print(imgText.replace(' ', ''))
        
        # Muestra el porcentaje de similitud de los textos procesados
        lev = Levenshtein()
        s1 = lev.similarity(predictText, imgText.replace(' ', ''))
        messagebox.showinfo("Found text",  f'The percentage of similarity between the recognized texts is: {s1}')
        
        return imgText, predictText     

    def __reproduceTranslation(self):
        converter = pyttsx3.init() 

        converter.setProperty('rate', 150) 
        converter.setProperty('volume', 1) 

        voices = converter.getProperty('voices')
        
        if self.__DEFAULT_LANG == 'en':
            converter.setProperty('voice', voices[1].id) # Ingles
        else:
            converter.setProperty('voice', voices[0].id) # Español

        converter.say(self.__translatedText)
        
        converter.runAndWait() 
        converter.stop()

    def __createTextarea(self, text):
        textarea = Text(self.__root, height = 15, width = 120)
        textarea.place(x = 30, y = self.__IMAGE_HEIGHT + 190)
        textarea.insert(INSERT, text)
        textarea.config(state = DISABLED)  