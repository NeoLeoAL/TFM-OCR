import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

BATCH_SIZE = 200
EPOCHS = 24

IMG_DIM = 28

# PREPARACIÓN DATOS DE ENTRENAMIENTO

def loadA_ZDataset():
    dataset = pd.read_csv("A_Z Handwritten Data.csv").astype('float32')
    dataset.rename(columns={'0':'label'}, inplace=True)

    x_letters = dataset.drop('label', axis = 1)
    y_letters = dataset['label']

    x_lettTrain, x_lettTest, y_lettTrain, y_lettTest = train_test_split(x_letters, y_letters)

    standard_scaler = MinMaxScaler()
    standard_scaler.fit(x_lettTrain)

    x_lettTrain = standard_scaler.transform(x_lettTrain)
    x_lettTest = standard_scaler.transform(x_lettTest)

    X_trainDim = x_lettTrain.shape
    print('Dimensiones de x_lettTrain:', X_trainDim)
    print('Número de ejemplos para entrenamiento:', X_trainDim[0])

    X_testDim = x_lettTest.shape
    print('Dimensiones de x_lettTest:', X_testDim)
    print('Número de ejemplos para entrenamiento:', X_testDim[0])

    x_lettTrain = x_lettTrain.astype('float32')
    x_lettTrain /= 255
    x_lettTrain = x_lettTrain.reshape(X_trainDim[0], IMG_DIM, IMG_DIM, 1)

    x_lettTest = x_lettTest.astype('float32')
    x_lettTest /= 255
    x_lettTest = x_lettTest.reshape(X_testDim[0], IMG_DIM, IMG_DIM, 1)

    x_letters = np.vstack([x_lettTrain, x_lettTest])
    y_letters = np.hstack([y_lettTrain, y_lettTest])
    
    return x_letters, y_letters

# ------------------------------------ #

def generateTrainAndTestData(x_letters, y_letters):
    x_train, x_test, y_train, y_test = train_test_split(x_letters, y_letters)

    print('Dimensiones de x_train:', x_train.shape)
    print('Número de ejemplos para entrenamiento:', x_train.shape[0])

    print('Dimensiones de x_test:', x_test.shape)
    print('Número de ejemplos para entrenamiento:', x_test.shape[0])

    numClass = len(np.unique(y_train))

    y_train = keras.utils.to_categorical(y_train, numClass)
    y_test = keras.utils.to_categorical(y_test, numClass)
    
    return x_train, y_train, x_test, y_test, numClass

# MODELO

def createModel(x_train, y_train, x_test, y_test, numClass):
    model = Sequential()
    model.add(Conv2D(32, kernel_size = (5, 5), activation = 'relu', input_shape = (IMG_DIM, IMG_DIM, 1)))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Conv2D(64, (3, 3), activation = 'relu'))

    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(numClass, activation = 'softmax')) 
    
    model.compile(loss = keras.losses.categorical_crossentropy, optimizer = 'adam', metrics = ['accuracy'])
    model.fit(x_train, y_train, batch_size = BATCH_SIZE, epochs = EPOCHS, verbose = 1, validation_data = (x_test, y_test))
    
    return model

def evaluateModel(model, x_test, y_test, numClass):
    score = model.evaluate(x_test, y_test, verbose = 1)

    x_forPrediction = x_test[0].reshape(1, IMG_DIM, IMG_DIM, 1)
    print(x_forPrediction)

    prediction = model.predict(x_forPrediction)
    print(prediction)
    print(np.sum(prediction))
    print(keras.utils.to_categorical(np.argmax(prediction), numClass))
    
    print('Pérdidas:', score[0])
    print('Precisión:', score[1])
    
def saveModel(model):
    model.save("ocrModel.h5")
        
if __name__ == '__main__':
    x_letters, y_letters = loadA_ZDataset()
    
    x_train, y_train, x_test, y_test, numClass = generateTrainAndTestData(x_letters, y_letters)
    model = createModel(x_train, y_train, x_test, y_test, numClass)
    
    evaluateModel(model, x_test, y_test, numClass)
    saveModel(model)