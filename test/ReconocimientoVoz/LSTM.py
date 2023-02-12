import numpy as np
import tensorflow as tf
#from tensorflow.contrib import rnn
import random
import collections

# Parámetros
velocidadDeAprendizaje = 0.001
numeroDeIteracionesParaElEntrenamiento = 50000
iteracionesParaMostrarInfo = 1000
numeroDeEntradas = 3

# Número de unidades ocultas en una celda RNN
numeroDeUnidadesOcultas = 512

def leerDatos(fname):
    with open(fname) as f:
        contenido = f.readlines() #aquí se mete cada línea del fichero en una posición del array 'contenido'
    #aquí cada posición del array 'contenido' se cambia eliminando los espacios en blanco que pudieran haber al principio y al final de cada línea
    contenido = [x.strip() for x in contenido]
    #ahora en cada posición del array 'contenido' se almacena cada palabra de cada línea
    contenido = [palabra for i in range(len(contenido)) for palabra in contenido[i].split()] 
    contenido = np.array(contenido)
    #print(contenido)
    return contenido

def construirDiccionarios(palabras):
    count = collections.Counter(palabras).most_common()
    diccionario = dict()
    for palabra, _ in count:
        diccionario[palabra] = len(diccionario)
    diccionarioInverso = dict(zip(diccionario.values(), diccionario.keys()))
    return diccionario, diccionarioInverso

#predictor
def RNN(x, pesos, biases):
    # redimensionar x a [1, numeroDeEntradas]
    x = tf.reshape(x, [-1, numeroDeEntradas])
    x = tf.split(x,numeroDeEntradas,1)

    # LSTM de 2 capas: cada capa tiene un número de unidades ocultas dado por numeroDeUnidadesOcultas.
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(numeroDeUnidadesOcultas),rnn.BasicLSTMCell(numeroDeUnidadesOcultas)])

    # LSTM de 1 capa: cada capa tiene un número de unidades ocultas dado por numeroDeUnidadesOcultas
    # pero tiene una menor "accuracy" (precisión).
    # Descomentar la línea de abajo para comprobarlo pero comentar las líneas de arriba para el LSTM de 2 capas
    # rnn_cell = rnn.BasicLSTMCell(numeroDeUnidadesOcultas)

    # Generación de la predicción
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32) 
    #outputs=[]1 x numeroDeUnidadesOcultas
    #pesos=[]numeroDeUnidadesOcultas x totalPalabrasEnDiccionario
    #biases=[]1 x totalPalabrasEnDiccionario
    # Hay tantas salidas como numeroDeEntradas pero sólo nos interesa la última salida
    return tf.matmul(outputs[-1], pesos['out']) + biases['out']

#-----------------------------------------
#PREPARACIÓN DE LOS DATOS DE ENTRENAMIENTO

ficheroDeEntrenamiento = 'TextoEntrenamiento.txt' #Gramatica.txt ó ControlRobot.txt ó TextoDeEntrenamiento.txt
palabrasDeEntrenamiento = leerDatos(ficheroDeEntrenamiento)
print("Palabras de entrenamiento cargadas... ")

diccionario, diccionarioInverso = construirDiccionarios(palabrasDeEntrenamiento)
print("Diccionario <palabra,id>:")
print(diccionario)
print("Diccionario inverso <id,palabra>:")
print(diccionarioInverso)
totalPalabrasEnDiccionario = len(diccionario)
""" 
#--------------------------------------
#CREACIÓN DEL MODELO DE LA RED NEURONAL

tf.reset_default_graph() #limpiamos el graph antes de empezar a añadirle elementos

# Definimos la estructura que tendrá el "graph" de tensorflow

# Un placeholder se usa para indicar que creamos una variable con una determinada estructura a la que le asignaremos valores más tarde
# Nos creamos dos variables de tipo placeholder
# "None" se usa para reajustar el tamaño del array automaticamente según las necesidades específicas del código
# "None" significa que se puede entrenar la red con cualquier número de ejemplos
x = tf.placeholder("float", [None, numeroDeEntradas, 1]) # x representará las entradas de la RNN. 
y = tf.placeholder("float", [None, totalPalabrasEnDiccionario]) # y representará todas las palabras 

# Pesos y biases de el nodo de salida de la RNN
pesos = { #pesos=[]numeroDeUnidadesOcultas x totalPalabrasEnDiccionario
    'out': tf.Variable(tf.random_normal([numeroDeUnidadesOcultas, totalPalabrasEnDiccionario]))
}
biases = { #biases=[]1 x totalPalabrasEnDiccionario
    'out': tf.Variable(tf.random_normal([totalPalabrasEnDiccionario]))
}

prediccion = RNN(x, pesos, biases) #prediccion=[]1 x totalPalabrasEnDiccionario

# Indicamos las funciones de "Loss" (pérdidas, error) y optimizador del modelo que vamos a usar
funcionDeCoste = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediccion, labels=y))
optimizador = tf.train.RMSPropOptimizer(learning_rate=velocidadDeAprendizaje).minimize(funcionDeCoste)

# Indicamos las funciones de evaluación del modelo que vamos a usar
# Ver el siguiente enlace para comprender cómo funciona tf.argmax(XXX,1)
# https://stackoverflow.com/questions/41708572/tensorflow-questions-regarding-tf-argmax-and-tf-equal

#correct_pred contiene un array con valores 1 allí donde tf.argmax(prediccion,1) y tf.argmax(y,1) tienen el mismo valor
correct_pred = tf.equal(tf.argmax(prediccion,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) #hacemos un cast al tensor correct_pred convirtiéndolo en valores de tipo float32

# Arrancamos el "graph"
with tf.Session() as session:
    session.run(tf.global_variables_initializer()) # Inicialización de las variables
    
    idIteracion = 0
    offset = random.randint(0,numeroDeEntradas+1)
    end_offset = numeroDeEntradas + 1
    acc_total = 0
    loss_total = 0
    
    #-------------------------------------------------------------
    # PROCESO DE APRENDIZAJE
    
    while idIteracion < numeroDeIteracionesParaElEntrenamiento:
        # Asignamos aleatoriamente un offset a partir del cual 
        # elegiremos un conjunto con palabras consecutivas como numeroDeEntras tengamos.
        if offset > (len(palabrasDeEntrenamiento)-end_offset):
            offset = random.randint(0, numeroDeEntradas+1)
        # Seleccionamos aleatoriamente un conjunto con tantas palabras consecutivas como numeroDeEntradas 
        # tengamos para usarlas para indicar a la RNN cuales son las palabras "condicionantes" o predecesoras
        # para obtener la siguiente palabra (conscuenciaOnehot)
        entradasPredecesoras = [ [diccionario[ str(palabrasDeEntrenamiento[i])]] for i in range(offset, offset+numeroDeEntradas) ]
        # le damos el formato adecuado
        entradasPredecesoras = np.reshape(np.array(entradasPredecesoras), [-1, numeroDeEntradas, 1])
        
        #conscuenciaOnehot indica la palabra que sigue a la secuencia 
        #(llamémosle "consecuencia") dada por entradasPredecesoras
        #Empezamos a construir un array tipo "oneshot":
        #-creo un array inicialmente todo con ceros
        conscuenciaOnehot = np.zeros([totalPalabrasEnDiccionario], dtype=float)
        #-pongo un uno en la posición que identifica al id de la palabra en cuestión
        conscuenciaOnehot[diccionario[str(palabrasDeEntrenamiento[offset+numeroDeEntradas])]] = 1.0
        #-le damos el formato adecuado: 1 fila, y tantas columnas como sean necesarias
        conscuenciaOnehot = np.reshape(conscuenciaOnehot,[1,-1])
        
        #REALIZAMOS EL APRENDIZAJE
        _, acc, loss, prob_palabraSiguiente_pred = session.run([optimizador, accuracy, funcionDeCoste, prediccion], \
                                                feed_dict={x: entradasPredecesoras, y: conscuenciaOnehot})
        loss_total += loss
        acc_total += acc
        
        #Cada iteracionesParaMostrarInfo se muestra información sobre el estado del aprendizaje
        if (idIteracion+1) % iteracionesParaMostrarInfo == 0:
            print("Iteración= " + str(idIteracion+1) + ", Loss media= " + \
                  "{:.6f}".format(loss_total/iteracionesParaMostrarInfo) + ", Accuracy media= " + \
                  "{:.2f}%".format(100*acc_total/iteracionesParaMostrarInfo))
            acc_total = 0
            loss_total = 0
            #se toman las palabras de entrenamiento (tantas como entradas tenga nuestra LSTM)
            #a partir de una posición dentro del array "palabrasDeEntrenamiento" dada por el offset
            palabrasPredecesoras = [palabrasDeEntrenamiento[i] for i in range(offset, offset + numeroDeEntradas)] 
            #cogemos el último elemento del array "palabrasDeEntrenamiento" para el offset dado
            #este elemento debería corresponderse con la salida predicha (prob_palabraSiguiente_pred)
            palabraSiguiente = palabrasDeEntrenamiento[offset + numeroDeEntradas]
            #con el diccionario inverso obtenemos la palabra que se correspondería con el código "onehot" de la palabra predicha
            palabraSiguiente_pred = diccionarioInverso[int(tf.argmax(prob_palabraSiguiente_pred, 1).eval())]
            print("%s - [%s] vs [%s]" % (palabrasPredecesoras,palabraSiguiente,palabraSiguiente_pred))
            
        idIteracion += 1
        offset += (numeroDeEntradas+1)
    print("Optimización acabada!")
    
    #-------------------------------------------------------------
    # COMPROBACIÓN DEL APRENDIZAJE
    
    #Número de palabras consecutivas que generará la RNN
    #es decir, la longitud del texto que deberá crear la RNN, expresada en número de palabras
    #numeroDePalabrasEncadenadasAgenerar=32
    
    while True:
        #preparo una variable con el mensaje para pedir palabras
        varSecuenciaAprobar = "Escriba %s palabras: " % numeroDeEntradas 
        secuenciaAprobrar = input(varSecuenciaAprobar)
        #una vez tomada la secuenciaAprobrar, se le quitan los espacios en blanco que pueda haber al principio y al final de la secuenciaAprobrar
        secuenciaAprobrar = secuenciaAprobrar.strip()
        #se trocea la secuenciaAprobrar rompiéndola por los espacios en blanco, obteniendo sólo la lista de palabras que la contienen
        palabrasDeLaSecuenciaAprobrar = secuenciaAprobrar.split(' ')
        #Si el número de palabras del mensaje introducido no se corresponde con el número de entradas de la RNN
        #se salta el resto del código y, por tanto, se volverá a pedir que se escriban las palabras
        #if len(palabrasDeLaSecuenciaAprobrar) != numeroDeEntradas:
        #    continue
        
        #Completo la secuencia introducida, insertando . antes de ella hasta completar un total de numeroDeEntradas palabras
        while len(palabrasDeLaSecuenciaAprobrar) <= numeroDeEntradas:
            palabrasDeLaSecuenciaAprobrar.insert(0,". ")
        
        #Si la secuencia introducida tiene más de numeroDeEntradas palabras, voy quitando las del principio
        while len(palabrasDeLaSecuenciaAprobrar) > numeroDeEntradas:
            palabrasDeLaSecuenciaAprobrar.pop(0)
        
        siguientePalabraPredicha=""
        fraseGenerada=""
        
        try:
            entradasPredecesoras = [diccionario[str(palabrasDeLaSecuenciaAprobrar[i])] for i in range(len(palabrasDeLaSecuenciaAprobrar))]
            #for i in range(numeroDePalabrasEncadenadasAgenerar):
            while ("." not in siguientePalabraPredicha) and ("?" not in siguientePalabraPredicha):
                palabrasPrevias = np.reshape(np.array(entradasPredecesoras), [-1, numeroDeEntradas, 1])
                #se obtiene la salida predicha (en formato "probabilistico") para las palabras dadas
                prob_palabraSiguiente_pred = session.run(prediccion, feed_dict={x: palabrasPrevias})
                #convertimos prob_palabraSiguiente_pred en su número de palabra correspondiente
                #buscando la posición del valor máximo dentro del vector
                palabraSiguiente_pred_id = int(tf.argmax(prob_palabraSiguiente_pred, 1).eval()) 
                siguientePalabraPredicha = diccionarioInverso[palabraSiguiente_pred_id]
                #la frase generada será la palabra predicha anterior añadiéndole la nueva palabra predicha
                fraseGenerada = "%s %s" % (fraseGenerada,siguientePalabraPredicha)
                #quito la primnera palabra
                entradasPredecesoras = entradasPredecesoras[1:]
                #Añadimos el código "onehot" de la palabra predicha al final del array "entradasPredecesoras"
                entradasPredecesoras.append(palabraSiguiente_pred_id)
            print(fraseGenerada)
        except:
            print("Una palabra no está en el diccionario") """