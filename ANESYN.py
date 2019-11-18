#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Ejemplo ejecución: "python aux.py basesDatos/csv/ionosphere.csv"
"""
Created on jue abr 28 17:42:16 2016
@author: fjMaestre
"""

import sys
import numpy as np
import pandas as pd
from scipy.stats import itemfreq
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_fscore_support as score
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
import matplotlib.pyplot as plt
import scipy as scp
from itertools import permutations


def configuration():
    """"Función para la configuración del sistema
    Parametros por defecto
    - test_size = 0.5  # Tamaño porcentual del conjunto de test
    - train_size = 0.3  # Tamaño porcentual del conjunto de entrenamiento
    - runs = 10
    - beta = 1  # β∈[0, 1] specify the balance level after generation of the synthetic data (β = 1 means fully balanced)
    - k = 5  # Number of neighbors to calculate the border
    - n = 0.5 # Factor de ponderación entre cercanía al borde o cercanía al primer padre
    - imbalanced_degree = None  Variable que se utilizará para encontrar las clases minoritarias en tiempo de ejecución
    - rmv_dup = False
    - rmv_noise = False
    """
    test_size = 0.5  # Tamaño porcentual del conjunto de test
    train_size = 0.3  # Tamaño porcentual del conjunto de entrenamiento
    runs = 100
    beta = 1  # β∈[0, 1] specify the balance level after generation of the synthetic data (β = 1 means a fully balanced)
    k = None  # Number of neighbors to calculate the border
    n = 0.7
    imbalanced_degree = None  # Variable que se utilizará para encontrar las clases minoritarias en tiempo de ejecución
    rmv_dup = False
    rmv_noise = False

    return test_size, train_size, runs, beta, k, n, imbalanced_degree, rmv_dup, rmv_noise


def lectura_datos(ruta, rmv_dup=False):
    """ Realiza la lectura de datos.
        Recibe los siguientes parámetros:
            - ruta: Ruta de la base de datos incluyendo el nombre del archivo.
        Devuelve:
            - inputs: Matriz con las variables de entrada.
            - outputs: Matriz con las variables de salida.
    """

    try:
        data_set = pd.read_csv(ruta, header=None)
    except IOError:
        print "no existe la BBDD introducida\n"
        sys.exit("Parando ejecución")

    if rmv_dup:
        data_set = buscar_duplicados(data_set)
    # Normaliza las variables de entrada
    # Estandarización con media=0 y varianza=1
    inputs = preprocessing.scale(np.array(data_set.values[:, 0:-1], dtype='float'))
    outputs = data_set.values[:, -1]

    return inputs, outputs


def buscar_duplicados(data_set):
    """ Elimina los duplicados en la BBDD.
        Recibe los siguientes parámetros:
            - data_set: Conjunto de datos.
        Devuelve:
            - data_set: Conjunto de datos (sin ducplicados).
    """
    data_set.drop_duplicates(keep='first', inplace=True)
    return data_set


def bi_clase(outputs, imbalanced_degree=None):
    """ Transforma el conjunto de datos en otro de dos clases. Pertenencia a la clase minoritaria o no.

    Esta función también se podrá utilizar para detectar las clases minoritarias en tiempo de ejecución. Todas las
    clases minoritarias se juntaran formando una única clase y lo mismo pasara con las clases mayoritarias. Se calculará
    la presencia acumulada de asl clases minoritarias mientars que esta suma sea menor que imbalanced_degree.
    Las clases que hayan sido utilizadas en dicha suma, serán consideradas como minoritaria. Por tanto es muy importante
    conocer la distribución de la BBDD y ajustar imbalanced_degree de forma precisa.

    *** Se recomienda que se haga un formateo de la base de datos de forma independiente a este código, para un ajuste
    más preciso ***

        Recibe los siguientes parámetros:
            - outputs: Matriz con las variables de salida.
            - imbalanced_degree (opcional): Grado de desbalanceo para determinar las clases minoritarias. Si no se
              introduce solo se considerará una clase como minoritaria y todas las demás mayoritarias
        Devuelve:
            - outputs: Variables de salida (Con solo dos clases).
    """

    clases = conteo_clases(outputs)
    if imbalanced_degree is not None:
        imbalance = clases[:, 1] / float(clases[:, 1].sum())  # imbalance ratio
        imbalance = np.cumsum(imbalance)
        etiquetas = clases[:, 0][imbalance < imbalanced_degree]
        indices = np.where(np.in1d(outputs, etiquetas))[0]
        mask = np.zeros(outputs.shape[0], dtype=bool)
        mask[indices] = True
        if not etiquetas.size:
            print 'Para la BBDD utilizada, el valor de imbalanced_degree debe ser almenos superior a', imbalance[0]
            print 'Introduce un valor superior a este o "None", para que el sistema pueda ejecutarse'
            sys.exit("Valor erroneo para imbalanced_degree")
    else:
        mask = (outputs == clases[0][0])

    return mask.astype(int)


def particion_datos(inputs, outputs, test_size=0.5, train_size=0.5):
    """ Realiza la partición de los datos. Crea un conjunto de entrenamiento y otro de test, y si se desea puede crear
        también un conjunto de validación

        Recibe los siguientes parámetros:
            - inputs: Matriz con las variables de entrada.
            - outputs: Matriz con las variables de salida.
            - test_size: Porción estratificada de la base de datos que se utilizará en el test (default is 0.5)
            - train_size: Porción estratificada de la base de datos que se utilizará en el entrenamiento (default  0.5)
        Devuelve:
            - train_inputs: Matriz con las variables de entrada de
              entrenamiento.
            - train_outputs: Matriz con las variables de salida de
              entrenamiento.
            - test_inputs: Matriz con las variables de entrada de
              test.
            - test_outputs: Matriz con las variables de salida de
              test.
    """

    rest_size = 1 - (test_size + train_size)
    # La suma de los tres conjuntos debe ser 1
    if round(test_size + train_size + rest_size, 3) != 1:
        sys.exit("En particion_datos: Los tamaños relativos deben sumar estrictamente 1.0")

    # El tamaño relativo de los conjuntos debe estar entre 0 y 1
    if not (0 <= test_size <= 1 and 0 <= train_size <= 1 and 0 <= rest_size <= 1):
        sys.exit("En particion_datos: Los tamaños relativos deben estar todos entre 0.0 y 1.0")

    train_inputs = train_outputs = test_inputs = test_outputs = None

    train_inputs, test_inputs, train_outputs, test_outputs = train_test_split(inputs, outputs, test_size=test_size,
                                                                              stratify=outputs)

    if rest_size > 0:
        train_inputs, val_inputs, train_outputs, val_outputs = train_test_split(train_inputs, train_outputs,
                                                                                test_size=(rest_size / (1 - test_size)),
                                                                                stratify=train_outputs)

    return train_inputs, train_outputs, test_inputs, test_outputs


def conteo_clases(outputs):
    """ Realiza la identificación de clases y un recuento del número de apariciones de cada una.

        Recibe los siguientes parámetros:
            - outputs: Matriz con las variables de salida.
        Devuelve:
            - labels: Matriz con las clases y su tamaño (ordenada ascendentemente por columna 1)[
                 columna 0: Etiqueta de clase.
                 columna 1: Tamaño de clase.
            ]
    """

    labels = itemfreq(outputs)
    labels = labels[labels[:, 1].argsort()]

    return labels


def class_index(outputs, label):
    """ Calcula los índices de una clase.
        Recibe los siguientes parámetros:
            - outputs: Matriz con las variables de salida.
            - label: Etiqueta de la clase a identificar
        Devuelve:
            - index: Máscara binaria que representa los índices de la clase identificada
    """
    index = outputs == label

    return index


def ratio_borde(inputs_set1, inputs_set2, outputs_set2, minority_label, k=None, return_rlist=False):
    """ calcula el ratio de cercanía al borde (vecinos clase mayoritaria / k vecinos)
        Recibe los siguientes parámetros:
            - inputs_set1: Conjunto de entrada al que se le calcularán los vecinos con respecto a inputs_set2.
            - inputs_set2: Conjunto de entrada de donde se calcularán los vecinos de inputsSet1.
            - outputs_set2: Matriz con las variables de salida de inputsSet2.
            - minority_label: Etiqueta de la clase minoritaria
            - k(opcional): Número de vecinos a tener en cuenta (si no se proporciona se calculará automaticamente)
            - return_rlist(opcional): Si es True, también devuelve rlis sin normalizar (default False)
        Devuelve:
            - normalizedrlist: Vector normalizado con los ratios de cercanía al borde
            - rlist(opcional): Vector con los ratios de cercanía al borde sin normalizar
    """

    neighbors, k = neighbors_calculator(inputs_set1, inputs_set2, outputs_set2, minority_label, k, self_neighbour=False)

    # _todo aquello que no sea clase minoritaria se considerará clase mayoritaria
    majorclass__neighbors = np.sum(neighbors != minority_label, axis=1, dtype='float')

    rlist = majorclass__neighbors / k
    if rlist.sum():
        normalizedrlist = rlist / rlist.sum()
    else:
        normalizedrlist = rlist

    if return_rlist:
        return normalizedrlist, k, rlist
    return normalizedrlist, k


def neighbors_calculator(inputs_set1, inputs_set2, outputs_set2, minority_label, k=None, self_neighbour=False):
    """ calcula la etiqueta de clase de los k vecinos más cercanos
        Recibe los siguientes parámetros:
            - inputsSet1: Conjunto de entrada al que se le calcularan los vecinos con respecto a inputsSet2.
            - inputsSet2: Conjunto de entrada de donde se calcularan los vecinos de inputsSet1.
            - outputs_set2: Matriz con las variables de salida de inputsSet2.
            - k (opcional): Número de vecinos a tener en cuenta (si no se proporciona se calculará automáticamente)
            - minority_label: Etiqueta de la clase minoritaria (default 1)
            - selfNeighbour: Si es True, se considerará que un patrón es vecino de sí mismo (default False)
        Devuelve:
            - elegidos: Matriz con la etiqueta de clase de los patrones vecinos [
                 filas: Una fila por cada individuo de conjunto1.
                 columna: Una columna por cada vecino calculado.
            ]
            - k: número de vecinos tenidos en cuenta
    """

    dist = scp.spatial.distance.cdist(inputs_set1, inputs_set2, 'euclidean')
    nearest_index = np.argsort(dist, axis=1)

    if k is None:
        k = k_calculator(outputs_set2[nearest_index], minority_label)

    elegidos = nearest_index[:, 1 * (not self_neighbour):k + 1 * (not self_neighbour)]
    return outputs_set2[elegidos], k


def k_calculator(neighbors, minority_label):
    """ calcula en número de vecinos a tener en cuenta para que cualquier patrón pueda tener al menos un vecino de
        la clase mayoritaria
            Recibe los siguientes parámetros:
                - neighbors: Matriz con la etiqueta de clase de los patrones vecinos [
                     Filas: Una fila por cada patrón.
                     Columna: Una columna por cada vecino calculado.
                ]
                - minority_label: etiqueta de la clase minoritaria
            Devuelve:
                - k: numero de vecinos a tener en cuenta
        """

    # Se buscan los índices de los vecinos que son de la clase mayoritaria
    indices = np.where(neighbors != minority_label)

    # Se busca el primer vecino de la clase mayoritaria de cada patrón de  la clase mayoritaria
    index = np.unique(indices[0], return_index=True)[1]

    # El maximo se guarda en k
    k = indices[1][index].max() + 1
    if k < 5:
        k=5
    return k


def noise_remover(inputs, outputs, index, rlist):
    """ Elimina los patrones considerados como ruido
        (patrones de la clase minoritaria donde todos sus vecinos son de la clase mayoritaria)
            Recibe los siguientes parámetros:
                - inputs: Matriz con las variables de entrada.
                - outputs: Matriz con las variables de salida.
                - index: Índices de la clase minoritaria
                - rlist: Vector con los ratios de cercanía al borde sin normalizar
            Devuelve:
                - inputs: Matriz con las variables de entrada sin ruido.
                - outputs: Matriz con las variables de salida sin ruido.
                - index: Índices de la clase minoritaria sin ruido.
                - normalizedrlist: Vector con los ratios de cercanía al borde normalizado tras eliminar el ruido
    """
    noise_index = np.copy(index)
    noise = rlist == 1

    if sum(np.logical_not(noise)) > 15:
        # índices de patrones de la clase minoritaria para los que todos sus vecinos son de la clase mayoritaria (Ruido)
        noise_index[index] = noise

        rlist = rlist[~noise]
        normalizedrlist = rlist / rlist.sum()

        return inputs[~noise_index], outputs[~noise_index], index[~noise_index], normalizedrlist
    normalizedrlist = rlist / rlist.sum()
    return inputs, outputs, index, normalizedrlist


def generate_synthetics(inputs, outputs, minority_label, k, n, G):
    """ Genera patrones sintéticos utilizando una metaheurística y los incluye en el conjunto de datos.
        Recibe los siguientes parámetros:
            - inputs: Matriz con las variables de entrada.
            - outputs: Matriz con las variables de salida.
            - minority_label: Etiqueta de la clase minoritaria
            - normalizedrlist: Vector con los ratios de vecindad normalizado
            - k: Número de vecinos a tener en cuenta
            - n: Factor de ponderación entre cercanía al borde o cercanía al primer padre
            - G: Número de sintéticos a generar
        Devuelve:
            - newdata_inputs: Matriz con las variables de entrada + sintéticos.
            - newdata_outputs: Matriz con las variables de salida + sintéticos.
    """

    minority_index = class_index(outputs, minority_label)  # índices de la clase minoritaria
    minority_inputs = inputs[minority_index]
    normalizedrlist, k = ratio_borde(minority_inputs, inputs, outputs, minority_label=minority_label, k=k,
                                     return_rlist=False)  # ratio de cercanía normalizado

    distancias, indices = neighbors_index(minority_inputs, minority_inputs, k, return_distances=True,
                                          self_neighbour=False)

    # inicializaciones
    synthetics_inputs = np.empty((0, inputs.shape[1]))
    synthetics_outputs = (np.ones(G, dtype='int') * minority_label)
    newdata_outputs = np.array(np.append(outputs, synthetics_outputs))

    g = g_calculator(normalizedrlist, G)

    for i in np.nonzero(g)[0]:  # un ciclo por cada individuo
        friends = better_choice(distancias[i], indices[i], normalizedrlist[indices[i]], g[i], n=n)
        synthetics = minority_inputs[i] + (minority_inputs[friends] - minority_inputs[i]) * np.random.rand(1)
        synthetics_inputs = np.append(synthetics_inputs, synthetics, axis=0)

    newdata_inputs = np.array(np.concatenate((inputs, synthetics_inputs), axis=0))

    return newdata_inputs, newdata_outputs


def neighbors_index(inputs_set1, inputs_set2, k=5, return_distances=False, self_neighbour=False):
    """ calcula los índices y si es necesario las distancias de los patrones más cercanos de inputs_set2 a inputs_set1
        Recibe los siguientes parámetros:
            - inputs_set1: Conjunto de entrada al que se le calcularan los vecinos con respecto a inputs_set2.
            - inputs_set2: Conjunto de entrada de donde se calcularan los vecinos de inputs_set1.
            - k: Número de vecinos a tener en cuenta (default k=5)
            - return_distances (opcional): Booleano que determina si se devuelve las distancias o no (default False)
            - selfNeighbour: Si es True, se considerará que un patrón es vecino de si mismo (default False)
        Devuelve:
            - elegidos: Matriz con los índices de los patrones vecinos [
                 Filas: Una fila por cada patrón de conjunto1.
                 Columna: Una columna por cada vecino calculado.
            ]
            - dist (opcional): Matriz con las distancias de los patrones vecinos [
                 Filas: Una fila por cada patrón de conjunto1.
                 Columna: Una columna por cada vecino calculado.
            ]
    """

    dist = scp.spatial.distance.cdist(inputs_set1, inputs_set2, 'euclidean')

    elegidos = np.argsort(dist, axis=1)[:, 1 * (not self_neighbour):k + 1 * (not self_neighbour)]

    if return_distances:
        dist = np.sort(dist)[:, 1 * (not self_neighbour):(k + 1 * (not self_neighbour))]
        return dist, elegidos
    return elegidos


def g_calculator(normalizedrlist, G):
    """ Calcula cuantos sintéticos debe generar cada patrón de la clase minoritaria
            Recibe los siguientes parámetros:
                - normalizedrlist: Vector normalizado con los ratios de cercanía al borde
                - G: Número de sintéticos a generar
            Devuelve:
                - g_trunc: Vector con los sintéticos que debe generar cada patrón de la clase minoritaria
        """

    g_float = normalizedrlist * G

    # Primero asigna a cada patrón su parte entera correspondiente
    g_trunc = np.trunc(g_float)
    restantes = G - int(sum(g_trunc))

    if restantes:
        # Utiliza el resto para asignar los restantes hasta G
        resto = g_float - g_trunc
        probabilities = resto / restantes

        # Los que más resto tengas tienen más posibilidades de ser elegidos para generar un sintético adicional
        elegidos = np.random.choice(range(resto.shape[0]), restantes, p=probabilities, replace=False)
        g_trunc[elegidos] += 1

    return g_trunc.astype(int)


def better_choice(distancias, indices, normalizedrlist, g, n=0.5):
    """ Utiliza un método para elegir al segundo padre con el que generar un sintético
        Recibe los siguientes parámetros:
            - distancias: Distancias de los k vecinos.
            - indices: Índices de los vecinos (para devolver aquellos más cercanos)
            - normalizedrlist: Vector normalizado con los ratios de cercanía al borde
            - n: Factor de ponderación entre cercanía al borde o cercanía al primer padre
            - g: Número de segundos padres a elegir
        Devuelve:
            - elegidos: Índices de los segundos padres elegidos
    """

    factor1 = normalizedrlist / normalizedrlist.sum()
    vector2 = 1 / distancias
    factor2 = vector2 / vector2.sum()

    importancia = (n * factor1) + ((1 - n) * factor2)

    probabilities = importancia / importancia.sum()
    elegidos = np.random.choice(indices, g, p=probabilities)

    return elegidos


def tester(train_inputs, train_outputs, test_inputs, test_outputs, datos, clf=None):
    """ Almacena las medidas de precisión calculadas en diferentes ejecuciones en una matriz
        Recibe los siguientes parámetros:
        - train_inputs: Matriz con las variables de entrada de entrenamiento.
        - train_outputs: Matriz con las variables de salida de entrenamiento.
        - test_inputs: Matriz con las variables de entrada de test.
        - test_outputs: Matriz con las variables de salida de test.
        - datos: Matriz con las medidas de precisión de ejecuciones anteriores
        - clf (opcional): Árbol ya entrenado
        Devuelve:
            - datos: Matriz con las medidas de precisión de cada ejecución [
                 Filas: Una fila por cada medida de precisión.
                 Columnas: Una columna por cada ejecución.
            ]
    """

    oa, precision, recall, f1_measure, g_mean, auc = evaluator(train_inputs, train_outputs, test_inputs, test_outputs,
                                                               clf)
    one_loop_dates = np.hstack((oa, precision, recall, f1_measure, g_mean, auc))

    return np.c_[datos, one_loop_dates]


def evaluator(train_inputs, train_outputs, test_inputs, test_outputs, clf=None):
    """ Calcula una serie de medidas de precisión para evaluar la clasificación
        Recibe los siguientes parámetros:
            - train_inputs: Matriz con las variables de entrada de entrenamiento.
            - train_outputs: Matriz con las variables de salida de entrenamiento.
            - test_inputs: Matriz con las variables de entrada de test.
            - test_outputs: Matriz con las variables de salida de test.
            - clf (opcional): Árbol ya entrenado (Si no se proporciona se creara y entrenará un árbol en la ejecución
        Devuelve:
            - oa
            - precision
            - recall
            - f1_measure
            - g_mean
            - auc
    """

    if clf is None:
        clf = decision_tree(train_inputs, train_outputs)

    predicted__test = clf.predict(test_inputs)

    oa = clf.score(test_inputs, test_outputs)
    precision, recall, f1_measure, support = score(test_outputs, predicted__test, beta=1)
    g_mean = np.sqrt(recall[0] * recall[1])

    # fpr, tpr, thresholds = roc_curve(test_outputs, predicted__test)
    # auc2 = metrics.auc(fpr, tpr)
    auc = roc_auc_score(test_outputs, predicted__test)

    return oa, precision, recall, f1_measure, g_mean, auc


def imprimir_resultado(datos, aux1_dates=None, aux2_dates=None, aux1_name=['Aux1_column'], aux2_name=['Aux2_column'],
                       to_screen=True, to_csv=True, file_name='Results'):
    """ Imprime en pantalla o sobre un fichero la media de las medidas de precisión alcanzadas en cada ejecución
        Recibe los siguientes parámetros:
        - datos: Matriz con las medidas de precisión de ejecuciones anteriores
        - aux1_dates (opcional): Matriz con medidas de precisión de ejecuciones anteriores sobre un conjunto auxiliar
        - aux2_dates (opcional): Matriz con medidas de precisión de ejecuciones anteriores sobre un conjunto auxiliar
        - aux1_name: Nombre de la columna donde aparecerán las medidas del conjunto Aux1_column.(default 'Aux1_column')
        - aux2_name: Nombre de la columna donde aparecerán las medidas del conjunto Aux2_column.(default 'Aux2_column')
        - to_screen (opcional): Si es True, imprime las medidas por pantalla (default True)
        - to_csv (opcional): Si es True, creara un fichero con las medidas calculadas (default True)
        - file_name (opcional): Nombre del fichero que será creado con las medidas. Solo tendrá sentido si toCsv=True.
        (default 'Results')
    """

    datos = datos.reshape(-1, 1)
    cabecera = ['ANESYN']
    if aux1_dates is not None:
        aux1_dates = aux1_dates.reshape(-1, 1)
        cabecera += aux1_name
        datos = np.c_[datos, aux1_dates]

    if aux1_dates is not None:
        aux1_dates = aux1_dates.reshape(-1, 1)
        cabecera += aux2_name
        datos = np.c_[datos, aux1_dates]

    medidas = np.array(
        ['OA', 'Precision Class1', 'Precision Class0', 'Recall Class1', 'Recall Class0', 'F1_measure Class1',
         'F1_measure Class0', 'G_mean', 'AUC'])
    df = pd.DataFrame(datos, columns=cabecera, index=medidas)

    if to_screen:
        print df
    if to_csv:
        file_name = ''.join(file_name)
        file_name = file_name.replace(".csv", "")
        file_name += '.csv'
        df.to_csv(file_name, header=True, index=True, sep=',')


def decision_tree(inputs, outputs):
    """ Función que genera un árbol de decisión
        Recibe los siguientes parámetros:
            - inputs: Matriz con las variables de entrada.
            - outputs: Matriz con las variables de salida.
        Devuelve:
            - clf: Árbol entrenado mediante inputs y outputs

    """
    clf = DecisionTreeClassifier(class_weight=None)
    clf.fit(inputs, outputs)

    return clf


def principal(fichero, semillas, test_size, train_size, beta=1, k_original=5, n=0.5, imbalanced_degree=None,
              rmv_dup=False, rmv_noise=False):

    # inicialización de los datos
    runs = semillas.shape[0]
    inputs, outputs = lectura_datos(fichero, rmv_dup)
    outputs = bi_clase(outputs, imbalanced_degree)

    # Estructura de datos donde se almacenará las medidas de precision
    datos = np.empty((9, 0))

    for loop in xrange(0, runs, 1):
        k = k_original
        semilla = semillas[loop]
        print 'loop:', loop+1, '======', 'seed:', semilla
        np.random.seed(semilla)

        train_inputs, train_outputs, test_inputs, test_outputs = particion_datos(inputs,
                                                                                                          outputs,
                                                                                                          test_size,
                                                                                                          train_size)

        minority_index = class_index(train_outputs, True)  # indices de la clase minoritaria
        clases = conteo_clases(train_outputs)  # column 0 (class label) column 1 (class size)
        minority_label = clases[0][0]  # Etiqueta de la clase minoritaria
        minority_inputs = train_inputs[minority_index]  # entradas de la clase minoritaria

        if rmv_noise and (clases[0][1] / float(clases[1][1])) > 0.1:  # Se elimina ruido
            normalizedrlist, k, rlist = ratio_borde(minority_inputs, train_inputs, train_outputs, True, k,
                                                    return_rlist=True)  # ratio de cercania normalizado
            train_inputs, train_outputs, minority_index, normalizedrlist = noise_remover(train_inputs, train_outputs,
                                                                                         minority_index, rlist)
            clases = conteo_clases(train_outputs)  # column 0 (class label) column 1 (class size)
        else:  # No se elimina ruido
            normalizedrlist, k = ratio_borde(minority_inputs, train_inputs, train_outputs, minority_label=True, k=k,
                                             return_rlist=False)  # ratio de cercania normalizado


        """"variables"""
        # number of synthetic data examples to generate for the minority class
        G = np.ceil(((clases[1:, 1].sum() - clases[0][1]) * beta) - 1).astype(int)


        # # Sin metaheuristica
        aux_inputs, aux_outputs = generate_synthetics(train_inputs, train_outputs, minority_label, k, n, G)


        # //////////////////////////////////////////////////////////////////////////////////////////////////////////////
        # Calculo de precisiones
        # //////////////////////////////////////////////////////////////////////////////////////////////////////////////

        datos = tester(aux_inputs, aux_outputs, test_inputs, test_outputs, datos)

    imprimir_resultado(datos.mean(axis=1), to_screen=True, to_csv=True)

    return True


if __name__ == "__main__":

    ejecucion = "Para ejecutar: ./ 'nombre fichero' 'Ruta de una BBDD en .CSV' 'Semilla para aleatoriedad'"
    try:
        fichero = sys.argv[1]
    except IndexError:
        print "Falta la ruta de una BBDD\n", ejecucion
        sys.exit("Parando ejecución")

    try:
        semilla = int(sys.argv[2])
    except IndexError:
        semilla = np.random.randint(low=10000000, high=99999999)
        print 'semilla aleatoria:', semilla


    test_size, train_size, runs, beta, k, n, imbalanced_degree, rmv_dup, rmv_noise = configuration()

    if k < 2 and k != None:
        print "k vale", k, "pero deberia tener un valor igual o superior a 2"
        sys.exit("Valor erroneo para k")

    if runs < 2:
        print "runs vale", runs, "pero deberia tener un valor igual o superior a 2"
        sys.exit("Valor erroneo para runs")

    if not (0.1 <= beta <= 1):
        print "beta vale", beta, "pero deberia tener un valor entre 0 y 1"
        sys.exit("Valor erroneo para beta")

    if not (0.1 < imbalanced_degree <= 0.5) and imbalanced_degree is not None:
        print "imbalanced_degree vale", imbalanced_degree, "pero deberia tener un valor entre 0 y 0.5"
        sys.exit("Valor erroneo para imbalanced_degree")

    np.random.seed(semilla)
    semillas = np.random.choice(np.asarray(list(map("".join, permutations('12345678'))), dtype=int), runs)


    principal(fichero, semillas, test_size, train_size, beta, k, n, imbalanced_degree=imbalanced_degree,
              rmv_dup=rmv_dup, rmv_noise=rmv_noise)
