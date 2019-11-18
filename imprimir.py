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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def paint(val_array, test_array, first_newinputs_looparray, better_newinputs_looparray,
          to_screen=True, to_pdf=True, file_name='grafica'):


    if to_screen:
        print test_array[1], test_array[-1]
        generaciones = test_array.shape[0]
        first_value = np.ones(generaciones) * test_array[1]
        t= np.arange(0, generaciones, 1)
        graphic = plt.figure()
        pj = plt.plot(t, val_array, 'g-', t, test_array, 'r-', t, first_value, 'b--')
        plt.ylabel('G-Mean')
        plt.xlabel('Iteraciones de Iterated Greedy ')
        plt.title('val= Green, test = Red \ntest con los primeros sinteticos = blue')
        plt.show(pj)


        if to_screen:
            data_to_plot = np.c_[
                first_newinputs_looparray[1:], better_newinputs_looparray[1:]]
            plot = plt.figure()
            ax = plot.add_subplot(111)
            ax.boxplot(data_to_plot)
            plt.ylim([(data_to_plot.min() - 0.005), data_to_plot.max() + 0.005])
            plt.xticks([1, 2], ['Primera_ejecucion ANEIGSYN', 'Ultima_ejecucion ANEIGSYN'])
            # ax.set_title('axes title')
            # ax.set_xlabel('xlabel')
            ax.set_ylabel('G-Mean')
            plt.show()

        if to_pdf:
            graphic.savefig('graphic.pdf')
            with PdfPages('box-plot.pdf') as pdf:
                pdf.savefig(plot)


def graphics_mean():

    name_val = 'validaciones.csv'
    name_test = 'testeos.csv'
    name_first = 'firstSet_Trees.csv'
    name_better = 'betterSet_Trees.csv'
    try:
        file_val = pd.read_csv(('./' + name_val), header=None)
        file_test = pd.read_csv(('./' + name_test), header=None)
        val = np.array(file_val.values[:, :], dtype='float')
        test = np.array(file_test.values[:, :], dtype='float')

        file_first = pd.read_csv(('./' + name_first), header=None)
        first = np.array(file_first.values[:, :], dtype='float')

        file_better = pd.read_csv(('./' + name_better), header=None)
        better = np.array(file_better.values[:, :], dtype='float')


    except IOError:
        print "no existe los archivos \n"
        sys.exit("Parando ejecución")

    val_array = val.mean(axis=1)
    test_array = test.mean(axis=1)
    first_array = first.mean(axis=1)
    better_array = better.mean(axis=1)

    paint(val_array, test_array, first_array, better_array)


if __name__ == "__main__":
    graphics_mean()
