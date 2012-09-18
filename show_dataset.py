#! /usr/bin/python
#-*- coding: utf-8 -*-


#Importation des fonctions necessaires
import numpy, pylab, random, math
from generate_tests import *
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: %s, dataset_file_name" % sys.argv[0])
        exit(1)

    show_data(sys.argv[1])
