#! /usr/bin/python
#-*- coding: utf8 -*-

import pickle # the serializer class
import numpy, pylab, random, math

# Fonction generant des donnees random, utilisables pour l'apprentissage
def generate_data(out_file_name, allow_overwrite=False):
    # Notre nombre de points doit être divisible par 4
    nb_pts = 40

    # Vérifier qu'on ne va pas écraser de fichier, on pourrait avoir les boules
    if not allow_overwrite:
        try:
            open(out_file_name)
        except IOError:
            pass
        else:
            print("The output file %s already exists. Aborting." % out_file_name)
            exit(1)


    if nb_pts % 4 != 0:
        print("Nombre de points {0} non divisible par 4. Abandon.")
        exit(1)


    # Moitie des elements dans la classe 1 = union de deux gaussiennes
    classA = [(random.normalvariate(-1.5, 1), random.normalvariate(0.5, 1), 1.0) for i in range(nb_pts/4)] + \
             [(random.normalvariate(1.5, 1), random.normalvariate(0.5, 1), 1.0) for i in range(nb_pts/4)]
    
    # Autre moitie dans la classe 2 = une autre gaussienne
    #classB = [(random.normalvariate(0.0, 0.5), random.normalvariate(-0.5, 0.5), -1.0) for i in range (nb_pts/2)]
    classB = [(random.normalvariate(-1.5, 0.5), random.normalvariate(-1.5, 0.5), -1.0) for i in range (nb_pts/2)]

    # Joindre les deux liste et melanger (mais pourquoi ??)
    data = classA + classB
    random.shuffle(data)

    # Mettre tout ça dans un fichier pour l'utiliser plus tard.
    out_file = open(out_file_name, "w")
    pickle.dump((classA, classB, data), out_file)

    return classA, classB, data




def load_data(filename):
    in_file = open(filename)
    classA, classB, data = pickle.load(in_file)
    return classA, classB, data


def show_data(filename):
    classA, classB, _data = load_data(filename)

    pylab.hold(True)        
    pylab.plot([p[0] for p in classA], [p[1] for p in classA], 'bo')
    pylab.plot([p[0] for p in classB], [p[1] for p in classB],'ro')

    pylab.show()


        
if __name__ == "__main__":
    generate_data("fresh_test_set.dat", True)
    show_data("fresh_test_set.dat")
