#! /usr/bin/python
#-*- coding: utf-8 -*-

#Importation des librairies
from cvxopt.solvers import qp
from cvxopt.base import matrix

#Importation des fonctions necessaires
import numpy, pylab, random, math


# !!! ATTENTION !!!
# Numpy a des subtilites discutables entre matrice et array.
#   - Le type numpy.array ne contient que le nombre de dimensions necessaires. Autrement
#     dit, il n'y a aucune discinction entre un vecteur ligne et un vecteur colone: ce sont
#     deux arrays de dimension 1. Cela implique que le 'dot product' numpy.dot(x, y) sur
#     des numpy.array est TOUJOURS le produit scalaire, et ne peut PAS produire une grande
#     matrice, même si on essaie d'appeler transpose() sur l'un des arguments. L'avantage
#     est qu'on ne se fera pas crier dessus parce que les vecteurs ne sont pas dans le bon
#     sens. Le 'dot product' marche comme un produit de matrice dans le cas ou l'array est
#     de dimension 2. Je n'ai meme pas envie de penser a ce que ca fait en dimension > 2.
#     Sur les numpy.array, l'operateur '*' fait une multiplication coordonnee par coordonnee,
#
#   - Le type numpy.matrix, en revenche, represente une matrice au sens mathematique (sauf
#     que la numerotation commence a 0 et pas a 1 a la difference de matlab). Une numpy.matrix a
#     TOUJOURS deux dimensions. Le 'dot product' (numpy.dot(a, b)) ET l'operation '*' representent
#     la multiplication matricielle normale, par ex. le produit de 2 vecteurs produira
#     soit une grosse matrice soit le produit scalaire selon qu'on multiplie colone par ligne
#     ou ligne par colone. La transposition sur un vecteur represente par un type numpy.matrix
#     a un effet, contrairement aux arrays, la distinction ligne/colone a un sens.


# Number of points in the training set
NB_PTS = 40


# Fonction principale contenant la logique de l'algorithme d'apprentissage des SVM
# In: training_set  Points pour l'entrainement, sous forme de liste de triplets (x, y, classe)
# In: kernel        Kernel a utiliser pour la transformation des donnees
#
# Returns: indicator :: point -> classe
#                       Fonction pouvant servir a classifier un nouvel element:
def learn_indicator(training_data, kernel):
    global NB_PTS # Ca, c'est vraiment de la merde pythonesque...
    # On defini les matrices qui serviront a trouver alpha (cf enonce TP).
    # Comme ce TP est cense nous faire utiliser numpy, faisons-en des arrays numpy
    P = def_P(training_data, kernel)
    q = -numpy.ones(NB_PTS)
    h = numpy.zeros(NB_PTS)
    G = -numpy.eye(NB_PTS)

    # Trouver les alphas solution du probleme d'optimisation
    alphas = find_alphas(P, q, G, h)
    ts = numpy.array([ech[2] for ech in training_data]) # Classe des différents points (t_i dans l'énoncé)
    ys = [ numpy.array([e[0], e[1]]) for e in training_data ] # points du training set sans leur classe

    # Notre fonction indicator a retourner, dependant de alphas et training_data
    # !! Attention, pour faire ça il faut être sûr que alphas, ts, ys ne seront pas modifiés
    # ultérieurement, ce qui est le cas ici. S'ils étaient modifiés, indicator le serait par
    # effet de bord, car Pyton ne gère pas les fermetures.
    def indicator(x):
        kernel_values = numpy.array([kernel(x, y) for y in ys])
        # Sur des arrays, '*' est la multiplication coordonnee par coordonnee
        return sum(alphas * ts * kernel_values)

    return indicator


# Calcule la matrice P tq P_i,j = t_i * t_j * K(x_i, x_j) pour x_i, x_j se baladant
# dans les données d'entrainement
def def_P(training_data, kernel):
    global NB_PTS
    P = numpy.zeros((NB_PTS, NB_PTS))
    for i, d_i in enumerate(training_data):
        for j, d_j in enumerate(training_data):
            t_i, t_j = d_i[2], d_j[2]

            # Kernel prend deux matrices numpy en entree
            x_i = numpy.array([d_i[0], d_i[1]])
            x_j = numpy.array([d_j[0], d_j[1]])

            P[i, j] = t_i * t_j * kernel(x_i, x_j)

    print("Rang de P: %d" % sum([1 for e in numpy.linalg.eigvals(P) if math.fabs(e) > 10e-5]))
    return P



# Utilise cvxopt pour resoudre le probleme (dual) d'optimisation convexe
# Ne pas oublier qu'on fait du calcul numerique, donc un peu de chimie,
# donc tout nombre < 10e-5 sera considere comme etant nul.
def find_alphas(P, q, G, h):
    # Utilisation de cvxopt: cf enonce du TP
    r = qp(matrix(P), matrix(q), matrix(G), matrix(h))
    alphas = numpy.array(list(r['x']))
    # remplacer les valeurs trop petites par 0, garder le reste
    alphas = numpy.where(numpy.fabs(alphas) < 10e-5, 0, alphas)

    return alphas



################### Partie concernant le test #####################


# Fonction generant des donnees random, utilisables pour l'apprentissage
def generateData():
    global NB_PTS
    # Notre nombre de points doit être divisible par 4
    if NB_PTS % 4 != 0:
        new_NB_PTS = math.floor(NB_PTS / 4) * 4
        print("Nombre de points {0} non divisible par 4, utilisera {1} points".format(NB_PTS, new_NB_PTS))
        NB_PTS = new_NB_PTS

    # Moitie des elements dans la classe 1 = union de deux gaussiennes
    classA = [(random.normalvariate(-1.5, 1), random.normalvariate(0.5, 1), 1.0) for i in range(NB_PTS/4)] + \
             [(random.normalvariate(1.5, 1), random.normalvariate(0.5, 1), 1.0) for i in range(NB_PTS/4)]
    
    # Autre moitie dans la classe 2 = une autre gaussienne
    classB = [(random.normalvariate(0.0, 0.5), random.normalvariate(-0.5, 0.5), -1.0) for i in range (NB_PTS/2)]

    # Joindre les deux liste et melanger (mais pourquoi ??)
    data = classA + classB
    random.shuffle(data)
    return classA, classB, data

        

def plotBoundary (classA, classB, indicator):
        pylab.hold(True)        
        pylab.plot([p[0] for p in classA], [p[1] for p in classA], 'bo')
        pylab.plot([p[0] for p in classB], [p[1] for p in classB],'ro')
       
        xrange = numpy.arange(-4, 4, 0.05)
        yrange = numpy.arange(-4, 4, 0.05)

        grid = matrix([[indicator(numpy.array([x, y])) for y in yrange] for x in xrange])

        pylab.contour(xrange, yrange, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))
        pylab.show()



def try_indicator(kernel):
    classA, classB, data = generateData()
    indicator = learn_indicator(data, kernel)
    plotBoundary(classA, classB, indicator)


# Les differents kernels possibles. Un kernel est de prototype
# In: a     array numpy de taille 2 (abscisse, ordonnee)
# In: b     array numpy de taille 2 (abscisse, ordonnee)
#
# Returns   valeur du kernel applique a a et b

# NB: La plupart kernels sont parametrables, on defini donc plutot des constructeurs
# de kernels, prenant ces parametres en argument et retournant une fonction
# respectant le prototype ci-dessus. linear_kernel n'est pas parametrable,
# mais on en defini quand meme un constructeur pour etre consistant avec le
# reste.

def linear_kernel():
    def kernel(a, b):
        return numpy.dot(a, b) + 1
    return kernel

def polynomial_kernel(p):
    def kernel(a, b):
        return (numpy.dot(a, b) + 1) ** p
    return kernel

def radial_basis_kernel(sigma):
    def kernel(a, b):
        delta = a - b
        return math.exp( -numpy.dot(delta, delta) / (2 * sigma**2) )
    return kernel

def sigmoid_kenel(k, delta):
    def kernel(a, b):
        return math.tanh( k*numpy.dot(a, b) - delta )
    return kernel



if __name__ == "__main__" :
    #try_indicator(sigmoid_kenel(1, 1))
    #try_indicator(polynomial_kernel(3))
    try_indicator(radial_basis_kernel(10))

# TRUCS A FAIRE:
#   - essayer de faire du VRAI test, i.e ne donner que la moitié du jeu de données pour
#     l'entrainement, ajouter les données de test, et tracer l'ensemble sur le graphe.
#     ça permettrait de voir le sur-apprentissage, et voir comment il évolue avec le degré
#     du polynome par exemple.
#   - Une bonne idée pourrait de comparer les différents noyaux et les différentes valeurs
#     des paramètres de ces noyaux sur les mêmes jeux de données. Comme pour l'instant on
#     les régénère à chaque fois, c'est possible. Utiliser "pickle" pour tirer des jeux
#     de données, les sérialiser, les mettre dans des fichiers, puis les réutiliser ici.
