#! /usr/bin/python
#-*- coding: utf-8 -*-

#Importation des librairies
from cvxopt.solvers import qp
from cvxopt.base import matrix

#Importation des fonctions necessaires
import numpy as np
import pylab, random, math
import sys
from generate_tests import *


# !!! ATTENTION !!!
# Numpy a des subtilites discutables entre matrice et array.
#   - Le type np.array ne contient que le nombre de dimensions necessaires. Autrement
#     dit, il n'y a aucune discinction entre un vecteur ligne et un vecteur colone: ce sont
#     deux arrays de dimension 1. Cela implique que le 'dot product' np.dot(x, y) sur
#     des np.array est TOUJOURS le produit scalaire, et ne peut PAS produire une grande
#     matrice, même si on essaie d'appeler transpose() sur l'un des arguments. L'avantage
#     est qu'on ne se fera pas crier dessus parce que les vecteurs ne sont pas dans le bon
#     sens. Le 'dot product' marche comme un produit de matrice dans le cas ou l'array est
#     de dimension 2. Je n'ai meme pas envie de penser a ce que ca fait en dimension > 2.
#     Sur les np.array, l'operateur '*' fait une multiplication coordonnee par coordonnee,
#
#   - Le type np.matrix, en revenche, represente une matrice au sens mathematique (sauf
#     que la numerotation commence a 0 et pas a 1 a la difference de matlab). Une np.matrix a
#     TOUJOURS deux dimensions. Le 'dot product' (np.dot(a, b)) ET l'operation '*' representent
#     la multiplication matricielle normale, par ex. le produit de 2 vecteurs produira
#     soit une grosse matrice soit le produit scalaire selon qu'on multiplie colone par ligne
#     ou ligne par colone. La transposition sur un vecteur represente par un type np.matrix
#     a un effet, contrairement aux arrays, la distinction ligne/colone a un sens.


USE_SLACK_VARIABLES = False
SLACK_CTE = 5 # Le 'C' à faire varier pour étudier l'effet des clack variables


# Fonction principale contenant la logique de l'algorithme d'apprentissage des SVM
# In: training_set  Points pour l'entrainement, sous forme de liste de triplets (x, y, classe)
# In: kernel        Kernel a utiliser pour la transformation des donnees
#
# Returns: indicator :: point -> classe
#                       Fonction pouvant servir a classifier un nouvel element:
def learn_indicator(training_data, kernel):
    nb_pts = len(training_data)
    # On defini les matrices qui serviront a trouver alpha (cf enonce TP).
    # Comme ce TP est cense nous faire utiliser numpy, faisons-en des arrays numpy
    P = def_P(training_data, kernel)
    q = -np.ones(nb_pts)

    # Si on utilise les slack variables, G et h sont un peu différentes:
    #   G = | - Id_n |
    #       | C*Id_n |
    #   h = | (0) |
    #       | (C) |
    if USE_SLACK_VARIABLES:
        # Cf la doc de numpy pour voir ce qu'est la liste r_. En gros c'est un moyen
        # malin de concaténer des lignes. Il existe c_ pour concaténer des colones.
        # Comme on utilise des np.array, la concaténation de colones correspond à
        # concaténer "les listes les plus extérieures" en ligne, donc r_ marche.
        h = np.r_[np.zeros(nb_pts), SLACK_CTE * np.ones(nb_pts)]
        G = np.r_[-np.eye(nb_pts), np.eye(nb_pts)]
    else:
        h = np.zeros(nb_pts)
        G = -np.eye(nb_pts)

    # Trouver les alphas solution du probleme d'optimisation
    alphas = find_alphas(P, q, G, h)
    ts = np.array([ech[2] for ech in training_data]) # Classe des différents points (t_i dans l'énoncé)
    ys = [ np.array([e[0], e[1]]) for e in training_data ] # points du training set sans leur classe

    # Notre fonction indicator a retourner, dependant de alphas et training_data
    # !! Attention, pour faire ça il faut être sûr que alphas, ts, ys ne seront pas modifiés
    # ultérieurement (c'est OK ici). S'ils étaient modifiés, indicator le serait par
    # effet de bord, car Pyton ne gère pas les fermetures.
    def indicator(x):
        kernel_values = np.array([kernel(x, y) for y in ys])
        # Sur des arrays, '*' est la multiplication coordonnee par coordonnee
        return sum(alphas * ts * kernel_values)

    return indicator


# Calcule la matrice P tq P_i,j = t_i * t_j * K(x_i, x_j) pour x_i, x_j se baladant
# dans les données d'entrainement
def def_P(training_data, kernel):
    nb_pts = len(training_data)
    P = np.zeros((nb_pts, nb_pts))
    for i, d_i in enumerate(training_data):
        for j, d_j in enumerate(training_data):
            t_i, t_j = d_i[2], d_j[2]

            # Kernel prend deux matrices numpy en entree
            x_i = np.array([d_i[0], d_i[1]])
            x_j = np.array([d_j[0], d_j[1]])

            P[i, j] = t_i * t_j * kernel(x_i, x_j)

    print("Rang de P: %d" % sum([1 for e in np.linalg.eigvals(P) if math.fabs(e) > 10e-5]))
    return P



# Utilise cvxopt pour resoudre le probleme (dual) d'optimisation convexe
# Ne pas oublier qu'on fait du calcul numerique, donc un peu de chimie,
# donc tout nombre < 10e-5 sera considere comme etant nul.
def find_alphas(P, q, G, h):
    # Utilisation de cvxopt: cf enonce du TP
    r = qp(matrix(P), matrix(q), matrix(G), matrix(h))
    alphas = np.array(list(r['x']))
    # remplacer les valeurs trop petites par 0, garder le reste
    alphas = np.where(np.fabs(alphas) < 10e-5, 0, alphas)

    return alphas



################### Partie concernant le test #####################



def plot_boundary (classA, classB, indicator):
        pylab.hold(True)        
        pylab.plot([p[0] for p in classA], [p[1] for p in classA], 'bo')
        pylab.plot([p[0] for p in classB], [p[1] for p in classB],'ro')
       
        xrange = np.arange(-4, 4, 0.05)
        yrange = np.arange(-4, 4, 0.05)

        grid = matrix([[indicator(np.array([x, y])) for y in yrange] for x in xrange])

        pylab.contour(xrange, yrange, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))
        pylab.show()



def try_indicator(kernel, dataset_filename):
    # Récupérer les données dans un fichier
    classA, classB, data = load_data(dataset_filename)

    indicator = learn_indicator(data, kernel)
    plot_boundary(classA, classB, indicator)


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
        return np.dot(a, b) + 1
    return kernel

def polynomial_kernel(p):
    def kernel(a, b):
        return (np.dot(a, b) + 1) ** p
    return kernel

def radial_basis_kernel(sigma):
    def kernel(a, b):
        delta = a - b
        return math.exp( -np.dot(delta, delta) / (2 * sigma**2) )
    return kernel

def sigmoid_kernel(k, delta):
    def kernel(a, b):
        return math.tanh( k*np.dot(a, b) - delta )
    return kernel



if __name__ == "__main__" :
    # Dégueu, d'ailleurs il crie, mais marche quand même, donc tant pis
    global USE_SLACK_VARIABLES
    global SLACK_CTE
    if len(sys.argv) < 2:
        print("Will generate a new random dataset in /tmp/dataset.dat")
        dataset_filename = "/tmp/dataset.dat"
        generate_data(dataset_filename, allow_overwrite=True)
    else: 
        dataset_filename = sys.argv[1]

        if len(sys.argv) > 2: # the second argument is the value of SLACK_CTE
            USE_SLACK_VARIABLES = True
            SLACK_CTE = float(sys.argv[2])

    #try_indicator(polynomial_kernel(3), dataset_filename)
    try_indicator(radial_basis_kernel(0.5), dataset_filename)
    #try_indicator(sigmoid_kernel(100, 0.01), dataset_filename)
    #try_indicator(linear_kernel(), dataset_filename)
