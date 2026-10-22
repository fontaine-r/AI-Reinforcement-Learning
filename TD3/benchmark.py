import timeit
from math import sqrt
import matplotlib.pyplot as plt # type: ignore

from game import (
    create_board
)

import algorithms

def benchmark(board, algorithm, *args):
    startTime = timeit.default_timer()
    result = algorithm(board, args[0])
    endTime = timeit.default_timer()

    dictionnary = dict()
    dictionnary["name"] = algorithm.__name__
    dictionnary["move"] = result.get("move")
    dictionnary["score"] = result.get("score")
    dictionnary["numberExplored"] = result.get("nodes_explored")
    dictionnary["runtime"] = endTime - startTime    
    return dictionnary


def printJoli(dict):
    for key in dict:
        print(str(key)+" : " + str(dict[key]))
    print("\n")

def compareMAndAB():
    for i in range(1,6):
        dictMinimax = benchmark(create_board(), algorithms.minimax, i)
        dictAlphabeta = benchmark(create_board(), algorithms.alphabeta, i)
        printJoli(dictMinimax)
        printJoli(dictAlphabeta)
        if (dictMinimax["score"] != dictAlphabeta["score"]):
            raise ValueError("AlphaBeta et Minimax n'ont pas le même score")
        if (dictMinimax["move"] != dictAlphabeta["move"]):
            raise ValueError("AlphaBeta et Minimax ne jouent pas le même coup")
        if (dictMinimax["numberExplored"] < dictAlphabeta["numberExplored"]):
            raise ValueError("Minimax explore moins que AlphaBeta")


# printJoli(benchmark(create_board(), algorithms.mcts, 1000))
def compareMCTS():
    for N in [100, 500, 1000, 2000]:
        dictMCTS = benchmark(create_board(), algorithms.mcts, N, sqrt(2), 5679)
        printJoli(dictMCTS)


compareMCTS()

def createGraphs():
    miniMaxX = []
    alphaBetaX = []
    minimaxY = []
    for i in range(1,6):
        dictMinimax = benchmark(create_board(), algorithms.minimax, i)
        dictAlphabeta = benchmark(create_board(), algorithms.alphabeta, i)
        miniMaxX.append(dictMinimax["numberExplored"])
        alphaBetaX.append(dictAlphabeta["numberExplored"])
        minimaxY.append(i)

    plt.figure(miniMaxX, minimaxY)
    plt.figure(alphaBetaX, minimaxY)
    plt.show()
