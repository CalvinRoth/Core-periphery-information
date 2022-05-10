from __future__ import annotations

import networkx as nx
import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt
from  rational import RationalF as Ratio
import itertools


def centralty(A: np.matrix, rho: float) -> np.matrix:
    """

    Parameters
    ----------
    A : np matrix
    rho : network effect

    Returns
    -------
    Centrality vector as described in paper
    """
    n = A.shape[0]
    ident = np.eye(n, n)
    ones = np.ones((n, 1))
    ApA = A + A.T
    central = lin.inv(ident - (rho * ApA))
    central = central @ ones  
    return central

def price_vector(a, c, rho, G):
    n = len(G)
    frac1 = (a+c)/2
    frac2 = rho * ( (a-c)/2)
    return (frac1 * np.ones((n,1))) + (frac2 * (G - G.T) @ centralty(G, rho))
 

def consumption(n, rho, a, c, G):
    p = price_vector(a, c, rho, G)
    v = a * np.ones((n,1))
    v = v - p 
    mat = lin.inv(np.eye(n,n) - 2 * rho * G  )
    return 0.5 * mat @ v 

def optProfit(G, discount, a, c):
    n = len(G)
    rho = discount / specNorm(G + G.T)
    price = price_vector(a, c, discount, G)
    consu = lin.inv(np.eye(n,n) - 2*rho*G)
    consu = 0.5 * consu @ (a * np.ones((n,1)) - price)
    profit = (price - c*np.ones((n,1))).T @ consu
    return profit[0,0]
 
def computeProfit(G, v, discount, a, c):
    n = len(G)
    rho = discount / np.lin.norm(G + G.T, ord=2)
    price = v
    consu = lin.inv(np.eye(n,n) - 2*rho*G)
    consu = 0.5 * consu @ (a * np.ones((n,1)) - price)
    profit = (price - c*np.ones((n,1))).T @ consu
    return profit[0,0]

# get max_G Profit_G* - Profit_G(v)
def worst_gap(graphs, opts, v, rho, a, c):
    profits = [computeProfit(G, v, rho, a,c) for G in graphs]
    worst_score = -np.inf 
    worst_idx = -1
    for i in range(len(graphs)):
        s = opts[i] - profits[i]
        if (s < 0):
            print("???????")
        if(s > worst_score):
            worst_score = s 
            worst_idx = i 
    return worst_score


# def aveverge_G Profit-G* - Profit_G(v)
def average_gap(graphs, opts, v, rho, a, c):
    profits = [computeProfit(G, v, rho, a,c) for G in graphs]
    total = 0 
    for i in range(len(graphs)):
        s = opts[i] - profits[i] 
        total += opts[i] - profits[i]

    return total / len(graphs)

## From itertools, but with modified initial args 
def product(args, repeat=1):
    # product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
    # product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
    pools = [tuple(pool) for pool in args] * repeat
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    for prod in result:
        yield tuple(prod)

def get_mesh(coords : np.arrary, steps : int) ->  itertools.product :
    ### 
    # coords: nx2 array where each row is the start and end index for that dimension 
    # steps : number of steps to take in each dimension. Total size of mesh is steps ^ dimension 
    # Returns a generator to all the coordinate space in this domain
    one_d_spaces = [np.linspace(i[0], i[1], steps) for i in coords]
    return product(one_d_spaces)

