#!/usr/bin/env python

import uvnpy.tools.graph as graph

print('random graph:')
G = graph.Random(V=[1,2,3,4,5])
print(G.E)
print(len(G.E))
print(G.W)

print('custom graph:')
G = graph.Custom(V=[1,2,3,4,5], E=[(1,3), (1,5), (2,4), (5,3)])
print(G.E)
print(len(G.E))
print(G.W)

print('custom graph 2:')
G = graph.Custom(V=[1,2,3,4,5], connect=lambda i,j: (i+j)%2==0)
print(G.E)
print(len(G.E))
print(G.W)
