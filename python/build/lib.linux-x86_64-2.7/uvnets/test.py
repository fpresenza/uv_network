#!/usr/bin/env python

import uvnets.tools.graph as graph

G = graph.Random(V=[1,2,3,4,5])

print(G.E)
print(len(G.E))
print(G.W)