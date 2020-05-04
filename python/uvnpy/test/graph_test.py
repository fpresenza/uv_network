#!/usr/bin/env python
import numpy as np
import uvnpy.network.graph as graph
from uvnpy.lib.ros import Vector3
from uvnpy.network.vehicles import Rover

G = graph.Graph()
G.add_nodes_from(map(lambda id: Rover(id), range(1,10)))
# 
print(G.nodes)
n, = G.get_nodes(5)
print(n)
# G.n(0).x = 7 
# print(G.n(0))