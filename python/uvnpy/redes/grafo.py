#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 06 12:41:07 2020
@author: fran
"""
from graph_tool import Graph

from uvnpy.toolkit import iterable

__all__ = [
    'grafo'
]


class grafo(Graph):
    def __init__(self, *args, **kwargs):
        super(grafo, self).__init__(*args, **kwargs)
        self.vertex_properties['uvs'] = self.new_vertex_property('object')
        self._map = {}

    @property
    def vehiculos(self):
        return list(self.vp.uvs)

    @property
    def enlaces(self):
        uvs = self.vp.uvs
        return [[uvs[s].id, uvs[t].id] for (s, t) in self.get_edges()]

    def __actualizar_mapa(self):
        """ Actualiza el mapa de ids """
        indices = self.get_vertices()
        ids = [self.vp.uvs[idx].id for idx in indices]
        self._map = dict(zip(ids, indices))

    def agregar_vehiculos(self, uvs):
        """Agrega vehiculos como propiedad de los vértices. """
        N = len(uvs)
        nuevos_vertices = self.add_vertex(N)
        if not iterable(nuevos_vertices):
            nuevos_vertices = [nuevos_vertices]  # por si agrega un sólo uv
        for v, uv in zip(nuevos_vertices, uvs):
            self.vp.uvs[v] = uv
        self.__actualizar_mapa()
        return nuevos_vertices

    def remover_vehiculos(self, ids):
        """Elimina una lista de vehículos. """
        idx = [self._map[i] for i in ids]
        self.remove_vertex(idx)
        self.__actualizar_mapa()

    def vehiculo(self, i):
        return self.vp.uvs[self._map[i]]

    def agregar_enlace(self, i, j):
        """ Agregar un enlace entre dos vehículos. """
        s, t = self._map[i], self._map[j]
        edges = self.get_edges().tolist()
        if [s, t] not in edges and [t, s] not in edges:
            nuevo_enlace = self.add_edge(s, t, add_missing=False)
            return nuevo_enlace

    def remover_enlace(self, i, j):
        """ Elimina un enlace entre dos vehículos. """
        s, t = self._map[i], self._map[j]
        enlace = self.edge(s, t)
        if enlace is not None:
            self.remove_edge(enlace)

    def reconectar(self, condicion, *args):
        vehiculos = self.vehiculos
        for v_i in vehiculos:
            resto = vehiculos.copy()
            resto.remove(v_i)
            id_i = v_i.id
            for v_j in resto:
                id_j = v_j.id
                if condicion(v_i, v_j, *args):
                    self.agregar_enlace(id_i, id_j)
                else:
                    self.remover_enlace(id_i, id_j)

    def vecinos(self, i):
        vertex = self.vertex(self._map[i])
        neighbors = vertex.all_neighbors()
        return [self.vp.uvs[n] for n in neighbors]

    def intercambiar(self, swap, *args):
        """Intercambio entre vecinos. """
        edges = self.get_edges()
        for s, t in edges:
            v_i = self.vp.uvs[s]
            v_j = self.vp.uvs[t]
            swap(v_i, v_j, *args)

    def compartir(self, i, share, *args):
        """Compartir con vecinos. """
        vehiculo = self.vehiculo(i)
        vecinos = self.vecinos(i)
        share(vehiculo, vecinos, *args)
        # msg = vehiculo.box.salida
        # return [v.box.recibir(msg) for v in vecinos]
