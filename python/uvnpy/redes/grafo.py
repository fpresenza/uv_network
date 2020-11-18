#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 06 12:41:07 2020
@author: fran
"""
import numpy as np
import scipy.linalg
from graph_tool import Graph

from uvnpy.toolkit import iterable
from uvnpy.filtering import consenso

__all__ = [
    'distancia',
    'distancia_jacobiano',
    'proximidad',
    'grafo'
]


def distancia(p, D, n=2):
    Dt = np.kron(D, np.eye(n)).T
    diff = Dt.dot(p).reshape(-1, n)
    sqrdiff = diff * diff
    return np.sqrt(sqrdiff.sum(1))


def distancia_jacobiano(p, D, n=2):
    Dt = np.kron(D, np.eye(n)).T
    diff = Dt.dot(p).reshape(-1, n)
    sqrdiff = diff * diff
    dist = np.sqrt(sqrdiff.sum(1))
    h = diff / dist.reshape(-1, 1)
    M = scipy.linalg.block_diag(*h)
    return M.dot(Dt)


def proximidad(v_i, v_j, rango_max):
    p_i, p_j = v_i.din.p, v_j.din.p
    dist = np.linalg.norm(p_i - p_j)
    return dist <= rango_max


class grafo(Graph):
    def __init__(self, *args, **kwargs):
        super(grafo, self).__init__(*args, **kwargs)
        self.vertex_properties['uvs'] = self.new_vertex_property('object')
        # self.edge_properties['links'] = self.new_edge_property('object')
        self._map = {}

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

    @property
    def vehiculos(self):
        return list(self.vp.uvs)

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

    @property
    def enlaces(self):
        uvs = self.vp.uvs
        return [[uvs[s].id, uvs[t].id] for (s, t) in self.get_edges()]

    def iniciar_dinamica(self, pi, vi={}, ti=0.):
        for v in self.vehiculos:
            v.din.iniciar(pi[v.id], vi=vi.get(v.id), ti=ti)

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

    def compartir(self, i):
        """Compartir mensaje con sus vecinos. """
        vehiculo = self.vehiculo(i)
        vecinos = self.vecinos(i)
        msg = vehiculo.box.salida
        return [v.box.recibir(msg) for v in vecinos]

    def intercambiar(self):
        """Intercambiar mensajes entre vecinos. """
        edges = self.get_edges()
        for s, t in edges:
            v_i = self.vp.uvs[s]
            v_j = self.vp.uvs[t]
            msg_i = v_i.box.salida
            msg_j = v_j.box.salida
            v_i.box.recibir(msg_j)
            v_j.box.recibir(msg_i)

    def iniciar_consenso_promedio(self, dic, ti=0.):
        for v in self.vehiculos:
            num = len(v.promedio)
            v.promedio.append(consenso.promedio(dic[v.id], ti))
            v.box.actualizar_salida(('avg', num), dic[v.id])

    def iniciar_consenso_lpf(self, dic, ti=0.):
        for v in self.vehiculos:
            v.lpf.iniciar(dic[v.id]['x'], ti)
            v.box.actualizar_salida('lpf', dic[v.id])

    def iniciar_consenso_comparador(self, dic, funcion):
        for v in self.vehiculos:
            v.comparador.iniciar(dic[v.id]['x'], dic[v.id]['u'], funcion)
            v.box.actualizar_salida('comparador', dic[v.id])
