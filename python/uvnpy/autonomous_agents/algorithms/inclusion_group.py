#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date mar 14 dic 2021 15:12:35 -03
"""
import numpy as np
import collections


Token = collections.namedtuple(
    'Token',
    'center, \
    timestamp, \
    extent, \
    geodesic, \
    action')

Token.__new__.__defaults__ = (
    None,
    None,
    None,
    np.inf,
    None)


class InclusionGroup(object):
    def __init__(self, node_id, node_extent, t=0.):
        """Clase para guardar info del grupo de inclusion de un nodo

        El grupo de inclusion del nodo "i" es el conjunto de nodos "j"
        tales que el nodo "i" pertenece a los grupos Gj.
        """
        self.id = node_id
        self._ig = {
            node_id: Token(
                center=node_id,
                timestamp=t,
                extent=node_extent,
                geodesic=0,
                action={})}

    def __getitem__(self, center):
        return self._ig[center]

    def __call__(self):
        """Devuelve un tuple con los ids del grupo de inclusion"""
        m = tuple(token.center for token in self._ig.values())
        return m

    def tokens(self):
        return self._ig.values()

    def extents(self):
        """Devuelve un tuple con los extents del grupo de inclusion"""
        h = tuple(token.extent for token in self._ig.values())
        return h

    def geodesics(self):
        """Devuelve un tuple con los extents del grupo de inclusion"""
        h = tuple(token.geodesic for token in self._ig.values())
        return h

    def broadcast(self, t, u):
        """Envia a sus vecinos info de los nodos "j" en el grupo de inclusion
        si el nodo "i" no es un nodo en la ultima capa de Gj"""
        self._ig[self.id] = self._ig[self.id]._replace(
            timestamp=t, action=u)
        tokens = [tk for tk in self._ig.values() if tk.geodesic < tk.extent]
        return tokens

    def update(self, token):
        """Actualiza la informacion de los nodos del grupo de inclusion al
        recibir un token"""
        token = token._replace(geodesic=token.geodesic + 1)
        self._ig[token.center] = self._ig.get(token.center, token)
        if token.geodesic < self._ig[token.center].geodesic:
            self._ig[token.center] = token

    def clear(self):
        self._ig = {self.id: self._ig[self.id]}
