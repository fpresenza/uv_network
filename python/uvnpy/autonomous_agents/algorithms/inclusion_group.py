#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date mar 14 dic 2021 15:12:35 -03
"""
import numpy as np
import collections


TokenData = collections.namedtuple(
    'TokenData',
    'center, \
    timestamp, \
    extent, \
    geodesic')

TokenData.__new__.__defaults__ = (
    None,
    None,
    None,
    np.inf)


class InclusionGroup(object):
    def __init__(self, node_id, node_extent, t=0.):
        """Clase para guardar info del grupo de inclusion de un nodo

        El grupo de inclusion del nodo "i" es el conjunto de nodos "j"
        tales que el nodo "i" pertenece a los grupos Gj.
        """
        self._ig = {
            node_id: TokenData(
                center=node_id,
                timestamp=t,
                extent=node_extent,
                geodesic=0)}

    def __getitem__(self, center):
        return self._ig[center]

    def tokens(self):
        return self._ig.values()

    def members(self):
        """Devuelve un tuple con los ids del grupo"""
        m = tuple(token.center for token in self._ig.values())
        return m

    def broadcast(self):
        """Envia a sus vecinos info de los nodos "j" en el grupo de inclusion
        si el nodo "i" no es un nodo en la ultima capa de Gj"""
        tokens = [tk for tk in self._ig.values() if tk.geodesic < tk.extent]
        return tokens

    def update(self, token):
        """Actualiza la informacion de los nodos del grupo de inclusion al
        recibir un token"""
        center = token.center
        self._ig[center] = self._ig.get(center, TokenData(center=center))
        self._ig[center] = self._ig[center]._replace(
            timestamp=token.timestamp,
            extent=token.extent,
            geodesic=min(self._ig[center].geodesic, token.geodesic + 1))
