#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date dom nov  1 21:04:58 -03 2020
"""
import collections

__all__ = [
    'box'
]


class box(object):
    def __init__(self, out={}, maxlen=10):
        """ Clase para recibir, guardar y publicar mensajes. """
        self._in = collections.deque(maxlen=maxlen)
        self._out = out.copy()

    @property
    def entrada(self):
        return self._in

    @property
    def salida(self):
        return self._out.copy()

    def recibir(self, msg):
        self._in.append(msg)
        return 1

    def actualizar(self, key, value):
        self._out[key] = value

    def limpiar(self):
        self._in.clear()

    def limpiar_salida(self):
        self._out.clear()

    def extraer(self, *keys):
        caja = self._in
        for key in keys:
            caja = [msg[key] for msg in caja]
        return caja
