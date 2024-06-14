#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
"""
import numpy as np
import collections


Token = collections.namedtuple(
    'Token',
    'transmitter, \
    counter, \
    hops_to_target, \
    hops_travelled, \
    data'
)

Token.__new__.__defaults__ = (
    None,
    None,
    None,
    np.inf,
    None
)


class TokenPassing(object):
    """
    Clase para implementar el ruteo necesario para el intercambio
    de informacion entre agentes de una red.

    El protocolo se basa en el pase de tokens. Estos se propagan
    hasta que la cantidad de hops viajados es igual a la indicada.
    """
    def __init__(self, node_id):
        self.node_id = node_id
        self.counter = 0
        self.token_record = {}    # guarda los tokens recibidos (fifo)

    def extract_data(self, hops=np.inf):
        """ Extrae los datos en los token registrados"""
        return {
            token.transmitter: token.data
            for token in self.token_record.values()
            if token.hops_travelled <= hops
        }

    def geodesics(self, hops=np.inf):
        """Calcula las distancias geodesicas de los nodos en funcion de los
        hops atravesados por sus mensajes"""
        return {
            token.transmitter: token.hops_travelled
            for token in self.token_record.values()
            if token.hops_travelled <= hops
        }

    def tokens_to_transmit(self, data, extent):
        """Prepara una lista con los tokens que se deben enviar.
        Luego elimina todos los tokens recibidos de otros nodos"""
        # Tokens de otros nodos para retransmitir
        to_transmit = {
            token.transmitter: token
            for token in self.token_record.values()
            if token.hops_travelled < token.hops_to_target
        }

        # Token propio para transmitir
        to_transmit[self.node_id] = Token(
            transmitter=self.node_id,
            counter=self.counter,
            hops_to_target=extent,
            hops_travelled=0,
            data=data.copy()
        )

        self.counter += 1
        self.token_record.clear()

        return to_transmit.values()

    def update_record(self, tokens):
        """Actualiza la informacion que es recibida en forma de tokens"""
        for token in tokens:
            if token.transmitter != self.node_id:
                # Actualiza los hops atravesados
                token = token._replace(hops_travelled=token.hops_travelled + 1)
                transmitter = token.transmitter
                # Si ya cuenta con un el token del transmisor, solo lo
                # actualiza si el nuevo tiene menor recorrido
                try:
                    curr_token = self.token_record[transmitter]
                    if token.hops_travelled < curr_token.hops_travelled:
                        self.token_record[transmitter] = token
                except KeyError:
                    self.token_record[transmitter] = token
