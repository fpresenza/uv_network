#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date mar 14 dic 2021 15:12:35 -03
"""
import numpy as np
import collections


__all__ = (
    'subgraph_protocol', )

Token = collections.namedtuple(
    'Token',
    'center, \
    timestamp, \
    hops_to_target, \
    hops_travelled, \
    data')

Token.__new__.__defaults__ = (
    None,
    None,
    None,
    np.inf,
    None)


class subgraph_protocol(object):
    def __init__(self, node_id, extent, queue=1):
        """Clase para implementar el ruteo necesario para el control
        de formaciones basado en subgrafos.

        El protocolo se basa en dos tipos de tokens: accion y estado.

        Cada nodo recibe un token de estado por cada nodo en su subgrafo,
        computa las acciones de control, las empaqueta en un token de accion
        y las reenvia al mismo.

        Los tokens de accion se propagan hasta que la cantidad de hops
        viajados es igual al extent del nodo emisor.

        Cada nodo envia sus tokens de estado a los nodos de los cuales
        recibio token de accion. Los tokens de estado son ruteados por los
        caminos por los cuales llegaron aquellos tokens de accion.
        """
        self.node_id = node_id
        self.action = collections.deque(maxlen=queue)
        self.action.append({})
        self.state = collections.deque(maxlen=queue)
        self.state.append({})

    def action_tokens(self):
        return tuple(self.action[0].values())

    def state_tokens(self):
        return tuple(self.state[0].values())

    def to_target(self, token):
        return token.hops_travelled < token.hops_to_target

    def extract_action(self):
        cmd = {
            token.center: token.data[self.node_id]
            for token in self.action[0].values()
            if self.node_id in token.data}
        return cmd

    def extract_state(self, key, hops):
        p = {
            token.center: token.data[key]
            for token in self.state[0].values()
            if token.hops_travelled <= hops}
        return p

    def geodesics(self, hops):
        g = {
            token.center: token.hops_travelled
            for token in self.state[0].values()
            if token.hops_travelled <= hops}
        return g

    def broadcast(self, timestamp, action, state, extent):
        """Prepara las listas con los tokens que se deben enviar.
        Luego elimina todos los tokens recibidos de otros nodos."""
        action_tokens = {
            token.center: token
            for token in self.action[0].values()
            if self.to_target(token)}

        action_tokens[self.node_id] = Token(
            center=self.node_id,
            timestamp=timestamp,
            hops_to_target=extent.copy(),
            hops_travelled=0,
            data=action.copy())

        state_tokens = {
            token.center: token
            for token in self.state[0].values()}

        state_tokens[self.node_id] = Token(
            center=self.node_id,
            timestamp=timestamp,
            hops_travelled=0,
            data=state.copy())

        self.action.append({})
        self.state.append({})

        return action_tokens, state_tokens

    def update_action(self, tokens):
        for token in tokens:
            """Actualiza la informacion de accion que es recibida"""
            if token.center != self.node_id:
                """Actualiza los hops atravesados"""
                token = token._replace(hops_travelled=token.hops_travelled + 1)
                center = token.center
                """Toma el token existente o lo reemplaza por el nuevo"""
                try:
                    """Se queda con el de menor hops atravesados"""
                    curr_token = self.action[-1][center]
                    if token.hops_travelled < curr_token.hops_travelled:
                        self.action[-1][center] = token
                except KeyError:
                    self.action[-1][center] = token

    def update_state(self, tokens):
        for token in tokens:
            """Actualiza la informacion de estado que es recibida"""
            if token.center != self.node_id:
                """Actualiza los hops atravesados"""
                token = token._replace(hops_travelled=token.hops_travelled + 1)
                center = token.center
                """Toma el token existente o lo reemplaza por el nuevo"""
                try:
                    """Se queda con el de menor hops atravesados"""
                    curr_token = self.state[-1][center]
                    if token.hops_travelled < curr_token.hops_travelled:
                        self.state[-1][center] = token
                except KeyError:
                    self.state[-1][center] = token
