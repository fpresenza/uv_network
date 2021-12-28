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
    path, \
    data')

Token.__new__.__defaults__ = (
    None,
    None,
    None,
    np.inf,
    None,
    None)


class subgraph_protocol(object):
    def __init__(self, node_id, extent, t=0.):
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
        self.action = {}
        self.state = {}

    def action_tokens(self):
        return tuple(self.action.values())

    def state_tokens(self):
        return tuple(self.state.values())

    def to_target(self, token):
        return token.hops_travelled < token.hops_to_target

    def in_path(self, token):
        return np.any([self.node_id in p[1:-1] for p in token.path])

    def extract_action(self):
        cmd = {
            token.center: token.data[self.node_id]
            for token in self.action.values()
            if self.node_id in token.data}
        return cmd

    def extract_state(self, key, hops):
        p = {
            token.center: token.data[key]
            for token in self.state.values()
            if token.hops_travelled <= hops}
        return p

    def geodesics(self, hops):
        g = {
            token.center: token.hops_travelled
            for token in self.state.values()
            if token.hops_travelled <= hops}
        return g

    def broadcast(self, timestamp, action, state, extent):
        """Prepara las listas con los tokens que se deben enviar.
        Luego elimina todos los tokens recibidos de otros nodos."""
        action_tokens = {
            token.center: token
            for token in self.action.values()
            if self.to_target(token)}

        action_tokens[self.node_id] = Token(
            center=self.node_id,
            timestamp=timestamp,
            hops_to_target=extent,
            hops_travelled=0,
            path=[self.node_id],
            data=action)

        state_tokens = {
            token.center: token
            for token in self.state.values()
            if self.in_path(token)}

        state_tokens[self.node_id] = Token(
            center=self.node_id,
            timestamp=timestamp,
            hops_travelled=0,
            path=[token.path for token in self.action_tokens()],
            data=state)

        self.action.clear()
        self.state.clear()

        return action_tokens, state_tokens

    def update_action(self, token):
        """Actualiza la informacion de accion que es recibida"""
        if token.center != self.node_id:
            token = token._replace(
                hops_travelled=token.hops_travelled + 1,
                path=token.path + [self.node_id])
            self.action[token.center] = self.action.get(token.center, token)
            if token.hops_travelled < self.action[token.center].hops_travelled:
                self.action[token.center] = token

    def update_state(self, token):
        """Actualiza la informacion de estado que es recibida"""
        if token.center != self.node_id:
            token = token._replace(
                hops_travelled=token.hops_travelled + 1)
            self.state[token.center] = self.state.get(token.center, token)
            if token.hops_travelled < self.state[token.center].hops_travelled:
                self.state[token.center] = token
