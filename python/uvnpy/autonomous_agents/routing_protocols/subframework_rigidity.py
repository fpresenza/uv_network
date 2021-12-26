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
    'subframework_rigidity', )

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

State = collections.namedtuple(
    'State',
    'position, \
    covariance')

State.__new__.__defaults__ = (
    None,
    None)


class subframework_rigidity(object):
    def __init__(self, node_id, node_extent, t=0.):
        """Clase para implementar el ruteo necesario para el control
        de rigidez basado en subframeworks.

        Cada nodo envia a sus vecinos un token de accion y uno de posicion con
        informacion propia.
        El primero contiene los comandos para los nodos en el subframework
        propio, y el segundo contiene el estado actual del nodo en cuestion
        para aquellos nodos que lo incluyan en su subframework.

        Los tokens de accion que llegan a cada nodo son reenviados solo si la
        cantidad de hops que viajaron son menores al extent del nodo emisor.

        Los tokens de posicion que llegan a cada nodo son reenviados solo si
        el nodo esta en un camino que une dos nodos que desean comunicarse.
        """
        self.id = node_id
        self.extent = node_extent
        self.action = {}
        self.state = {}

    def action_tokens(self):
        return tuple(self.action.values())

    def state_tokens(self):
        return tuple(self.state.values())

    def to_target(self, token):
        return token.hops_travelled < token.hops_to_target

    def in_path(self, token):
        return np.any([self.id in p[1:-1] for p in token.path])

    def commands(self):
        cmd = {
            token.center: token.data[self.id]
            for token in self.action.values()
            if self.id in token.data}
        return cmd

    def positions(self):
        p = {
            token.center: token.data.position
            for token in self.state.values()
            if token.hops_travelled <= self.extent}
        return p

    def broadcast(self, timestamp, action, position, covariance):
        """Prepara las listas con los tokens que se deben enviar.
        Luego elimina todos los tokens recibidos de otros nodos."""
        action_tokens = {
            token.center: token
            for token in self.action.values()
            if self.to_target(token)}

        action_tokens[self.id] = Token(
            center=self.id,
            timestamp=timestamp,
            hops_to_target=self.extent,
            hops_travelled=0,
            path=[self.id],
            data=action)

        state_tokens = {
            token.center: token
            for token in self.state.values()
            if self.in_path(token)}

        state_tokens[self.id] = Token(
            center=self.id,
            timestamp=timestamp,
            hops_travelled=0,
            path=[token.path for token in self.action_tokens()],
            data=State(position, covariance))

        self.action.clear()
        self.state.clear()

        return action_tokens, state_tokens

    def update_action(self, token):
        """Actualiza la informacion de accion que es recibida"""
        if token.center != self.id:
            token = token._replace(
                hops_travelled=token.hops_travelled + 1,
                path=token.path + [self.id])
            self.action[token.center] = self.action.get(token.center, token)
            if token.hops_travelled < self.action[token.center].hops_travelled:
                self.action[token.center] = token

    def update_state(self, token):
        """Actualiza la informacion de estado que es recibida"""
        if token.center != self.id:
            token = token._replace(
                hops_travelled=token.hops_travelled + 1)
            self.state[token.center] = self.state.get(token.center, token)
            if token.hops_travelled < self.state[token.center].hops_travelled:
                self.state[token.center] = token
