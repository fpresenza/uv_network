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
    hops_to_target, \
    hops_travelled, \
    data')

Token.__new__.__defaults__ = (
    None,
    None,
    None,
    np.inf,
    None)


class TokenPassing(object):
    """
    Clase para implementar el ruteo necesario para el control
    de formaciones basado en subgrafos.

    El protocolo se basa en dos tipos de tokens: accion y estado.

    Los tokens de accion se propagan hasta que la cantidad de hops
    viajados es igual al action_extent del nodo emisor.
    Los tokens de estado se propagan hasta que la cantidad de hops
    viajados es igual al state_extent del nodo emisor.
    """
    def __init__(self, node_id):
        self.node_id = node_id
        self.action = {}    # guarda los tokens de accion recibidos (fifo)
        self.state = {}    # guarda los tokens de estado recibidos (fifo)

    def action_tokens(self):
        # Extrae los tokens de accion de la queue
        return tuple(self.action.values())

    def state_tokens(self):
        # Extrae los tokens de estado de la queue
        return tuple(self.state.values())

    def extract_action(self):
        """ Extrae los datos encontrados en los token de accion que tienen
        info para este nodo en el campo data"""
        cmd = {
            token.center: token.data[self.node_id]
            for token in self.action.values()
            if self.node_id in token.data
        }
        return cmd

    def extract_state(self, key, hops):
        """ Extrae los datos asociados a key encontrados en los token de
        estado que hayan atravesado una cantidad de enlaces menor o igual
        a hops"""
        p = {
            token.center: token.data[key]
            for token in self.state.values()
            if token.hops_travelled <= hops
        }
        return p

    def geodesics(self, hops):
        """Calcula las distancias geodesicas de otros nodos en funcion de los
        hops atravesados por sus mensajes"""
        g = {
            token.center: token.hops_travelled
            for token in self.state.values()
            if token.hops_travelled <= hops
        }
        return g

    def broadcast(self, timestamp, action, state, action_extent, state_extent):
        """Prepara las listas con los tokens que se deben enviar.
        Luego elimina todos los tokens recibidos de otros nodos"""
        # Tokens de accion de otros nodos para retransmitir
        action_tokens = {
            token.center: token
            for token in self.action.values()
            if token.hops_travelled < token.hops_to_target}

        # Token de accion propio para transmitir
        if action_extent > 0:
            action_tokens[self.node_id] = Token(
                center=self.node_id,
                timestamp=timestamp,
                hops_to_target=action_extent,
                hops_travelled=0,
                data=action
            )

        # Tokens de estado de otros nodos para retransmitir
        state_tokens = {
            token.center: token
            for token in self.state.values()
            if token.hops_travelled < token.hops_to_target}

        # Token de estado propio para transmitir
        if state_extent > 0:
            state_tokens[self.node_id] = Token(
                center=self.node_id,
                timestamp=timestamp,
                hops_to_target=state_extent,
                hops_travelled=0,
                data=state
            )

        self.action.clear()
        self.state.clear()

        return action_tokens, state_tokens

    def update_action(self, tokens):
        """Actualiza la informacion de accion que es recibida en
        forma de tokens"""
        for token in tokens:
            if token.center != self.node_id:
                """Actualiza los hops atravesados"""
                token = token._replace(hops_travelled=token.hops_travelled + 1)
                center = token.center
                """Toma el token existente o lo reemplaza por el nuevo"""
                try:
                    """Se queda con el de menor hops atravesados"""
                    curr_token = self.action[center]
                    if token.hops_travelled < curr_token.hops_travelled:
                        self.action[center] = token
                except KeyError:
                    self.action[center] = token

    def update_state(self, tokens):
        """Actualiza la informacion de estado que es recibida en
        forma de tokens"""
        for token in tokens:
            if token.center != self.node_id:
                """Actualiza los hops atravesados"""
                token = token._replace(hops_travelled=token.hops_travelled + 1)
                center = token.center
                """Toma el token existente o lo reemplaza por el nuevo"""
                try:
                    """Se queda con el de menor hops atravesados"""
                    curr_token = self.state[center]
                    if token.hops_travelled < curr_token.hops_travelled:
                        self.state[center] = token
                except KeyError:
                    self.state[center] = token
