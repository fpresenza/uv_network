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


def radial_broadcast(hops_travelled, hops_to_target):
    return hops_travelled < hops_to_target


class subframework_rigidity(object):
    def __init__(self, node_id, node_extent, t=0.):
        """Clase para guardar info del grupo de inclusion de un nodo

        El grupo de inclusion del nodo "i" es el conjunto de nodos "j"
        tales que el nodo "i" pertenece a los grupos Gj.
        """
        self.id = node_id
        self.action = {}
        self.action[node_id] = Token(
            center=node_id,
            timestamp=t,
            hops_to_target=node_extent,
            hops_travelled=0,
            path=[self.id],
            data={})
        self.state = {}
        self.state[node_id] = Token(
            center=node_id,
            timestamp=t,
            hops_to_target=1,
            hops_travelled=0,
            path=None,
            data=State())

    def action_tokens(self):
        return tuple(self.action.values())

    def action_members(self):
        """Devuelve un tuple con los extents del grupo de accion"""
        m = tuple(token.center for token in self.action.values())
        return m

    def action_rebroadcast(self, token):
        return token.hops_travelled < token.hops_to_target

    def action_geodesics(self):
        """Devuelve un tuple con los extents del grupo de inclusion"""
        g = tuple(token.hops_travelled for token in self.action.values())
        return g

    def state_rebroadcast(self, token):
        center = self.id == token.center
        in_path = np.any([(len(p) > 1) and (self.id in p) for p in token.path])
        hops_left = token.hops_travelled < token.hops_to_target
        broadcast = center or in_path and hops_left
        return broadcast

    def state_tokens(self):
        return tuple(self.state.values())

    def state_members(self):
        """Devuelve un tuple con los extents del grupo de accion"""
        m = tuple(token.center for token in self.state.values())
        return m

    def broadcast(self, timestamp, action, position, covariance):
        """Envia a sus vecinos info de los nodos "j" en el grupo de inclusion
        si el nodo "i" no es un nodo en la ultima capa de Gj"""
        self.action[self.id] = self.action[self.id]._replace(
            timestamp=timestamp,
            data=action)
        self.state[self.id] = self.state[self.id]._replace(
            timestamp=timestamp,
            hops_to_target=max(1, max(self.action_geodesics())),
            path=[
                tkn.path[:-1] for tkn in self.action_tokens() if
                len(tkn.path) > 1],
            data=State(position, covariance))
        action_tokens = [
            tkn for tkn in self.action.values() if
            self.action_rebroadcast(tkn)]
        state_tokens = [
            tkn for tkn in self.state.values() if self.state_rebroadcast(tkn)]
        return action_tokens, state_tokens

    def update_action(self, token):
        """Actualiza la informacion de los nodos del grupo de inclusion al
        recibir un token"""
        token = token._replace(
            hops_travelled=token.hops_travelled + 1,
            path=token.path + [self.id])
        self.action[token.center] = self.action.get(token.center, token)
        if token.hops_travelled < self.action[token.center].hops_travelled:
            self.action[token.center] = token

    def update_state(self, token):
        """Actualiza la informacion de los nodos del grupo de inclusion al
        recibir un token"""
        token = token._replace(
            hops_travelled=token.hops_travelled + 1)
        self.state[token.center] = self.state.get(token.center, token)
        if token.hops_travelled < self.state[token.center].hops_travelled:
            self.state[token.center] = token

    def restart(self):
        self.action = {self.id: self.action[self.id]}
        self.state = {self.id: self.state[self.id]}
