#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  17 15:08:05 2018

@author: pato

Modified by fran on Mon Dec 7 15:58:34 2019
"""

import numpy as np

class PIDController(object):
    """Implementación de un controlador PID

    Uso:
        pid = PIDController(kp=(0.2, 0.2, 0.06),
                            ki=(0.1, 0.1, 0.03),
                            kd=(0.0, 0.0, 0.00),
                            isats=[(-5., 5.), (-5., 5.), (-1., 1.)],
                            econd=lambda e: angular_wrap(e, (0, 0, 1)))
        u, _, _ = pid.update(ref, ym, t)

    TODO: resetear integrador al disminuir el error
    """
    def __init__(self, kp, ki=0, kd=0, t0=0, usats=None, isats=None, econd=None):
        """Inicializador del controlador. kp, ki y kd son arreglos con
        las ganacias proporcionales, integrales y derivativas,
        respectivamente. usats son los valores en los que se desea
        saturar la señal de control. isats son los valores de saturación
        de las integrales"""
        self.kp = np.array(kp)  # ganancias proporcionales
        self.ki = np.array(ki)  # ganancias integrales
        self.kd = np.array(kd)  # ganancias derivativas
        self.usats = np.array(usats) # saturación de la acción de ctrl
        self.isats = np.array(isats) # saturación de la integral
        self.t = t0 # tiempo anterior (para las derivadas)
        self.e = np.zeros_like(kp) # error anterior (para las derivadas)
        self.i = np.zeros_like(ki)  # integral (error acumulado)
        self.warp = econd # Función para preprocesar el error obtenido

    def update(self, ref, ym, t, debug=False):
        """Entrega una nueva señal de control en base a la referencia
        ref, la medición ym y los valores almacenados en el
        controlador"""

        # Error
        e = np.subtract(ref, ym)
        try:
            e = self.warp(e)
        except TypeError:
            pass

        # Derivada del error
        dt = np.subtract(t, self.t)
        # de = np.divide(np.subtract(e, self.e), dt)
        de = 0
        # Integral del error, saturada.
        self.i = np.add(self.i, np.multiply(dt, e))
        try:
            self.i = np.array([max(L[0], min(v, L[1])) for v, L in zip(self.i, self.isats)])
        except TypeError:
            pass

        p = np.multiply(self.kp, e)
        i = np.multiply(self.ki, self.i)
        d = np.multiply(self.kd, de)

        u = p + i + d
        try:
            u = np.array([max(L[0], min(v, L[1])) for v, L in zip(u, self.usats)])
        except TypeError:
            pass

        self.e = e
        self.t = t

        if debug:
            return u, e, self.i, p, i, d, de
        else:
            return u

    def reset(self):
        """Resetea el controlador, seteando a cero la integral y el
        error"""
        self.e = np.zeros_like(self.e)
        self.i = np.zeros_like(self.i)