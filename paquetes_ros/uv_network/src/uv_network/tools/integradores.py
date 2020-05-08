#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author Patricio Moreno
@institute GPSIC - FIUBA, Universidad de Buenos Aires, Argentina
@date 29/01/2020

Implementación de algunos integradores para simular con python. Está
como me conviene en este momento hacerlo (29/01/2020).
'''

class MetodoDeIntegracion(object):
    '''Clase base para otros métodos de integración. Todos los métodos
    van a tener una dinámica a integrar, pueden tener que normalizar
    después de cada paso de integración (como los cuaterniones), y todos
    se inicializan con un tiempo y posición inicial---porque ahora me
    resulta conveniente hacerlo así.'''
    def __init__(self, dinamica, normalizar):
        self.dinamica = dinamica
        self.normalizar = normalizar
        self.inicializar(0., 0.)

    def inicializar(self, xi, ti):
        '''Inicializa el integrador y guarda los parámetros iniciales'''
        self.xi = xi
        self.ti = ti
        self.x = xi
        self.t = ti

    def reiniciar(self):
        '''Reinicia el integrador'''
        self.x = self.xi
        self.t = self.ti

class RK4(MetodoDeIntegracion):
    def __init__(self, dinamica, normalizar=None, xi=0., ti=0.):
        super(RK4, self).__init__(dinamica, normalizar)
        super(RK4, self).inicializar(xi, ti)

    def step(self, t, args):
        '''args contiene todos los argumentos adicionales que requiera
        la dinámica. Es una tupla o lista de 3 elementos (al menos). El
        primer elemento tiene el conjunto de argumentos que requiere la
        dinámica para el instante t, el segundo el elemento los que se
        requieren para el instante t + h/2, y así.'''
        h = t - self.t
        h2 = h / 2.0
        k1 = self.dinamica(self.x          , self.t     , *args[0])
        k2 = self.dinamica(self.x + k1 * h2, self.t + h2, *args[1])
        k3 = self.dinamica(self.x + k2 * h2, self.t + h2, *args[1])
        k4 = self.dinamica(self.x + k3 * h , t          , *args[2])
        self.t = t
        self.x += (h/6.0)*(k1 + k2 + k2 + k3 + k3 + k4)
        try:
            self.x = self.normalizar(self.x)
        except:
            pass
        return self.x

class EulerExplicito(MetodoDeIntegracion):
    def __init__(self, dinamica, normalizar=None, xi=0., ti=0.):
        super(EulerExplicito, self).__init__(dinamica, normalizar)
        super(EulerExplicito, self).inicializar(xi, ti)

    def step(self, t, args):
        '''args contiene todos los argumentos adicionales que requiera
        la dinámica. Es una tupla o lista de 1 elemento (al menos). El
        primer elemento tiene el conjunto de argumentos que requiere la
        dinámica para el instante t.'''
        h = t - self.t
        self.x += h * self.dinamica(self.x, self.t, *args[0])
        self.t = t
        try:
            self.x = self.normalizar(self.x)
        except:
            pass
        return self.x
