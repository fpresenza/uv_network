#!/usr/bin/env python
import numpy as np

class MotionModelVelocity(object):
    """ Esta clase implementa un modelo de dinamica de velidad 
    de un robot.
    """
    def __init__(self, dof, ctrl_gain=1., alphas=[[0.,0.,0.],[0.,0.,0.]]):
        """ Para inicializar una instancia se debe ingresar:
        dof: nro. de grados de libertad del robot
        ctrl_gain: controlador interno del modelo de movimiento
        Los parametros del modelo de error de la dinamica: 
        alpha[0]: std. dev. en la accion de los actuadores
        alpha[1]: std. dev. en la medicion interna de la velocidad
        """
        self.dof = dof
        self.alphas = alphas
        I = np.eye(self.dof)
        Z = np.zeros_like(I)
        #   controller matrix
        self.K = np.diag(np.broadcast_to(ctrl_gain, self.dof))
        #   state transition matrix        
        self.G_X = np.block([[-self.K, Z, Z],
                             [      I, Z, Z],
                             [      Z, Z, Z]])
        #   input transition matrix        
        self.G_v = np.block([[self.K],
                             [     Z],
                             [     Z]])
        #   disturbances
        self.B = np.block([[I, -self.K],
                           [Z,       Z],
                           [Z,       Z]])
        #   noise matrix
        self.Q = np.diag(np.square(self.alphas[0]+self.alphas[1]))
        
    def sample(self, state, cmd_vel, Ts):
        """Implementa un sistema de doble integrador con lazo 
        cerrado en velocidad. Como parametros toma comandos de velocidad,
        el tiempo actual en segundos. Devuelve una muestra de la futura 
        posicion y velocidad en base al modelo de perturbacion.
        K : controller gain, must be set in design step.
        """
        #   noise in control actuators
        n1 = np.random.normal(0., np.array(self.alphas[0]).reshape(-1,1))
        #   noise in internal velocity sensing 
        n2 = np.random.normal(0., np.array(self.alphas[1]).reshape(-1,1))       
        #   Discrete state matrix
        Ad = np.eye(*self.G_X.shape) + Ts*self.G_X
        #   Discrete state-input matrix
        Bd = Ts*np.block([self.G_v, self.B])
        #   system input
        u = np.block([[cmd_vel],
                      [     n1],
                      [     n2]])
        #   New state estimate
        sample = np.dot(Ad, state) + np.dot(Bd, u)        
        meas_vel = sample[:3] + n2
        acc = np.dot(self.K, cmd_vel-meas_vel) + n1
        return sample, acc

    def gaussian_propagation(self, mean, covariance, cmd_vel, Ts):
        """ This function takes mean and covariance of a gaussian proccess,
        commanded velocity and propagates gaussian moments using motion model.
        """
        #   Discretization
        Phi = np.eye(*self.G_X.shape) + Ts*self.G_X
        Ad = Phi
        Phi_int = Ts * np.eye(*self.G_X.shape)       
        Bd = np.dot(Phi_int, self.G_v)
        v = cmd_vel[:self.dof]
        #   Propagation of state and covariance
        mean = np.dot(Ad, mean) + np.dot(Bd, v)
        covariance = np.linalg.multi_dot([Phi, covariance, Phi.T]) \
            + np.linalg.multi_dot([self.B, self.Q, self.B.T]) * Ts
        return mean, covariance

    def f(self, mean, u):
        """ This function takes mean and covariance of a gaussian proccess,
        an IMU measurement to generate dot_X and matrices needed for prediction.
        """
        pass


class VelocityRandomWalk(object):
    """ This class emulates a Wiener proccess dynamic
    """
    def __init__(self, **kwargs):
        self.rate = kwargs.get('rate', 1.)
        NSD = kwargs.get('NSD', 0.)
        self.agents = kwargs.get('agents', 1)
        self.dim = kwargs.get('dim', 1)
        self.N = self.agents * self.dim  
        BW = 0.5 * self.rate
        self.sigma = NSD * np.sqrt(2*BW) * 0.01
        self.Q = np.diag(np.repeat(self.sigma**2, self.N))
    
    def __len__(self):
        return 2*self.N

    def f(self, mean, u):
        v = mean[:self.N]
        dot_v = np.random.normal(0., self.sigma * np.ones_like(v))
        dot_x = v
        dot_X = np.vstack((dot_v, dot_x)) 
        I = np.eye(self.N)
        Z = np.zeros_like(I)
        F_X = np.block([[Z, Z],
                        [I, Z]])
        B = np.block([[I],
                      [Z]])
        return dot_X, F_X, B, self.Q