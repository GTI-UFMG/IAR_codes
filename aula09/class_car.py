# -*- coding: utf-8 -*-
import colorsys
import numpy as np

########################################
# GLOBAIS
########################################
# parametros do ambiente
ENV = {
        'DT'     : 5.0e-2,
        'GRAV'   : 9.81, # gravidade [m/s^2]
        'RHO'    : 1.23, # densidade do ar [kg/m^3]}
        'VELMAX' : 72.0/3.6, # maxima velocidade da pista
    }

# parametros do carro
CAR = {
        'ZETALEADER' : 0.3, # dinamica do lider
        'MAXTORQUE'  : np.inf,
        'MASS'       : 1500.0, # massa [kg]
        'CD'         : 0.4, # coeficiente de arrasto de posicao
        'RW'         : 0.25, # raio da roda [m]
        'MI'         : 0.015, # constante de friccao
        'ETA'        : 0.8, # eficiencia do motor [%]
        'UMAX'       : 3.0,
    }

DELTA = 10.0 # delta minimo entre os carros

# limites dos estados
MAX_EP  = 2.0*DELTA
MAX_EV  = 2.0*ENV['VELMAX']
MAX_EA  = 2.0*CAR['UMAX']

########################################
# Carro
########################################
class Car:
    _counter = 0

    ########################################
    # construtor
    ########################################
    def __init__(self, x = None):

        self.ENV = ENV
        self.CAR = CAR

        # conta as instancias
        self.id = Car._counter
        Car._counter += 1

        # tempo
        self.t = 0.0
        self.dt = self.ENV['DT']

        # distancia de espacamento
        self.deltai = DELTA if self.id > 0 else 0.0
        
        self.zeta = self.CAR['ZETALEADER'] 
        self.mass  = self.CAR['MASS'] # massa [kg]
        self.eta   = self.CAR['ETA'] # eficiencia do motor [%]
        self.cd    = self.CAR['CD'] # coeficiente de arrasto de posicao
        self.rw    = self.CAR['RW'] # raio da roda [m]
        self.mi    = self.CAR['MI'] # constante de friccao

        # torque inicial
        self.T = 0.0
        self.Tref = 0.0
        
        if x is not None:
            self.p, self.v, self.a = x
        else:
            self.p = -1.0*self.id*self.deltai # comeca com erro zero
            self.v = 0.0
            self.a = 0.0

        # entrada inicial
        self.setU(0.0)

        # salva trajetoria
        self.saveTraj()

    ########################################
    # seta vel do lider
    ########################################
    def setLeader(self, u):

        # lider tem velocidade abaixo da pista
        if self.id == 0:
            # nao acelera ou freia mais que o permitido
            if self.v <= -self.ENV['VELMAX']:
                u = max(u, 0.0)
            if self.v >= self.ENV['VELMAX']:
                u = min(u, 0.0)

            # seta comando externo
            self.setU(u)

    ########################################
    # set u command
    ########################################
    def setU(self, u):
        u = np.squeeze(u)
        self.u = np.clip(u, -self.CAR['UMAX'], self.CAR['UMAX'])

        # envia comando de atuacao
        self.actuator()

    ########################################
    # modelo dinamico
    ########################################
    def model(self, x = [], u = None):

        ##################
        # se nao for o lider
        if (self.id > 0):
            self.setU(u)

        ##################
        # integra modelo
        self.modelRK()

        # salva trajetoria
        self.saveTraj()

    ########################################
    # atualiza o modelo do robo (runge-kutta)
    # http://doswa.com/2009/01/02/fourth-order-runge-kutta-numerical-integration.html
    ########################################
    def modelRK(self):

        # update de tempo
        self.t += self.dt
        v1 = self.v
        a1 = self.getAccel(v1)
        v2 = self.v + 0.5*a1*self.dt
        a2 = self.getAccel(v2)
        v3 = self.v + 0.5*a2*self.dt
        a3 = self.getAccel(v3)
        v4 = self.v + a3*self.dt
        a4 = self.getAccel(v4)
        # aceleracao translacional
        self.a = (a1 + 2.0*a2 + 2.0*a3 + a4)/6.0

        # velocidade translacional
        self.v += self.a*self.dt

        # robo nao anda de re
        self.p += self.v*self.dt + 0.5*self.a*(self.dt**2.0)

    ########################################
    # salva a trajetoria
    ########################################
    def saveTraj(self):

        # dados
        data = {'t'  : self.t,
                'p'  : self.p,
                'v'  : self.v,
                'a'  : self.a,
                'u'  : self.u,}

        # se ja iniciou as trajetorias
        try:
            self.traj.append(data)
        # se for a primeira vez
        except:
            self.traj = [data]

    ########################################
    # get data
    ########################################
    def getData(self):
        return np.array([self.p, self.v, self.a, self.deltai])

    ########################################
    # calcula aceleracao baseado no modelo
    ########################################
    def getAccel(self, v):

        # forcas que atuam logitudinalmente
        F = []
        F.append(-np.sign(v)*self.mass*self.ENV['GRAV']*self.mi) # atrito
        F.append(-0.5*self.ENV['RHO']*self.cd*v*abs(v)) # arrasto
        F.append((self.eta/self.rw)*self.T) # torque do motor

        # aceleracao translacional
        return np.sum(F)/self.mass

    ########################################
    # seta entrada do modelo
    ########################################
    def actuator(self):

        zeta = self.zeta
        mass = self.mass

        # controlador linearizante
        T = []
        T.append(0.5*self.ENV['RHO']*self.cd*self.v*(2.0*zeta*self.a + self.v))
        T.append(np.sign(self.v)*mass*self.ENV['GRAV']*self.mi)
        T.append(mass*self.u)

        # torque de referencia
        self.Tref = (self.rw/self.eta)*np.sum(T)

        # seta o torque do motor
        self.setTorque(self.Tref)

    ########################################
    # implementa dinamica do motor
    ########################################
    def setTorque(self, Tref):

        # torque da simulacao
        dT = (Tref - self.T)/self.zeta
        self.T = self.T + dT*self.dt

        # satura a tracao do motor
        self.T = np.clip(self.T, -self.CAR['MAXTORQUE'], self.CAR['MAXTORQUE'])

    ########################################
    # comeca a missao
    ########################################
    def startMission(self):

        # robos no time
        self.n = Car._counter

        # cor de plot
        if self.id == 0:
            self.cor = 'k'
        else:
            self.cor = colorsys.hsv_to_rgb(self.id/float(self.n-1), .8, .8)

    ########################################
    # termina a classe
    ########################################
    def __del__(self):
        try:
            Car._counter = 0
        except:
            None