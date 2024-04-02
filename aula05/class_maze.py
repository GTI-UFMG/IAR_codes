# -*- coding: utf-8 -*-
# Introdução ao Aprendizado por Reforço - PPGEE
# Prof. Armando Alves Neto
########################################################################
try:
    import gymnasium as gym
    from gymnasium import spaces
except:
    import gym
    from gym import spaces
import matplotlib.pyplot as plt
import numpy as np
import cv2
from functools import partial

NACTIONS = 9
FONTSIZE = 12
MAX_STEPS = 100

########################################
# classe do mapa
########################################
class Maze(gym.Env):
    ########################################
    # construtor
    def __init__(self, xlim=np.array([0.0, 10.0]), ylim=np.array([0.0, 10.0]), res=0.4, img='cave.png', alvo=np.array([9.5, 9.5])):

        # salva o tamanho geometrico da imagem em metros
        self.xlim = xlim
        self.ylim = ylim

        # resolucao
        self.res = res

        ns = int(np.max([np.abs(np.diff(self.xlim)), np.abs(np.diff(self.ylim))])/res)
        self.num_states = [ns, ns]
        
        # espaco de atuacao
        self.action_space = spaces.Discrete(NACTIONS)

        # cria mapa
        self.init2D(img)

        # converte estados continuos em discretos
        lower_bounds = [self.xlim[0], self.ylim[0]]
        upper_bounds = [self.xlim[1], self.ylim[1]]
        self.get_state = partial(self.obs_to_state, self.num_states, lower_bounds, upper_bounds)

        # alvo
        self.alvo = alvo

    ########################################
    # seed
    ########################################
    def seed(self, rnd_seed = None):
        np.random.seed(rnd_seed)
        return [rnd_seed]

    ########################################
    # reset
    ########################################
    def reset(self):

        # numero de passos
        self.steps = 0

        # posicao aleatória
        self.p = self.getRand()

        return self.get_state(self.p)

    ########################################
    # converte acão para direção
    def actionU(self, action):
        
        # action 0 faz ficar parado
        if action == 0:
            r = 0.0
        else:
            r = self.res
        
        action -= 1
        th = np.linspace(0.0, 2.0*np.pi, NACTIONS)[:-1]
        
        return r*np.array([np.cos(th[action]), np.sin(th[action])])
        
    ########################################
    # step -> new_observation, reward, done, info = env.step(action)
    def step(self, action):

        # novo passo
        self.steps += 1
        
        # seleciona acao
        u = self.actionU(action)

        # proximo estado
        nextp = self.p + u

        # fora dos limites (norte, sul, leste, oeste)
        if ( (self.xlim[0] <= nextp[0] < self.xlim[1]) and (self.ylim[0] <= nextp[1] < self.ylim[1]) ):
            self.p = nextp
         
        # reward
        reward = self.getReward()
        
        # estado terminal?
        done = self.terminal()

        # retorna
        return self.get_state(self.p), reward, done, {}

    ########################################
    # função de reforço
    def getReward(self):
        
        # reward
        reward = 0.0
        
        # colisao
        if self.collision(self.p):
            reward -= MAX_STEPS/2.0
            
        # chegou no alvo
        if np.linalg.norm(self.p - self.alvo) <= self.res:
            reward += MAX_STEPS
            
        if self.steps > MAX_STEPS:
            reward -= MAX_STEPS/5.0
            
        return reward
    
    ########################################
    # terminou?
    def terminal(self):
        # colisao
        if self.collision(self.p):
            return True
        # chegou no alvo
        if np.linalg.norm(self.p - self.alvo) <= self.res:
            return True
        if self.steps > MAX_STEPS:
            return True
        return False
        
    ########################################
    # ambientes em 2D
    def init2D(self, image):

        # le a imagem
        I = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

        # linhas e colunas da imagem
        self.nrow = I.shape[0]
        self.ncol = I.shape[1]

        # binariza imagem
        (thresh, I) = cv2.threshold(I, 127, 255, cv2.THRESH_BINARY)

        # inverte a imagem em y
        self.mapa = cv2.flip(I, 0)

        # parametros de conversao
        self.mx = float(self.ncol) / float(self.xlim[1] - self.xlim[0])
        self.my = float(self.nrow) / float(self.ylim[1] - self.ylim[0])

    ########################################
    # pega ponto aleatorio no voronoi
    def getRand(self):
        # pega um ponto aleatorio
        while True:
            qx = np.random.uniform(self.xlim[0], self.xlim[1])
            qy = np.random.uniform(self.ylim[0], self.ylim[1])
            q = (qx, qy)
            # verifica colisao
            if not self.collision(q):
                break

        # retorna
        return q

    ########################################
    # verifica colisao com os obstaculos
    def collision(self, q):

        # posicao de colisao na imagem
        px, py = self.mts2px(q)
        col = int(px)
        lin = int(py)

        # verifica se esta dentro do ambiente
        if (lin <= 0) or (lin >= self.nrow):
            return True
        if (col <= 0) or (col >= self.ncol):
            return True

        # colisao
        try:
            if self.mapa.item(lin, col) < 127:
                return True
        except IndexError:
            None

        return False

    ########################################
    # transforma pontos no mundo real para pixels na imagem
    def mts2px(self, q):
        qx, qy = q
        # conversao
        px = (qx - self.xlim[0])*self.mx
        py = self.nrow - (qy - self.ylim[0])*self.my

        return px, py

    ##########################################
    # converte estados continuos em discretos
    def obs_to_state(self, num_states, lower_bounds, upper_bounds, obs):
        state_idx = []
        for ob, lower, upper, num in zip(obs, lower_bounds, upper_bounds, num_states):
            state_idx.append(self.discretize_val(ob, lower, upper, num))

        return np.ravel_multi_index(state_idx, num_states)

    ##########################################
    # discretiza um valor
    def discretize_val(self, val, min_val, max_val, num_states):
        state = int(num_states * (val - min_val) / (max_val - min_val))
        if state >= num_states:
            state = num_states - 1
        if state < 0:
            state = 0
        return state

    ########################################
    # desenha a imagem distorcida em metros
    def render(self, Q):
        
        # desenha o robo
        plt.plot(self.p[0], self.p[1], 'rs')

        # desenha o alvo
        plt.plot(self.alvo[0], self.alvo[1], 'r', marker='x', markersize=20, linewidth=10)

        # plota mapa real e o mapa obsevado
        plt.imshow(self.mapa, cmap='gray', extent=[self.xlim[0], self.xlim[1], self.ylim[0], self.ylim[1]], alpha=0.5)

        # vector field
        m = self.num_states[0]
        xm = np.linspace(self.xlim[0], self.xlim[1], m)
        ym = np.linspace(self.ylim[0], self.ylim[1], m)
        XX, YY = np.meshgrid(xm, ym)

        th = np.linspace(0.0, 2.0*np.pi, NACTIONS)[:-1]
        vx = []
        vy = []
        for x in xm:
            for y in ym:
                S = self.get_state(np.array([y, x]))
                # plota a melhor ação                
                u = self.actionU(Q[S, :].argmax())
                vx.append(u[0])
                vy.append(u[1])
                    
        Vx = np.array(vx)
        Vy = np.array(vy)
        M = np.hypot(Vx, Vy)
        plt.gca().quiver(XX, YY, Vx, Vy, M, cmap='crest', angles='xy', scale_units='xy', scale=1.5, headwidth=5)

        plt.xticks([], fontsize=FONTSIZE)
        plt.yticks([], fontsize=FONTSIZE)
        plt.xlim(self.xlim + 0.05*np.abs(np.diff(self.xlim))*np.array([-1., 1.]))
        plt.ylim(self.ylim + 0.05*np.abs(np.diff(self.ylim))*np.array([-1., 1.]))
        plt.box(True)
        plt.show()
        plt.pause(.1)

    ########################################
    def __del__(self):
        None
