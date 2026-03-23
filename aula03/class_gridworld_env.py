########################################################################
# -*- coding: utf-8 -*-
# Introdução ao Aprendizado por Reforço - PPGEE
# Prof. Armando Alves Neto
########################################################################
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sn

##########################################
# globais
##########################################
GRID_SIZE = 4

RIGHT = "\u2192"
UP = "\u2193"
LEFT = "\u2190"
DOWN = "\u2191"

##########################################
# Grid world environment
##########################################
class Gridworld_Env:
    def __init__(self, size=GRID_SIZE):

        # salva tamanho
        self.size = size

        # acoes
        self.actions = [
                           np.array([-1,  0]),  # up
                           np.array([ 0,  1]),   # right
                           np.array([ 1,  0]),   # down
                           np.array([ 0, -1])   # left
                       ]

        # inicia o mundo
        self.reset()

    ##########################################
    # tamanho do grid quadrado
    def getSize(self):
        return self.size

    ##########################################
    # reseta o grid
    def reset(self):
        # estados terminais
        self.terminal_states = {(0, 0), (self.size-1, self.size-1)}
        
        # retorna estado aleatório
        state = np.random.choice(self.size, 2)

        self.state = tuple(state)
        return self.state

    ##########################################
    # step
    def step(self, state, action):

        # proximo estado
        next_state = np.round((state + action)).astype(int)

        # fora dos limites (norte, sul, leste, oeste
        if not ( (0 <= next_state[0] < self.size) and (0 <= next_state[1] < self.size) ):
            next_state = state

        # reward
        reward = self.getReward()

        # eh um estado final?
        if state in self.terminal_states:
            return tuple(state), 0.0, True, {'Terminou'}
        else:
            return tuple(next_state), reward, False, {}

    ##########################################
    # reforço
    def getReward(self):
        return -1.0

    ##########################################
    def render(self, value, pi=None):

        # Plota mapa de valor
        if not (pi is None):
            fig = plt.subplots(figsize=(2*self.size, self.size))
            plt.subplot(1, 2, 1)
        else:
            fig = plt.subplots(figsize=(self.size, self.size))

        # funcao valor
        sn.heatmap(value, annot=True, fmt=".1f", cmap='crest', linewidths=1, linecolor="black", cbar=False, square=True)
        plt.gca().set_title(r'$V(s)$')

        # Plota mapa da politica
        if not (pi is None):
            arrows = np.array([UP, RIGHT, DOWN, LEFT])
            labels = arrows[pi]
            # sem acoes nos terminais
            for t in self.terminais:
                labels[tuple(t[:])] = ''

            # Plota valor
            plt.subplot(1, 2, 2)
            plt.gca().set_title(r'$\pi(s) \approx \pi_*$')
            W = LinearSegmentedColormap.from_list('w', ["w", "w"], N=256)
            plt.gca().grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
            sn.heatmap(value, annot=labels, fmt="", cmap='crest', linewidths=1, linecolor="black", cbar=False)

        plt.show()
