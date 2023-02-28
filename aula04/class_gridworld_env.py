########################################################################
# -*- coding: utf-8 -*-
# Introdução ao Aprendizado por Reforço - PPGEE
# Prof. Armando Alves Neto
########################################################################
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from matplotlib.colors import LinearSegmentedColormap

##########################################
# globais
##########################################
GRID_SIZE = 8

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
        self.actions = [np.array([np.cos(th), np.sin(th)]) for th in np.linspace(0, 2.0*np.pi, 5)[:-1]]

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
        self.terminais = [np.array([0,0]), np.array([self.size-1, self.size-1])]
        
        yes = True
        while yes:
            yes = False
            state = np.random.choice(self.size, 2)
            for t in self.terminais:
                if  np.array_equal(state, t):
                    yes = True
                
        return tuple(state[:])

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
        
        # outros
        done = False
        info = {}

        # eh um estado final?
        for t in self.terminais:
            if  np.array_equal(state, t):
                next_state = state
                reward = 0.0
                done = True

        next_state = tuple(next_state[:])
        
        # retorna
        return next_state, reward, done, info

    ##########################################
    # reforço
    def getReward(self):
        return -1.0

    ##########################################
    def render(self, value, pi=None, title=None):

        # Plota mapa de valor
        fig, ax = plt.subplots(figsize=(self.size, self.size))
        if title is not None:
            ax.set_title(title)
        W = LinearSegmentedColormap.from_list('w', ["w", "w"], N=256)
        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
        sn.heatmap(value, annot=True, fmt=".1f", cmap='crest', linewidths=1, linecolor="black", cbar=False)

        # Plota mapa da politica
        if not (pi is None):
            arrows = np.array([UP, RIGHT, DOWN, LEFT])
            labels = arrows[pi]
            labels[value == 0.0] = ''

            # Plota valor
            fig2, ax2 = plt.subplots(figsize=(self.size, self.size))
            if title is not None:
                ax2.set_title(title)
            W = LinearSegmentedColormap.from_list('w', ["w", "w"], N=256)
            ax2.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
            sn.heatmap(value, annot=labels, fmt="", cmap='crest', linewidths=1, linecolor="black", cbar=False)

        plt.show()