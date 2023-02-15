#!/usr/bin/env python
# coding: utf-8

# # Algoritmo do Bandido
# 
# #### Prof. Armando Alves Neto - Introdução ao Aprendizado por Reforço - PPGEE/UFMG
# 
# <img src="k-armed_bandit.png" width="400">
# 
# Você deve repetidamente escolher uma entre $k$ diferentes ações. Após cada escolha, você recebe uma recompensa numérica (distribuição probabilística) ligada a sua ação e que pode influenciar suas escolhas futuras. Seu objetivo é maximizar a recompensa total esperada ao final de um período fazendo as escolhas certas.

# Importando bibliotecas.

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10,5)
import seaborn as sns


# Criando e inicializando a classe do Bandido.

# In[5]:


class Bandit:
    ##########################################
    def __init__(self, K, steps, eps=0.0, method='e-greedy'):

        # k bandits
        self.K = K
        
        # testbed
        self.testbed = [{'mean': 2.0*np.random.uniform(-1.0,1.0), 'std': 0.5} for a in range(K)]
        
        # numero de jogadas de 1 episodio
        self.steps = steps
        
        # epsilon-greedy
        self.eps = eps
        
        # constant c do UCB
        self.c = parameters['c']

        # initialize action values
        self.Q = np.array([0.0 for a in range(self.K)])
        self.N = np.array([0.0 for a in range(self.K)])

        # metodo
        self.method = method
        
        # instante de tempo
        self.t = 0
        
        # melhor acao dentre todas
        self.best_action = np.argmax([t['mean'] for t in self.testbed])

    ##########################################
    # bandit
    def bandit(self, A):
        return np.random.normal(self.testbed[A]['mean'], self.testbed[A]['std'])


# A função de classe ```selectAction()``` implementa duas versões de escolha da ação, a quase-gulosa (ou $\varepsilon$-gulosa),
#  
# $$
# A_t = 
# \begin{cases}
#     \arg\!\max\limits_{a} ~Q_t(a) & \text{com probabilidade}~ 1 - \varepsilon,\\
#     \textrm{ação aleatória} & \text{com probabilidade}~ \varepsilon
# \end{cases}
# $$
# 
# e a Upper-Confidence-Bound (UCB),
# 
# $$
# A_t \doteq \arg\!\max_{a} \left[ Q_t(a) + c~ \sqrt{\frac{\ln t}{N_t(a)}}~\right].
# $$

# In[6]:


class Bandit(Bandit):
    ##########################################
    def selectAction(self):

        ############################
        if self.method == 'e-greedy':
            # exploration
            if np.random.random() <= self.eps:
                a = np.random.choice(self.K)
            # explotation
            else:
                a = np.argmax(self.Q)

        ############################
        if self.method == 'ucb':
            ucb = np.array([self.c*np.sqrt(np.log(self.t)/(self.N[A]+0.01)) for A in range(self.K)])
            a = np.argmax(self.Q + ucb)
            
        return a


# Função da classe que executa um episódio.

# In[7]:


class Bandit(Bandit):
    ##########################################
    def runEpisode(self):
        # rewards de um epsodio
        rewards = []
        # escolha da melhor acao
        best_action = []

        # main loop
        for i in range(self.steps):

            self.t += 1
            # select action
            A = self.selectAction()

            # get reward
            R = self.bandit(A)

            # update action-value
            self.N[A] = self.N[A] + 1
            self.Q[A] = self.Q[A] + (1.0/self.N[A])*(R - self.Q[A])

            # accumulate rewards
            rewards.append(R)
            # acumula melhor acao
            best_action.append(A == self.best_action)

        return rewards, best_action


# Aqui, com os parâmetros fornecidos, rodamos vários episódios do problema, retornando o comportamento médio destes.

# In[8]:


def main(parameters):

    accum_reward = []
    accum_action = []
    for i in range(parameters['episodes']):
        # cria o testbed
        bandit = Bandit(K=parameters['k-armed'], steps=parameters['steps'], eps=parameters['eps'], method=parameters['method'])
        
        # roda um epsodio
        rewards, best_action = bandit.runEpisode()
        accum_reward.append(rewards)
        accum_action.append(best_action)

    # media de todos os epsodios
    avg_reward = np.mean(accum_reward, 0)

    # acao otima
    opt_action = 100.0*np.mean(accum_action, 0)
    
    return avg_reward, opt_action


# Definindo parâmetros principais do algoritmo.

# In[9]:


if __name__ == "__main__":

    sns.set()
    
    # parametros
    parameters = {
                'k-armed'  : 10,
                'steps'    : 1000,
                'episodes' : 2000,
                'method'   : '',
                'eps'      : 0.0,
                'c'        : 1.0,
            }
    
    ##########################################
    # greedy and nongreedy
    parameters['method'] = 'e-greedy'
    for eps in [0.0, 0.01, 0.1]:
        parameters['eps'] = eps
        avg_reward, opt_action = main(parameters)
        
        plt.figure(1)
        plt.plot(avg_reward, label='E-greedy (e=%.2f)'%eps)
        plt.figure(2)
        plt.plot(opt_action, label='E-greedy (e=%.2f)'%eps)
    
    ##########################################
    # Upper-Confidence-Bound
    parameters['method'] = 'ucb'
    for c in [1.0, 2.0]:
        parameters['c'] = c
        avg_reward, opt_action = main(parameters)

        plt.figure(1)
        plt.plot(avg_reward, label='UCB (c=%.2f)'%c)
        plt.figure(2)
        plt.plot(opt_action, label='UCB (c=%.2f)'%c)
    
    ##########################################
    plt.figure(1)
    plt.xlabel('Steps')
    plt.ylabel('Average rewards')
    plt.legend(loc='lower right', fancybox=True, shadow=True, fontsize=12, facecolor='w')

    plt.figure(2)
    plt.xlabel('Steps')
    plt.ylabel('Optimal action [%]')
    plt.legend(loc='lower right', fancybox=True, shadow=True, fontsize=12, facecolor='w')

    plt.show()

