# -*- coding: utf-8 -*-
try:
	import gymnasium as gym
	from gymnasium import spaces
except:
	import gym
	from gym import spaces
gym.logger.set_level(40)

import numpy as np
import matplotlib.pyplot as plt
import class_car as cc

########################################
# GLOBAIS
########################################
MAX_EP = cc.MAX_EP
MAX_EV = cc.MAX_EV
MAX_EA = cc.MAX_EA
VELMAX = cc.ENV['VELMAX']
MAX_U  = cc.CAR['UMAX']
DT = cc.ENV['DT']

TMAX = 40.0
LINEW = 1.5

########################################
# Normaliza ações no ambiente
########################################
class NormalizedEnv(gym.ActionWrapper):
	""" Wrap action """
	def action(self, action):
		act_k = (self.action_space.high - self.action_space.low)/ 2.
		act_b = (self.action_space.high + self.action_space.low)/ 2.
		return act_k * action + act_b

	def reverse_action(self, action):
		act_k_inv = 2./(self.action_space.high - self.action_space.low)
		act_b = (self.action_space.high + self.action_space.low)/ 2.
		return act_k_inv * (action - act_b)
	
########################################
# Carro
########################################
class PlatoonEnv(gym.Env):
	########################################
	# construtor
	########################################
	def __init__(self):

		# current episodes
		self.episodes = -1

		# reseta planta
		self.reset()

		# state box
		self.state_list_low  = np.array([-MAX_EP, -MAX_EV, -MAX_EA])
		self.state_list_high = np.array([ MAX_EP,  MAX_EV,  MAX_EA])
		self.observation_space = spaces.Box(low = self.state_list_low, high = self.state_list_high, dtype=np.float32)

		# action box
		self.action_space = spaces.Box(low=-MAX_U,  high=MAX_U,  shape=(1,), dtype=np.float32)

	########################################
	# seed
	########################################
	def seed(self, rnd_seed = None):
		np.random.seed(rnd_seed)
		return [rnd_seed]

	########################################
	# reset
	########################################
	def reset(self, seed=None, options=None):

		# incrementa episodes
		self.episodes += 1

		# aceleração do leader
		self.leaderAccel = 0.0

		# deleta carros anteriores
		self.close()

		# cria leader
		self.cars = [cc.Car()]

		# cria follower com posicao aleatoria
		p1 = self.cars[0].p - cc.DELTA + np.random.uniform(-MAX_EP, MAX_EP)
		v1 = self.cars[0].v + 0.01*np.random.uniform(-MAX_EV, MAX_EV)
		a1 = self.cars[0].a + 0.5*np.random.uniform(-MAX_EA, MAX_EA)
		self.cars.append(cc.Car(x = np.array([p1, v1, a1])))
		
		# acao de controle inicial
		self.u = 0.0

		# comeca a missao
		[c.startMission() for c in self.cars]

		# erros iniciais
		self.state = self.getError()

		return self.state, None

	########################################
	# get state
	def getError(self):

		# leader states
		x0 = self.cars[0].getData()

		# vehicle's states
		ep = self.cars[1].p - x0[0] + cc.DELTA
		ev = self.cars[1].v - x0[1]
		ea = self.cars[1].a - x0[2]

		ep = np.clip(ep, -MAX_EP, MAX_EP)
		ev = np.clip(ev, -MAX_EV, MAX_EV)
		ea = np.clip(ea, -MAX_EA, MAX_EA)
		
		# state
		return np.array((ep, ev, ea), dtype=np.float32)

	########################################
	# step -> new_observation, reward, done, info = env.step(action)
	def step(self, action):

		#####################
		# atuacao
		#####################
		action = np.squeeze(action)
		self.u = np.clip(action, -MAX_U, MAX_U)

		#####################
		# define leader accel
		#####################
		if 1.0 <= self.cars[0].t <= 5.0:
			self.leaderAccel = 2.0
		else:
			self.leaderAccel = 0.0
		self.cars[0].setLeader(self.leaderAccel)

		# atualizando modelo
		self.cars[0].model()
		self.cars[1].model(u = self.u)

		#####################
		# proximo estado
		#####################
		self.state = self.getError()

		#####################
		# reward
		#####################
		ep, ev, ea = self.state
		alfa = [0.3, 0.3, 0.3, 0.2]
		erros = np.array([ep, ev, ea, (self.u/MAX_U)])**2.0
		#
		reward = np.exp(-np.dot(alfa, erros))

		# done
		done = bool(self.cars[0].t > TMAX)

		#new_observation, reward, done, info
		return self.state, reward, done, {}, {}

	########################################
	def plotEsp(self):
		t0 = [traj['t'] for traj in self.cars[0].traj]
		p0 = [traj['p'] for traj in self.cars[0].traj]
		p1 = [traj['p'] for traj in self.cars[1].traj]
		erro = np.array(p1) - np.array(p0) + np.array(cc.DELTA)
		plt.plot(t0, erro, color = self.cars[1].cor, linewidth=LINEW)
		plt.plot(t0, 0.0*erro, 'k:', linewidth=LINEW)
		plt.ylabel('Espaçamento[m]')
	
	########################################
	def plotPos(self):
		for c in self.cars:
			t0 = [traj['t'] for traj in c.traj]
			vx = [traj['p'] for traj in c.traj]
			plt.plot(t0, vx, color = c.cor, linewidth=LINEW, label='%i' % c.id)
		plt.ylabel('Posição[m/s]')
	
	########################################
	def plotVel(self):
		for c in self.cars:
			t0 = [traj['t'] for traj in c.traj]
			vx = [traj['v'] for traj in c.traj]
			plt.plot(t0, vx, color = c.cor, linewidth=LINEW, label='%i' % c.id)
		plt.ylabel('Velocidade[m/s]')
	
	########################################
	def plotU(self):
		for c in self.cars:
			t0 = [traj['t'] for traj in c.traj]
			u = [traj['u'] for traj in c.traj]
			plt.plot(t0, u, color = c.cor, linewidth=LINEW)
			plt.plot(t0,  MAX_U*np.ones(len(t0)), 'k:', linewidth=LINEW)
			plt.plot(t0, -MAX_U*np.ones(len(t0)), 'k:', linewidth=LINEW)
		plt.ylabel('u[m/s^2]')
	
	########################################
	# desenha
	def render(self, render_mode = 'human'):
		None

	########################################
	# fecha ambiente
	def close(self):
		# deleta carros anteriores
		try:
			del self.cars
		except:
			None

	########################################
	# termina a classe
	def __del__(self):
		None
