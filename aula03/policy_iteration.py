#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Introdução ao Aprendizado por Reforço - PPGEE
# Prof. Armando Alves Neto
########################################################################
import class_gridworld_env as cge
import numpy as np

##########################################
# globais
##########################################
GAMMA = 0.9
THETA = 1.0e-3

##########################################
# Policy evaluation
##########################################
class PolicyIteration:
	def __init__(self):
		
		# environment
		self.env = cge.Gridworld_Env()
		self.size = self.env.getSize()
		
		# fator de desconto
		self.gamma = GAMMA
		
		# inicia o mundo
		self.reset()
	
	##########################################
	def reset(self):
		
		# reseta ambiente
		self.env.reset()
		
		# value function 
		self.value = np.zeros((self.size, self.size))
		
		# uniform random policy
		nactions = len(self.env.actions)
		
		# politica
		self.pi = np.random.choice(nactions, (self.size, self.size))
		
		# quantos passos
		self.steps = 0
		
	##########################################
	# Bellman equation
	def bellmanEquation(self, s):
		
		# pega a ação corrente da politica 
		action = self.env.actions[int(self.pi[s])]
		
		# interage com o ambiente
		sl, r, _, _ = self.env.step(s, action)
		
		# valor obtido
		v = r + self.gamma*self.value[sl]
		
		return v
		
	##########################################
	# Iterative policy evaluation
	def policyEvaluation(self, in_place=False):
		
		while True:
			Delta = 0.0
			
			# synchronous oy asynchronous mode?
			Vs  = self.value if in_place else np.empty_like(self.value)
			
			# para todos os estados
			for i in range(len(self.value)):
				for j in range(len(self.value[i])):
					
					# estado
					s = (i,j)
					
					# apply bellman expectation equation to each state
					v = Vs[s]
					Vs[s] = self.bellmanEquation(s)
					
					Delta = np.max([Delta, np.abs(v - Vs[s])])
			
			# atualiza valores
			self.value = Vs
			
			self.steps = int(self.steps + 1)
			
			print(Delta)
			
			# convergiu?
			if Delta < THETA: break
			
		return self.value
	
	##########################################
	# Policy improvement
	def policyImprovement(self):
		
		policy_stable = True
		
		# para todos os estados
		for i in range(len(self.value)):
			for j in range(len(self.value[i])):
				
				# estado
				s = (i,j)
				
				# calcula a politica otima corrente para cada estado
				old_action = self.pi[s]
				
				# para todas as possiveis ações
				acts = []
				for a, action in enumerate(self.env.actions):
					sl, r, _, _ = self.env.step(s, action)
					acts.append(r + self.gamma*self.value[sl])
				
				self.pi[s] = np.argmax(acts)
				
				# nao convergiu ainda
				if old_action != self.pi[s]:
					policy_stable = False
					
		return policy_stable
		
	##########################################
	# Policy iteration
	def runEpsisode(self, in_place=False):
		
		iterations = 0
		
		while True:
			iterations += 1
			
			# Policy Evaluation
			value_table = self.policyEvaluation(in_place)
			
			# Policy Improvement
			if self.policyImprovement():
				print('Convergiu em %d iteracoes' % iterations)
				break
		
		return value_table
	
##########################################
if __name__ == "__main__":
	
	pol_ite = PolicyIteration()
	
	value_table = pol_ite.runEpsisode()
	
	print('Convergiu em %d passos' % pol_ite.steps)
	
	# reprocuce Figure 4.1
	pol_ite.env.render(value_table)
	
