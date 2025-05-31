import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple
from pettingzoo.sisl import multiwalker_v9
import random
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (8,8)

# ============================== #
# Replay Buffer
# ============================== #
class ReplayBuffer:
	def __init__(self, batch_size, device, buffer_size=int(1e6)):
		self.batch_size = batch_size
		self.device = device
		self.buffer = deque(maxlen=buffer_size)
		self.experience = namedtuple("Experience", field_names=["S", "A", "R", "Sl", "done"])

	def push(self, S, A, R, Sl, done):
		e = self.experience(S, A, R, Sl, done)
		self.buffer.append(e)

	def sample(self):
		experiences = random.sample(self.buffer, k=min(len(self.buffer), self.batch_size))

		states = torch.from_numpy(np.vstack([e.S for e in experiences])).float().to(self.device)
		actions = torch.from_numpy(np.vstack([e.A for e in experiences])).float().to(self.device)
		rewards = torch.from_numpy(np.vstack([e.R for e in experiences])).float().to(self.device)
		next_states = torch.from_numpy(np.vstack([e.Sl for e in experiences])).float().to(self.device)
		dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(self.device)

		return (states, actions, rewards, next_states, dones)

	def __len__(self):
		return len(self.buffer)

# ============================== #
# Noise Process
# ============================== #
class OUNoise:
	def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
		self.mu = np.ones(size, dtype=np.float32)*mu
		self.theta = theta
		self.sigma = sigma
		self.size = size
		self.reset()

	def reset(self):
		self.state = np.ones(self.size, dtype=np.float32)*self.mu

	def get(self):
		dx = self.theta*(self.mu - self.state) + self.sigma*np.random.randn(self.size)
		self.state += dx
		return MAX_ACTION*self.state
		
# ============================== #
# Networks
# ============================== #
class Actor(nn.Module):
	def __init__(self, state_size, action_size, hidden_size=64, temperature=1.0):
		super(Actor, self).__init__()
		self.fc1 = nn.Linear(state_size, 400)
		self.fc2 = nn.Linear(400, 300)
		self.out = nn.Linear(300, action_size)
		self.temperature = temperature # suaviza a entrada da tanh

	def forward(self, state):
		x = F.relu(self.fc1(state))
		x = F.relu(self.fc2(x))
		return MAX_ACTION*torch.tanh(self.out(x)/self.temperature)

####################################################################
class Critic(nn.Module):
	def __init__(self, total_state_size, total_action_size, hidden_size=128):
		super(Critic, self).__init__()
		self.fc1 = nn.Linear(total_state_size + total_action_size, 400)
		self.fc2 = nn.Linear(400, 300)
		self.out = nn.Linear(300, 1)

	def forward(self, states, actions):
		x = torch.cat([states, actions], dim=1)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		return self.out(x)

# ============================== #
# MADDPG Agent
# ============================== #
class MADDPGAgent:
	####################################################################
	def __init__(self, agent_name, obs_size, act_size, all_obs_size, all_act_size, params):
		self.name = agent_name
		self.device = params['device']
		self.gamma = params['gamma']
		self.tau = params['tau']
		self.batch_size = params['batch_size']

		self.actor = Actor(obs_size, act_size).to(self.device)
		self.target_actor = Actor(obs_size, act_size).to(self.device)

		self.critic = Critic(all_obs_size, all_act_size).to(self.device)
		self.target_critic = Critic(all_obs_size, all_act_size).to(self.device)

		self.target_actor.load_state_dict(self.actor.state_dict())
		self.target_critic.load_state_dict(self.critic.state_dict())

		self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=params['lr_actor'])
		self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=params['lr_critic'])

		self.noise = OUNoise(act_size)

	####################################################################
	def update(self, replay_buffer, agents):
		if len(replay_buffer) < self.batch_size:
			return

		states, actions, rewards, next_states, dones = replay_buffer.sample()

		idx = list(agents.keys()).index(self.name)
		obs_dim = states.shape[1] // len(agents)
		act_dim = actions.shape[1] // len(agents)

		with torch.no_grad():
			next_actions = torch.cat([
				agents[a].target_actor(next_states[:, obs_dim*i:obs_dim*(i+1)])
				for i, a in enumerate(agents)
			], dim=1)

			target_Q = rewards[:, idx].unsqueeze(1) + \
					   (1 - dones[:, idx].unsqueeze(1)) * \
					   self.gamma * self.target_critic(next_states, next_actions)

		current_Q = self.critic(states, actions)

		critic_loss = F.mse_loss(current_Q, target_Q)

		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		current_actions = []
		for i, a in enumerate(agents):
			if a == self.name:
				act = self.actor(states[:, obs_dim*i:obs_dim*(i+1)])
			else:
				act = actions[:, act_dim*i:act_dim*(i+1)]
			current_actions.append(act)

		current_actions = torch.cat(current_actions, dim=1)
		actor_loss = -self.critic(states, current_actions).mean()

		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		self.soft_update(self.actor, self.target_actor)
		self.soft_update(self.critic, self.target_critic)

	####################################################################
	def soft_update(self, local_model, target_model):
		for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
			target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

# ============================== #
# Training Loop
# ============================== #
NAGENTS = 3
PERIOD = 20
MAX_ACTION = 1.0

env1 = multiwalker_v9.parallel_env(render_mode=None,    n_walkers=NAGENTS)
env2 = multiwalker_v9.parallel_env(render_mode='human', n_walkers=NAGENTS)
env = env1
obs, _ = env.reset()

obs_size = env.observation_space(env.agents[0]).shape[0]
act_size = env.action_space(env.agents[0]).shape[0]
total_obs_size = obs_size * NAGENTS
total_act_size = act_size * NAGENTS

params = {
	'episodes': 10000,
	'gamma': 0.95,
	'tau': 1.0e-2,
	'lr_actor': 1.0e-4,
	'lr_critic': 1.0e-3,
	'batch_size': 512,
	'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

plt.ion()
# historico dos reforços
recompensas = []
avg_rewards = []

agents = {
	name: MADDPGAgent(name, obs_size, act_size, total_obs_size, total_act_size, params)
	for name in env.agents
}

replay_buffer = ReplayBuffer(params['batch_size'], params['device'])

for episode in range(params['episodes']):
	
	if episode % PERIOD == 0:
		env = env2
	else:
		env = env1
	
	env.forward_reward = 10.0*(episode/params['episodes'])
	retorno = 0.0
	acoes = []
	
	obs, _ = env.reset()
	
	for agent in agents.values():
		agent.noise.reset()

	done = {agent: False for agent in env.agents}

	while not all(done.values()):
		
		env_action_space = env.action_space(env.agents[0])
			
		actions = {
			agent: np.clip(
				agents[agent].actor(torch.tensor(obs[agent],dtype=torch.float32).to(params['device'])).cpu().detach().numpy() + agents[agent].noise.get(),
				-MAX_ACTION, MAX_ACTION) #env_action_space.low, env_action_space.high)
			for agent in env.agents
		}

		next_obs, rewards, dones, _, _ = env.step(actions)
		
		for agent in env.agents:
			rewards[agent] = rewards[agent] + 0.35
		
		acoes.append(actions['walker_0'])
		
		retorno += rewards['walker_0']

		try:
			obs_concat = np.concatenate([obs[a] for a in env.agents])
			actions_concat = np.concatenate([actions[a] for a in env.agents])
			next_obs_concat = np.concatenate([next_obs[a] for a in env.agents])
			rewards_array = np.array([rewards[a] for a in env.agents])
			dones_array = np.array([dones[a] for a in env.agents])

			replay_buffer.push(obs_concat, actions_concat, rewards_array, next_obs_concat, dones_array)
		except:
			None
		
		obs = next_obs
		done = dones

		for agent in agents.values():
			agent.update(replay_buffer, agents)

	print(f'Episode {episode} completed. Reward {env.forward_reward}.')
	
	# rewards
	recompensas.append(retorno)
	# reward medio
	avg_rewards.append(np.mean(recompensas[-2*PERIOD:]))
	
	if not (episode % PERIOD):
		plt.figure(1)
		plt.clf()
		plt.subplot(211)
		plt.plot(avg_rewards, 'b', linewidth=2)
		plt.plot(recompensas, 'r', alpha=0.3)
		plt.xlabel('Episódios')
		plt.ylabel('Recompensa')
		plt.subplot(212)
		#
		plt.plot(acoes)
		plt.ylim([-1.1, 1.1])
		plt.show()
		plt.pause(.1)

plt.ioff()
plt.show()

env.close()
