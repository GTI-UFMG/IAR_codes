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
import numpy as np
from functools import partial
import pygame

# Globais
NACTIONS = 9
MAX_STEPS = 100
SCREEN_SIZE = 500

########################################
# classe do mapa
########################################
class Maze(gym.Env):
    ########################################
    # construtor
    def __init__(self, xlim=np.array([0.0, 10.0]), ylim=np.array([0.0, 10.0]), res=0.4, img='labirinto.png', alvo=np.array([9.5, 9.5]), render=False):

        # salva o tamanho geometrico da imagem em metros
        self.xlim = xlim
        self.ylim = ylim

        # resolucao
        self.res = res

        ns = int(np.max([np.abs(np.diff(self.xlim)), np.abs(np.diff(self.ylim))])/res)
        self.num_states = [ns, ns]
        
        # espaco de atuacao
        self.action_space = spaces.Discrete(NACTIONS)

        # converte estados continuos em discretos
        lower_bounds = [self.xlim[0], self.ylim[0]]
        upper_bounds = [self.xlim[1], self.ylim[1]]
        self.get_state = partial(self.obs_to_state, self.num_states, lower_bounds, upper_bounds)

        # alvo
        self.alvo = alvo
        
        # renderizar
        self.render_env = render
        
        # cria mapa
        pygame.init()
        pygame.display.set_mode((1, 1))  # Inicializa com uma janelinha mínima
        self.init2D(img)
        pygame.quit()
        
        # Inicializa pygame se necessário
        if self.render_env:
            pygame.init()
            self.screen_size = (SCREEN_SIZE, SCREEN_SIZE)
            self.screen = pygame.display.set_mode(self.screen_size)
            self.clock = pygame.time.Clock()
            pygame.display.set_caption("Labirinto")
        
            # Converte o mapa para uma imagem pygame
            mapa_norm = self.mapa.astype(np.uint8)  # inverte e escala pra 0-255
            mapa_rgb = np.stack([mapa_norm]*3, axis=-1)  # gray -> RGB
            mapa_surface = pygame.surfarray.make_surface(np.transpose(mapa_rgb, (1, 0, 2)))
            self.map_surface = pygame.transform.scale(mapa_surface, self.screen_size)

    ########################################
    # ambientes em 2D
    def init2D(self, image):
        
        # Carrega a imagem em escala de cinza
        I_surface = pygame.image.load(image).convert()
        I_array = pygame.surfarray.pixels3d(I_surface)
        
        # Transpõe para (altura, largura, canais), como no OpenCV
        I_array = I_array.transpose(1, 0, 2)

        # Converte para escala de cinza (média dos canais RGB)
        I_gray = np.mean(I_array, axis=2).astype(np.uint8)

        # Pega o número de linhas e colunas
        self.nrow = I_gray.shape[0]
        self.ncol = I_gray.shape[1]

        # Binariza a imagem (limiar 127)
        I_binary = np.where(I_gray > 127, 255, 0).astype(np.uint8)

        # Inverte a imagem no eixo Y
        self.mapa = np.flipud(I_binary)

        # Parâmetros de conversão (como no original)
        self.mx = float(self.ncol) / float(self.xlim[1] - self.xlim[0])
        self.my = float(self.nrow) / float(self.ylim[1] - self.ylim[0])
        
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
        
        # trajetoria
        self.traj = [self.p]

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
        if ( (self.xlim[0] <= nextp[0] <= self.xlim[1]) and (self.ylim[0] <= nextp[1] <= self.ylim[1]) ):
            self.p = nextp
            
        # trajetoria
        self.traj.append(self.p)
         
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
    # Mapeamento de coordenadas reais para pixels
    def world_to_screen(self, pos):
        x = int((pos[0] - self.xlim[0]) / (self.xlim[1] - self.xlim[0]) * self.screen_size[0])
        y = int(self.screen_size[1] - (pos[1] - self.ylim[0]) / (self.ylim[1] - self.ylim[0]) * self.screen_size[1])
        return (x, y)
        
    ########################################
    # desenha a imagem distorcida em metros
    def render(self, Q, arrow_size=0.5, target_size=5, robot_size=10):
        
        if not self.render_env:
            return
        
        # Trata eventos para manter a janela viva
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()        
        
        # Desenha o mapa
        self.screen.blit(self.map_surface, (0, 0))
        
        # Desenha o alvo (um X)
        alvo_pos = self.world_to_screen(self.alvo)
        pygame.draw.line(self.screen, (0, 200, 0), (alvo_pos[0] - target_size, alvo_pos[1] - target_size), (alvo_pos[0] + target_size, alvo_pos[1] + target_size), 5)
        pygame.draw.line(self.screen, (0, 200, 0), (alvo_pos[0] - target_size, alvo_pos[1] + target_size), (alvo_pos[0] + target_size, alvo_pos[1] - target_size), 5)
        
        # Desenha trajetoria do robo
        for p in self.traj:
            pygame.draw.rect(self.screen, (155, 0, 200), (*self.world_to_screen(p), 0.5*robot_size, 0.5*robot_size))
        # Desenha o robô
        pygame.draw.rect(self.screen, (0, 0, 255), (*self.world_to_screen(self.p), robot_size, robot_size))
        
        # Desenha o campo vetorial
        m = self.num_states[0]
        xm = np.linspace(self.xlim[0], self.xlim[1], m)
        ym = np.linspace(self.ylim[0], self.ylim[1], m)
        for x in xm:
            for y in ym:
                # verifica colisao
                if self.collision((x, y)):
                    continue
                # desenha a seta
                S = self.get_state(np.array([x, y]))
                u = arrow_size*self.actionU(Q[S, :].argmax())
                start = self.world_to_screen([x, y])
                end = self.world_to_screen([x + u[0], y + u[1]])
                if np.linalg.norm(u) > 0:
                    self.draw_arrow(self.screen, (0, 100, 150), start, end)

        # Atualiza a tela
        pygame.display.flip()
        self.clock.tick(30)  # FPS
     
    ########################################
    def draw_arrow(self, surface, color, start, end, width=2, head_size=3):
        # Linha principal
        pygame.draw.line(surface, color, start, end, width)

        # Vetor da seta
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        angle = np.arctan2(dy, dx)

        # Cálculo da cabeça da seta (duas linhas formando um "V")
        sin_a = np.sin(angle)
        cos_a = np.cos(angle)

        left = (
            end[0] - head_size * cos_a + head_size * sin_a,
            end[1] - head_size * sin_a - head_size * cos_a
        )
        right = (
            end[0] - head_size * cos_a - head_size * sin_a,
            end[1] - head_size * sin_a + head_size * cos_a
        )

        pygame.draw.line(surface, color, end, left, width)
        pygame.draw.line(surface, color, end, right, width)

    ########################################
    def __del__(self):
        None