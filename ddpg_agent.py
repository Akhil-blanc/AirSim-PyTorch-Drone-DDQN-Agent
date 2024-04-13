from ddpg_models import DDPG_Actor, DDPG_Critic, OrnsteinUhlenbeckActionNoise
import math
import random
from collections import deque
import airsim
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from setuptools import glob
from env import DroneEnv
# from torch.utils.tensorboard import SummaryWriter
import time
from prioritized_memory import Memory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DDPG_Agent:
    def __init__(self, useDepth=False):
        self.useDepth = useDepth
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 30000
        self.gamma = 0.8
        self.tau = 0.05 #Check standard values for tau
        self.learning_rate = 0.001
        self.batch_size = 512
        self.memory = Memory(10000)
        self.max_episodes = 10000
        self.save_interval = 2
        self.test_interval = 10
        self.network_update_interval = 10
        self.episode = -1
        self.steps_done = 0
        self.max_steps = 34

        self.actor = DDPG_Actor()
        self.initialize_weights(self.actor)
        self.critic = DDPG_Critic()
        self.initialize_weights(self.critic)

        self.actor_target = DDPG_Actor()
        self.initialize_weights(self.actor_target)
        self.critic_target = DDPG_Critic()
        self.initialize_weights(self.critic_target)

        self.env = DroneEnv(useDepth)

        if torch.cuda.is_available():
            print('Using device:', device)
            print(torch.cuda.get_device_name(0))
        else:
            print("Using CPU")

        # LOGGING
        cwd = os.getcwd()
        self.save_dir = os.path.join(cwd, "saved models")
        if not os.path.exists(self.save_dir):
            os.mkdir("saved models")
        if not os.path.exists(os.path.join(cwd, "videos")):
            os.mkdir("videos")

        if torch.cuda.is_available():
            self.actor = self.actor.to(device)  # to use GPU
            self.actor_target = self.actor_target.to(device)  # to use GPU
            self.critic = self.critic.to(device)  # to use GPU
            self.critic_target = self.critic_target.to(device)  # to use GPU
            
        self.save_dir = os.path.join(os.getcwd(), "ddpg")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        files = glob.glob(self.save_dir + '/*.pt')
        if len(files) > 0:
            files.sort(key=os.path.getmtime)
            file = files[-1]
            checkpoint = torch.load(file)
            self.policy.load_state_dict(checkpoint['state_dict'])
            self.episode = checkpoint['episode']
            self.steps_done = checkpoint['steps_done']
            self.updateNetworks()
            print("Saved parameters loaded"
                "\nModel: ", file,
                "\nSteps done: ", self.steps_done,
                "\nEpisode: ", self.episode)
        else:
            if os.path.exists("log.txt"):
                open('log.txt', 'w').close()
            if os.path.exists("last_episode.txt"):
                open('last_episode.txt', 'w').close()
            if os.path.exists("saved_model_params.txt"):
                open('saved_model_params.txt', 'w').close()

    def initialize_weights(self, model):
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0., std=0.1)
                torch.nn.init.constant_(m.bias, 0.1)

    def transformToTensor(self, img):
        tensor = torch.FloatTensor(img).to(device)
        tensor = tensor.unsqueeze(0)
        tensor = tensor.unsqueeze(0)
        tensor = tensor.float()
        return tensor

    def convert_size(self, size_bytes):
        if size_bytes == 0:
            return "0B"
        size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return "%s %s" % (s, size_name[i])
    
    def append_sample(self, state, action, reward, next_state):
        next_state = self.transformToTensor(next_state)
        next_state_actions = self.actor_target(next_state) #mu target given next state
        next_q_values = self.critic_target(next_state, next_state_actions) #Q target given next state and mu target
        expected_q_values = reward + (self.gamma * next_q_values)
        current_q_values = self.critic(state, action)
        error = torch.abs(current_q_values - expected_q_values).detach().cpu().numpy()
        self.memory.add(error, state, action, reward, next_state)

    def learn(self):
        if self.memory.tree.n_entries < self.batch_size:
            return

        states, actions, rewards, next_states, idxs, is_weights = self.memory.sample(self.batch_size)

        states = tuple(states)
        next_states = tuple(next_states)

        states = torch.cat(states)
        actions = torch.tensor(actions, dtype=torch.float).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(device)
        next_states = torch.cat(next_states)

        # Critic update
        next_state_actions = self.actor_target(next_states) #mu target given next state
        next_q_values = self.critic_target(next_states, next_state_actions) #Q target given next state and mu target
        expected_q_values = rewards + (self.gamma * next_q_values)
        current_q_values = self.critic(states, actions)

        errors = torch.abs(current_q_values.squeeze() - expected_q_values.squeeze()).detach().cpu().numpy()

        for i in range(self.batch_size):
            idx = idxs[i]
            self.memory.update(idx, errors[i])

        critic_loss = F.mse_loss(current_q_values.squeeze(), expected_q_values.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        policy_actions = self.actor(states)
        actor_loss = -self.critic(states, policy_actions).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def train(self):
        print("Starting...")

        score_history = []
        reward_history = []

        if self.episode == -1:
            self.episode = 1

        for e in range(1, self.max_episodes + 1):
            start = time.time()
            state, _ = self.env.reset()
            steps = 0
            score = 0
            while True:
                state = self.transformToTensor(state)

                # action = self.act(state)
                action = (torch.tensor(self.actor(state))).squeeze(dim=0)
                # print(action.squeeze(dim=0))
                action_noise = OrnsteinUhlenbeckActionNoise(action.shape[0])
                action = torch.cat((action, (action_noise.sample()).to(device)), dim=0)
                next_state, reward, done, _ = self.env.continuous_action_step(action)

                if steps == self.max_steps:
                    done = 1

                #self.memorize(state, action, reward, next_state)
                self.append_sample(state, action, reward, next_state)
                self.learn()

                state = next_state
                steps += 1
                score += reward
                if done:
                    print("----------------------------------------------------------------------------------------")
                    if self.memory.tree.n_entries < self.batch_size:
                        print("Training will start after ", self.batch_size - self.memory.tree.n_entries, " steps.")
                        break

                    print(
                        "episode:{0}, reward: {1}, mean reward: {2}, score: {3}, epsilon: {4}, total steps: {5}".format(
                            self.episode, reward, round(score / steps, 2), score, self.eps_threshold, self.steps_done))
                    score_history.append(score)
                    reward_history.append(reward)
                    with open('log.txt', 'a') as file:
                        file.write(
                            "episode:{0}, reward: {1}, mean reward: {2}, score: {3}, epsilon: {4}, total steps: {5}\n".format(
                                self.episode, reward, round(score / steps, 2), score, self.eps_threshold,
                                self.steps_done))

                    if torch.cuda.is_available():
                        print('Total Memory:', self.convert_size(torch.cuda.get_device_properties(0).total_memory))
                        print('Allocated Memory:', self.convert_size(torch.cuda.memory_allocated(0)))
                        print('Cached Memory:', self.convert_size(torch.cuda.memory_reserved(0)))
                        print('Free Memory:', self.convert_size(torch.cuda.get_device_properties(0).total_memory - (
                                torch.cuda.max_memory_allocated() + torch.cuda.max_memory_reserved())))

                        # tensorboard --logdir=runs
                        memory_usage_allocated = np.float64(round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1))
                        memory_usage_cached = np.float64(round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1))

                    # save checkpoint
                    if self.episode % self.save_interval == 0:
                        checkpoint = {
                            'episode': self.episode,
                            'steps_done': self.steps_done,
                            'state_dict': self.policy.state_dict()
                        }
                        torch.save(checkpoint, self.save_dir + '//EPISODE{}.pt'.format(self.episode))

                    if self.episode % self.network_update_interval == 0:
                        self.updateNetworks()

                    self.episode += 1
                    end = time.time()
                    stopWatch = end - start
                    print("Episode is done, episode time: ", stopWatch)

                    if self.episode % self.test_interval == 0:
                        self.test()

                    break
        # writer.close()

    def test(self):
        state, _ = self.env.reset()
        steps = 0
        score = 0
        while True:
            state = self.transformToTensor(state)
            action = self.actor(state)
            next_state, reward, done, _ = self.env.continuous_action_step(action)
            state = next_state
            steps += 1
            score += reward
            if done:
                print("----------------------------------------------------------------------------------------")
                print("Test episode, reward: {0}, mean reward: {1}, score: {2}".format(reward, round(score / steps, 2), score))
                break