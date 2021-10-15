import gym
import time


import numpy as np
import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common


# Agent - The agent is an entity that exists in an environment that takes actions to affect the state of the environment, to receive rewards.
# Environment - The environment is the universe that the agent exists in. The environment is always in a specific state that is changed by the actions of the agent.
# Actions - Steps that can be performed by the agent to alter the environment
# Step - A step occurs each time that the agent performs an action and potentially changes the environment state.
# Episode - A chain of steps that ultimately culminates in the environment entering a terminal state.
# Epoch - A training iteration of the agent that contains some number of episodes.
# Terminal State - A state in which further actions do not make sense. In many environments, a terminal state occurs when the agent has one, lost, or the environment exceeding the maximum number of steps.

#https://github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py

# class MountainCarEnv(gym.Env):
#     """
#     Description:
#         The agent (a car) is started at the bottom of a valley. For any given
#         state the agent may choose to accelerate to the left, right or cease
#         any acceleration.
#     Source:
#         The environment appeared first in Andrew Moore's PhD Thesis (1990).
#     Observation:
#         Type: Box(2)
#         Num    Observation               Min            Max
#         0      Car Position              -1.2           0.6
#         1      Car Velocity              -0.07          0.07
#     Actions:
#         Type: Discrete(3)
#         Num    Action
#         0      Accelerate to the Left
#         1      Don't accelerate
#         2      Accelerate to the Right
#         Note: This does not affect the amount of velocity affected by the
#         gravitational pull acting on the car.
#     Reward:
#          Reward of 0 is awarded if the agent reached the flag (position = 0.5)
#          on top of the mountain.
#          Reward of -1 is awarded if the position of the agent is less than 0.5.
#     Starting State:
#          The position of the car is assigned a uniform random value in
#          [-0.6 , -0.4].
#          The starting velocity of the car is always assigned to 0.
#     Episode Termination:
#          The car position is more than 0.5
#          Episode length is greater than 200 """

# Make the mountain car env
#env = gym.make("MountainCar-v0")

# How long should training run?
num_iterations = 3000
# How many initial random steps, before training start, to
# collect initial data.
initial_collect_steps = 1000   
# How many steps should we run each iteration to collect 
# data from.
collect_steps_per_iteration = 1 
# How much data should we store for training examples.
replay_buffer_max_length = 10000

batch_size = 128  
learning_rate = 1e-3 
# How often should the program provide an update.
log_interval = 100  

# How many episodes should the program use for each evaluation.
num_eval_episodes = 20
# How often should an evaluation occur.
eval_interval = 1000


def setEnv(name):
    env = suite_gym.load(name)
    # env = gym.make(name)
    # spec = gym.spec(name)
    # print(f"Action Space: {env.action_space}")
    # print(f"Observation Space: {env.observation_space}")
    # print(f"Max Episode Steps: {spec.max_episode_steps}")
    # print(f"Nondeterministic: {spec.nondeterministic}")
    # print(f"Reward Range: {env.reward_range}")
    # print(f"Reward Threshold: {spec.reward_threshold}")

    # print("Reward Specs:" , env.step)
    print('Reward Spec:', env.time_step_spec().reward)
    print("Observation Specs:", env.time_step_spec().observation)
    print('Action Spec:', env.action_spec())

    return env

env_name = "MountainCar-v0"
eval_py_env = setEnv(env_name)
train_env = tf_py_environment.TFPyEnvironment(setEnv(env_name))
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)



fc_layer_params = (100,)

q_net = q_network.QNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=fc_layer_params)

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

train_step_counter = tf.Variable(0)

agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter)

agent.initialize()

eval_policy = agent.policy
collect_policy = agent.collect_policy

random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())

example_environment = tf_py_environment.TFPyEnvironment(suite_gym.load(env_name))

time_step = example_environment.reset()
random_policy.action(time_step)

def compute_avg_return(environment, policy, num_episodes=10):

  total_return = 0.0
  for _ in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0

    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]



replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_max_length)


def collect_step(environment, policy, buffer):
  time_step = environment.current_time_step()
  action_step = policy.action(time_step)
  next_time_step = environment.step(action_step.action)
  traj = trajectory.from_transition(time_step, action_step, next_time_step)

  # Add trajectory to the replay buffer
  buffer.add_batch(traj)

def collect_data(env, policy, buffer, steps):
  for _ in range(steps):
    collect_step(env, policy, buffer)

collect_data(train_env, random_policy, replay_buffer, steps=100)

dataset = replay_buffer.as_dataset(
    num_parallel_calls=3, 
    sample_batch_size=batch_size, 
    num_steps=2).prefetch(3)

iterator = iter(dataset)




agent.train = common.function(agent.train)

# Reset the train step
agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, agent.policy, \
                                num_eval_episodes)
returns = [avg_return]

for _ in range(num_iterations):

  # Collect a few steps using collect_policy and save to the replay buffer.
  for _ in range(collect_steps_per_iteration):
    collect_step(train_env, agent.collect_policy, replay_buffer)

  # Sample a batch of data from the buffer and update the agent's network.
  experience, unused_info = next(iterator)
  train_loss = agent.train(experience).loss

  step = agent.train_step_counter.numpy()

  if step % log_interval == 0:
    print('step = {0}: loss = {1}'.format(step, train_loss))

  if step % eval_interval == 0:
    avg_return = compute_avg_return(eval_env, agent.policy, \
                                    num_eval_episodes)
    print('step = {0}: Average Return = {1}'.format(step, avg_return))
    returns.append(avg_return)
num_episodes = 15
policy = agent.policy 

for _ in range(num_episodes):
    time_step = eval_env.reset()
    eval_py_env.render()
    while not time_step.is_last():
        action_step = policy.action(time_step)
        time_step = eval_env.step(action_step.action)
        eval_py_env.render()
  