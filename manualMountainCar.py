import gym
import time

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

def setEnv(name):
    env = gym.make(name)
    spec = gym.spec(name)
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")
    print(f"Max Episode Steps: {spec.max_episode_steps}")
    print(f"Nondeterministic: {spec.nondeterministic}")
    print(f"Reward Range: {env.reward_range}")
    print(f"Reward Threshold: {spec.reward_threshold}")
    print(f"env.unwrapped",env.unwrapped)

    

    return env

env = setEnv("MountainCar-v0")
#A single episode
# load and reset the init stage
observation = env.reset()


print ("observation space", env.observation_space.low)
#neutral
action = 1
# default to 2000 steps
for _ in range(200):
  env.render()

  #action = env.action_space.sample() # your agent here (this takes random actions)
  observation, reward, done, info = env.step(action)

  #Manual way to solve the problem, still have some random elements to it

  # going forward accl right (2) if positive velocity ()
  if observation[1] > 0:
    action = 2
  #going backward accl left (0)
  else:
    action = 0
  
  print(f"iteration {_}, Step {action}, Velocity={observation[1]} Position={observation[0]}, Reward={reward}, Done={done}")
  # flag position 
  if done:
    break 
time.sleep(10)
env.close()

print(gym.__version__)
