
from ale_py import ALEInterface
#from ale_py.roms import Breakout
import gym

ale = ALEInterface() 

#ale.loadROM(Breakout)

print(gym.__version__)

def setEnv(name):
    env = gym.make(name,     
        obs_type='rgb',                   # ram | rgb | grayscale
        frameskip=5,                     # frame skip
        mode=0,                           # game mode, see Machado et al. 2018
        difficulty=0,                     # game difficulty, see Machado et al. 2018
        repeat_action_probability=0.25,   # Sticky action probability
        full_action_space=True,           # Use all actions
        render_mode='human'                  # None | human | rgb_array
    )

    spec = gym.spec(name)
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")
    print(f"Max Episode Steps: {spec.max_episode_steps}")
    print(f"Nondeterministic: {spec.nondeterministic}")
    print(f"Reward Range: {env.reward_range}")
    print(f"Reward Threshold: {spec.reward_threshold}")
    print(f"Env space: {env.action_space.n}")
    return env

# env = gym.make('ALE/Breakout-v5',
#     obs_type='rgb',                   # ram | rgb | grayscale
#     frameskip=5,                     # frame skip
#     mode=0,                           # game mode, see Machado et al. 2018
#     difficulty=0,                     # game difficulty, see Machado et al. 2018
#     repeat_action_probability=0.25,   # Sticky action probability
#     full_action_space=True,           # Use all actions
#     render_mode='human'                  # None | human | rgb_array
# )
env = setEnv('ALE/Breakout-v5')
observation = env.reset()

while True:
  
    #env.render()
    
    #your agent goes here
    action = env.action_space.sample() 
         
    observation, reward, done, info = env.step(action) 
    print(f"iteration, Step {action}, Reward={reward}, Done={done}")
   
        
    if done: 
      break;
            
env.close()
