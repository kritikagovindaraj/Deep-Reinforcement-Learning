import gym
import time
env = gym.make('Blackjack-v0')
env.reset()
for _ in range(1000):
    env.render()
time.sleep(5)
env.close()
