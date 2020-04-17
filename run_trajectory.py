import numpy as np
import tensorflow.compat.v1 as tf
from envs.env import ArmEnv
from contextual_policy_search.trajectory import TRAJECTROY as tra
tf.disable_v2_behavior()
print("import finished")


env = ArmEnv()
env.reset()
ep_reward = 0
kp = np.array([0.007, 0.007, 0.03, 0.0001, 0.0001, 0.005])
kd = np.array([0.01, 0.01, 0.05, 0.0001, 0.0001, 0.005])
K = [kp, kd]

trajectory = tra()
memory = trajectory.pd_trajectory(env, K)


