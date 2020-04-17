import numpy as np
import time


class TRAJECTROY:
    def __init__(self):
        self.memory = []
        self.s = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.action_1 = np.array([0., 0., 0., 0., 0., 0.])
        self.yk = np.array([0., 0., 0., 0., 0., 0.])
        self.yk_1 = np.array([0., 0., 0., 0., 0., 0.])
        self.yk_2 = np.array([0., 0., 0., 0., 0., 0.])
        self.kp = np.array([0., 0., 0., 0., 0., 0.])
        self.kd = np.array([0., 0., 0., 0., 0., 0.])
        self.done = False
        self.safe = True
        self.max_steps = 80

    def caculate_action(self):
        print('state: ', self.s)
        self.yk = np.array([self.s[6], self.s[7], self.s[8], self.s[9], self.s[10], self.s[11]])
        self.yk = self.yk - np.array([0, 0, -15, 0, 0, 0])
        print('yk:', self.yk)
        # discrete PD algorithm
        print(self.kp * (self.yk - self.yk_1))
        print(self.kd * (self.yk - 2 * self.yk_1 + self.yk_2))
        action = self.action_1 + self.kp * (self.yk - self.yk_1) + self.kd * (self.yk - 2 * self.yk_1 + self.yk_2)
        # renew variables
        for i in range(6):
            action[i] = round(action[i], 4)
            self.action_1[i] = action[i]
            self.yk_2[i] = self.yk_1[i]
            self.yk_1[i] = self.yk[i]
        # process before output
        if action[2] < 0:
            action[2] = 0.000
        print('action:', action)
        # mm back to m
        action[0] /= 1000
        action[1] /= 1000
        action[2] /= 1000
        return action

    def pd_trajectory(self, env, k):
        self.kp = k[0]
        self.kd = k[1]
        for k in range(self.max_steps):
            print('step:', k)
            # take actions
            action = self.caculate_action()
            s_1, r, self.done, self.safe = env.directstep(action)
            print('R:', r, '  done?', self.done, '  safe?', self.safe)
            if not self.safe:
                break
            if self.done:
                break
            # store trajectory
            transition = np.hstack((self.s, action, [r]))
            self.memory.append(transition)
            # renew variables
            for i in range(12):
                self.s[i] = s_1[i]
            print("")
            print("")

        return self.memory
