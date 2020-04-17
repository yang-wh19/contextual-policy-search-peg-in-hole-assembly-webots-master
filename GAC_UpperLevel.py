import numpy as np
from envs.env import ArmEnv
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from GAC_Lowerlevel import trainning as lowlevel_trainning


#####################  Hyper Parameters  #####################
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32

np.random.seed(2)
tf.set_random_seed(2)  # reproducible


###########################  GAC  ############################
class GAC(object):
    def __init__(self, a_dim, s_dim, a_bound):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)

        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        # 将具有各自标签的网络参数归类到各自的列表里

        td_error = tf.losses.mean_squared_error(labels=self.R, predictions=q)
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(-tf.reduce_mean(q), var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.a, feed_dict={self.S: s[np.newaxis, :]})[0]

    def learn_a(self):
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        # 从1-MEMORY_CAPACITY中随机取size个数，组成一个数组
        bt = self.memory[indices, :]
        # 矩阵memory中第（数组indices的每一个元素）行组成一个新矩阵
        bs = bt[:, :self.s_dim]
        # 矩阵bt的前s_dim列组成新矩阵
        self.sess.run(self.atrain, {self.S: bs})

    def learn_c(self):
        # 从memory中取出BATCH_SIZE组参数进行学习
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        # 从1-MEMORY_CAPACITY中随机取size个数，组成一个数组
        bt = self.memory[indices, :]
        # 矩阵memory中第（数组indices的每一个元素）行组成一个新矩阵
        bs = bt[:, :self.s_dim]
        # 矩阵bt的前s_dim列组成新矩阵
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, - 1:]

        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r]))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        # 方法1
        with tf.variable_scope(scope):
            layer1 = tf.layers.dense(s, 40, activation=tf.nn.relu,
                                     kernel_initializer=tf.random_normal_initializer(0., .1),
                                     bias_initializer=tf.constant_initializer(0.1), name='layer1', trainable=trainable)
            layer2 = tf.layers.dense(layer1, 30, activation=tf.nn.relu,
                                     kernel_initializer=tf.random_normal_initializer(0., .1),
                                     bias_initializer=tf.constant_initializer(0.1), name='layer2', trainable=trainable)
            a = tf.layers.dense(layer2, self.a_dim, activation=tf.nn.tanh,
                                kernel_initializer=tf.random_normal_initializer(0., .1),
                                bias_initializer=tf.constant_initializer(0.1), name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        # 方法2
        with tf.variable_scope(scope):
            n_l1 = 40
            n_l2 = 30
            w1_1 = tf.Variable(tf.random.uniform([self.s_dim, n_l1], -1, 1), name='w1_1', trainable=trainable)
            w1_2s = tf.Variable(tf.random.uniform([n_l1, n_l2], -1, 1), name='w1_2s', trainable=trainable)
            w1_2a = tf.Variable(tf.random.uniform([self.a_dim, n_l2], -1, 1), name='w1_2a',
                                trainable=trainable)
            b1 = tf.Variable(tf.random.uniform([1, n_l1], -1, 1), name='b1')
            b2 = tf.Variable(tf.random.uniform([1, n_l2], -1, 1), name='b2')
            layer1 = tf.nn.relu(tf.matmul(s, w1_1) + b1)
            layer2 = tf.nn.relu(tf.matmul(layer1, w1_2s) + tf.matmul(a, w1_2a) + b2)
            q = tf.layers.dense(layer2, 1)
            return q  # Q(s,a)


######################  GAC Parameters  ######################
env = ArmEnv()
T = 1000
T_random = 500    # ？？？？暂定为500 ？？？？
K = 8    # ？？？？重复学习次数暂定为8 ？？？？
MAX_EP_STEPS = 200
environment_dim = 6    # ？？？？暂定为工件的坐标 ？？？？
action_dim = 2    # Kp和Kd
action_bound = 1     # ？？？？Kp和Kd的范围暂定为1 ？？？？
gac = GAC(action_dim, environment_dim, action_bound)
var = 1  # ？？？？暂定为1 ？？？？
REWARD = []


###########################  Main  ###########################
for t in range(T):
    env.reset()
    environment = env.init_state

    action = gac.choose_action(environment)
    if t <= T_random:
        action = np.clip(np.random.normal(action, var), -action_bound, action_bound)

    R = lowlevel_trainning(env, action, MAX_EP_STEPS)
    gac.store_transition(environment, action, R)
    # Sample and learn
    if gac.pointer > MEMORY_CAPACITY:
        var *= .9995  # decay the action randomness
        for k in range(K):
            gac.learn_c()
        for k in range(K):
            gac.learn_a()

