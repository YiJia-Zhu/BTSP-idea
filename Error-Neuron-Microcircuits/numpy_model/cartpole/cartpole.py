#!/usr/bin/env python3
import numpy as np
import scipy


class Cartpole():
    def __init__(self, dt=0.02, rl_enable=False, random=False):
        self.g = 9.8  # gravity
        self.m_c = 1.0  # mass of cart
        self.m_p = 0.1  # mass of pole
        self.l = 0.5  # half the length of the pole
        self.dt = dt  # time step
        self.force_mag = 0.5  # force magnitude
        self.rand_bounds = 2.0   # random initial state bounds (0.5 rad = 28 deg, 1 rad = 57 deg)
        self.x_lim = 2.40           # limits where to reset cartpole
        self.theta_lim = 70 * 2 * np.pi / 360 # needs to be more than the init
        self.rl_enable = rl_enable
        self.random = random
        self.reset()

    def reset(self):
        if self.random:
            self.x, self.x_dot, self.theta, self.theta_dot = (np.random.rand(4) - 0.5) * self.rand_bounds
        else:
            self.x, self.x_dot, self.theta, self.theta_dot = [0.0, 0.0, 0.01, 0.0]
        return self.get_state()

    def get_state(self):
        return self.x, self.x_dot, self.theta, self.theta_dot

    def step(self, action):
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        # Unpack state
        x, x_dot, theta, theta_dot = self.get_state()

        # Apply action
        if self.rl_enable:
            force = self.force_mag if action > 0 else -self.force_mag
        else:
            force = action  # force applied to the cart
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        temp = (force + self.m_p * self.l * theta_dot ** 2 * sintheta) / (self.m_c + self.m_p)
        thetaacc = (self.g * sintheta - costheta * temp) / (self.l * \
                                    (4.0 / 3.0 - self.m_p * costheta ** 2 / (self.m_c + self.m_p)))  # angular acceleration
        xacc = temp - self.m_p * self.l * thetaacc * costheta / (self.m_c + self.m_p)  # acceleration of the cart

        # Update state
        x = x + self.dt * self.x_dot
        x_dot = x_dot + self.dt * xacc
        theta = theta + self.dt * theta_dot
        theta_dot = theta_dot + self.dt * thetaacc

        self.x, self.x_dot, self.theta, self.theta_dot = x, x_dot, theta, theta_dot

        # Check if cartpole has fallen
        failed = x < -self.x_lim or x > self.x_lim or theta < -self.theta_lim or theta > self.theta_lim
        #failed = 0

        return self.get_state(), failed

class PDController():
    def __init__(self, af, kp=10, ki=0, kd=1, dt=0.02):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.af = af
        self.reset()

    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    def reset(self):
        self.prev_error = 0
        self.integral = 0

    def pd_action(self, output, target):
        error = output - target
        derivative = (error - self.prev_error)
        self.integral += error
        action = self.kp * error + self.ki * self.integral + self.kd * derivative
        action = self.af(action)
        return action, error

class lqrController():
    def __init__(self, A, B, Q, R):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.reset()

    def reset(self):
        self.P = np.zeros_like(self.Q)
        self.K = np.zeros_like(self.B)

    def lqr(self):
        # solve DARE
        #P = np.matrix(scipy.linalg.solve_discrete_are(self.A, self.B, self.Q, self.R))
        P = np.matrix(scipy.linalg.solve_continuous_are(self.A, self.B, self.Q, self.R))

        # compute the LQR gain
        #K = np.matrix(scipy.linalg.inv(self.B.T @ P @ self.B + self.R) @ (self.B.T @ P @ self.A))
        K = np.dot(np.linalg.inv(self.R), np.dot(self.B.T, P))

        return K

    def lqr_action(self, state):
        state_a = np.array(state).reshape(-1, 1)
        action = -np.dot(self.K, state_a)
        return action[0, 0]

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    sim_time = 10000

    # simulate and plot dynamics
    #seed = 42
    #np.random.seed(seed)
    cartpole = Cartpole()
    state_log = []
    state_log.append(cartpole.get_state())
    for _ in range(sim_time):
        action = np.random.randint(-10, 10)
        state, done = cartpole.step(action)
        state_log.append(state)
        if done:
            cartpole.reset()
            state_log.append(np.ones(len(state)) * 5)
    state_log = np.array(state_log)
    state_names = [r"$x$", r"$\dot{x}$", r"$\theta$", r"$\dot{\theta}$"]
    plt.figure()
    for i in range(4):
        plt.plot(state_log[:, i], label=state_names[i])
    plt.title("Cartpole dynamics")
    plt.legend()

    # simulate cartpole with pd controller and plot
    cartpole = Cartpole()
    state_log = []
    state_log.append(cartpole.get_state())
    controller = PDController(af=PDController.linear, kp=10, ki=0, kd=2, dt=cartpole.dt)
    action_log = []
    theta_target = 0
    for _ in range(sim_time):
        x, x_dot, theta, theta_dot = cartpole.get_state()
        action, _ = controller.pd_action(theta, theta_target)
        state, done = cartpole.step(action)
        state_log.append(state)
        action_log.append(action)
        if done:
            cartpole.reset()
    state_log = np.array(state_log)
    plt.figure()
    for i in range(4):
        plt.plot(state_log[:, i], label=state_names[i])
    plt.plot(action_log, label="action", alpha=0.5, ls="--")
    plt.title("Cartpole dynamics with PD controller")
    plt.legend()

    # simulate cartpole with lqr controller and plot
    rl_enable = True
    manual_k_enable = False
    cartpole = Cartpole(random=True, rl_enable=rl_enable)
    state_log = []
    target_state = np.array([0., 0., 0., 0.])
    manual_K = np.array([3, 12, 100, 40])
    state_log.append(cartpole.get_state())
    # https://towardsdatascience.com/comparing-optimal-control-and-reinforcement-learning-using-the-cart-pole-swing-up-openai-gym-772636bc48f4
    temp_a = cartpole.g/(cartpole.l*(4.0/3 - cartpole.m_p/(cartpole.m_p + cartpole.m_c)))
    A = np.array([[0, 1, 0, 0],
                  [0, 1, temp_a, 0],
                  [0, 0, 0, 1],
                  [0, 0, temp_a, 1]])
    temp_b = -1/(cartpole.l*(4.0/3 - cartpole.m_p/(cartpole.m_p + cartpole.m_c)))
    B = np.array([[0],
                  [1/(cartpole.m_c + cartpole.m_p)],
                  [0],
                  [temp_b]
                  ])
    Q = np.eye(4) * np.array([1., 1., 10., 100.])
    R = np.eye(1) * 1
    controller = lqrController(A, B, Q, R)
    controller.K = controller.lqr()
    action_log = []
    for _ in range(1000):
        state = cartpole.get_state()
        if manual_k_enable:
            action = -np.sum(manual_K * np.array(target_state - state))
        else:
            action = controller.lqr_action(state)
        if rl_enable:
            action = 1 if action > 0 else 0
        state, done = cartpole.step(action)
        state_log.append(state)
        action_log.append(action)
        if done:
            cartpole.reset()
    state_log = np.array(state_log)
    plt.figure()
    for i in range(4):
        plt.plot(state_log[:, i], label=state_names[i])
    plt.plot(action_log, label="action", alpha=0.5, ls="--")
    # K in 1d
    lqr_k = [round(controller.K[0, i], 2) for i in range(controller.K.shape[1])]
    if rl_enable:
        rl_enable_str = " with RL enable"
        k_str = ""
    else:
        rl_enable_str = ""
        k_str = "K = " + str(lqr_k)
    plt.title("Cartpole dynamics with LQR controller" + rl_enable_str + ". " + k_str )
    plt.legend()

    plt.show()
