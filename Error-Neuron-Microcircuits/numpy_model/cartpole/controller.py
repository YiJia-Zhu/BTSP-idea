import numpy as np
import matplotlib.pyplot as plt
import logging

from cartpole.cartpole import Cartpole

def test_model_on_cartpole(mc, path=None):

	logging.info(f"Seed {mc.seed}: testing on cartpole")

	# simulate cartpole with microcircuit controller and plot
	seed = 42
	dt_cartpole = mc.dt
	T = 200  # msecs
	sim_time = int(T / dt_cartpole)
	random = True
	n_subsims = 10
	subsim_time = int(sim_time / n_subsims)
	use_x = True # use the carpole position
	use_x_dot = True
	use_theta = True
	use_theta_dot = True

	cartpole = Cartpole(dt=dt_cartpole, rl_enable=False, random=random)
	state_log = []
	state_log.append(cartpole.get_state())
	action_log = []
	theta_target = 0
	action_delay = 0 # dts
	tt_log = []
	penalty_log = []

	for tt in range(1, sim_time):
		if tt % subsim_time == 0:
			cartpole.reset()
			tt_log.append(tt)
		x, x_dot, theta, theta_dot = cartpole.get_state()
		if use_x:
			x_a = np.array([x]).reshape(len([x]), 1)
			error_cp_array = x_a
		if use_x_dot:
			x_dot_a = np.array([x_dot]).reshape(len([x]), 1)
			error_cp_array = np.concatenate((error_cp_array, x_dot_a), axis=1)
		if use_theta:
			x_theta_a = np.array([theta]).reshape(len([x]), 1)
			error_cp_array = np.concatenate((error_cp_array, x_theta_a), axis=1)
		if use_theta_dot:
			x_theta_dot_a = np.array([theta_dot]).reshape(len([x]), 1)
			error_cp_array = np.concatenate((error_cp_array, x_theta_dot_a), axis=1)
				
		error_time_series = np.repeat(error_cp_array, int(4), axis=0)
		for e in error_time_series:
			mc.evolve_system(r0=e, 
										 learn_weights=False, 
										 learn_bw_weights=False)
		emc_action = mc.rP_breve[-1][0]  # this IS correct
		emc_action *= mc.target_rescale_factor
		#print(action)
		state, failed = cartpole.step(emc_action)        
		state_log.append(state)
		action_log.append(emc_action)
		penalty_log.append(1 if failed else 0)
		if failed:
			cartpole.reset()

	state_log = np.array(state_log)
	penalty_log = np.array(penalty_log)

	if hasattr(mc, 'state_log'):
		mc.state_log = np.append(mc.state_log, state_log.copy(), axis=0)
	else:
		mc.state_log = state_log.copy()
	if hasattr(mc, 'action_log'):
		mc.action_log = np.append(mc.action_log, action_log.copy(), axis=0)
	else:
		mc.action_log = action_log.copy()
	if hasattr(mc, 'penalty_log'):
		mc.penalty_log = np.append(mc.penalty_log, penalty_log.copy(), axis=0)
	else:
		mc.penalty_log = penalty_log.copy()

	return mc
