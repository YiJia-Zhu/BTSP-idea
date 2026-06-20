import numpy as np
from src.microcircuit import *
import time
import logging
from cartpole.controller import *

# takes a microcircuit object and runs it based on the signal given

def run(mc, learn_weights=True, learn_lat_weights=True, learn_bw_weights=True, teacher=False, rate_target=None):
	t_start = time.time()

	logging.info(f"Seed {mc.seed}: initialising recording")

	mc.init_record(rec_per_steps=mc.rec_per_steps,
		rec_MSE=mc.rec_MSE,
		rec_error=mc.rec_error,
		rec_input=mc.rec_input,
		rec_target=mc.rec_target,
		rec_WPP=mc.rec_WPP,
		rec_WIP=mc.rec_WIP,
		rec_BPP=mc.rec_BPP,
		rec_BII=mc.rec_BII,
		rec_BPI=mc.rec_BPI,
		rec_dWPP=mc.rec_dWPP,
		rec_dWIP=mc.rec_dWIP,
		rec_dBPP=mc.rec_dBPP,
		rec_dBII=mc.rec_dBII,
		rec_dBPI=mc.rec_dBPI,
		rec_uP=mc.rec_uP,
		rec_uI=mc.rec_uI,
		rec_uP_breve=mc.rec_uP_breve,
		rec_uI_breve=mc.rec_uI_breve,
		rec_rI_breve=mc.rec_rI_breve,
		rec_rP_breve=mc.rec_rP_breve,
		rec_rP_breve_HI=mc.rec_rP_breve_HI,
		rec_vbas=mc.rec_vbas,
		rec_vapi=mc.rec_vapi,
		rec_vapi_noise=mc.rec_vapi_noise,
		rec_epsilon=mc.rec_epsilon,
		rec_noise=mc.rec_noise,
		rec_epsilon_LO=mc.rec_epsilon_LO,
		rec_lat_mismatch=mc.rec_lat_mismatch)


	# pre-train to settle voltages
	logging.info(f"Seed {mc.seed}: running pre-training")
	mc = pre_train_model(mc, time=mc.settling_time/mc.dt)

	if teacher:
		# if the mc is the teacher, record target signal
		mc.target = []
		mc.target_testing = []
		mc.target_validation = []
		logging.info(f"Seed {mc.seed}: recording teacher (train and test sets)")
		# record train and val set
		mc = train_model(mc, epochs=1, learn_weights=False, \
		learn_lat_weights=False, learn_bw_weights=False, teacher=True, testing=False, rate_target=rate_target)
		# record test set
		mc = train_model(mc, epochs=1, learn_weights=False, \
		learn_lat_weights=False, learn_bw_weights=False, teacher=True, testing=True, rate_target=rate_target)
		# if cart pole, also record teacher dynamics
		if mc.input_signal == 'cartpole':
			mc = test_model_on_cartpole(mc)

	if not teacher:
		# if not, perform training
		logging.info(f"Seed {mc.seed}: running training")
		mc = train_model(mc, epochs=mc.epochs, learn_weights=learn_weights, \
		learn_lat_weights=learn_lat_weights, learn_bw_weights=learn_bw_weights, teacher=teacher, rate_target=rate_target)

		# testing (without target injection)
		logging.info(f"Seed {mc.seed}: running testing")
		mc = test_model(mc, rate_target=rate_target, teacher=False)

	t_diff = time.time() - t_start
	logging.info(f"Seed {mc.seed}: done in {t_diff}s.")

	return mc


def pre_train_model(mc, time=None):

	r0_arr = mc.input

	if time == None:
		time = mc.settling_time / mc.dt
	assert time <= len(r0_arr), "settling_time is longer than input signal length"
	# pre-training to settle voltages -- if we don't do this, weights learn incorrectly due to the incorrect voltages in the beginning
	for i in range(int(time)):
		mc.evolve_system(r0=r0_arr[i], learn_weights=False, learn_lat_weights=False, learn_bw_weights=False)

	return mc

def train_model(mc, epochs=1, learn_weights=True, learn_lat_weights=True, learn_bw_weights=True, teacher=False, testing=False, rate_target=None):

	for epoch in range(epochs):
		logging.info(f"Seed {mc.seed}: working on epoch {epoch}")

		if testing:
			# testing
			# no target injection, records to mse_time_series and mse_test_time_series

			mc = shuffle_dataset(mc, teacher, testing=True, validation=False)

			r0_arr = mc.input_testing
			target = mc.target_testing

			mc = run_one_epoch(mc, r0_arr, target, teacher=teacher, \
												   learn_weights=learn_weights, \
												   learn_lat_weights=learn_lat_weights, \
												   learn_bw_weights=learn_bw_weights, \
												   testing=True, validation=False, \
												   rate_target=rate_target)
		else:
			# training

			mc = shuffle_dataset(mc, teacher, testing=False, validation=False)

			r0_arr = mc.input
			target = mc.target

			mc = run_one_epoch(mc, r0_arr, target, teacher=teacher, \
												   learn_weights=learn_weights, \
												   learn_lat_weights=learn_lat_weights, \
												   learn_bw_weights=learn_bw_weights, \
												   testing=False, validation=False, \
												   rate_target=rate_target)

			# for teacher, run validation to obtain validation targets.
			# for student, run validation every 10 epochs
			if teacher or (not teacher and epochs > 1 \
				and 10 * epoch % epochs == 0) or (not teacher and epochs == 1):

				logging.info(f"Seed {mc.seed}: validating epoch {epoch}")
				# validation
				# same as testing, but record to mse_time_series and mse_val_time_series
				r0_arr = mc.input_validation
				target = mc.target_validation

				mc = shuffle_dataset(mc, teacher, testing=False, validation=True)

				mc = run_one_epoch(mc, r0_arr, target, teacher=teacher, \
													   learn_weights=learn_weights, \
													   learn_lat_weights=learn_lat_weights, \
													   learn_bw_weights=learn_bw_weights, \
													   testing=False, validation=True, \
													   rate_target=rate_target)
			if not teacher and mc.input_signal == 'cartpole':
				# validate on cartpole task
				mc = test_model_on_cartpole(mc)
				
	return mc

def test_model(mc, rate_target, teacher):

	if mc.input_signal in ["step", "genMNIST"]:
		mc = train_model(mc, epochs=1, learn_weights=False, \
		learn_lat_weights=False, learn_bw_weights=False, teacher=teacher, testing=True, rate_target=rate_target)

	elif mc.input_signal == 'cartpole':
		mc = test_model_on_cartpole(mc)

	return mc


def run_one_epoch(mc, r0_arr, target, learn_weights, learn_lat_weights, learn_bw_weights, teacher, testing, validation, rate_target):

	for i in range(len(r0_arr)):
		# if mc is teacher, evolve and record
		if teacher:
			# NOTE: we first record and then evolve.
			# this is arbitrary, but changing the order
			# would need to be compensated by dt offsets.
			if rate_target:
				if testing:
					mc.target_testing.append(copy.deepcopy(mc.rP_breve[-1]))
				elif validation:
					mc.target_validation.append(copy.deepcopy(mc.rP_breve[-1]))
				else:
					mc.target.append(copy.deepcopy(mc.rP_breve[-1]))
			else:
				if testing:
					mc.target_testing.append(copy.deepcopy(mc.uP_breve[-1]))
				elif validation:
					mc.target_validation.append(copy.deepcopy(mc.uP_breve[-1]))
				else:
					mc.target.append(copy.deepcopy(mc.uP_breve[-1]))
			# evolve system for one step
			mc.evolve_system(r0=r0_arr[i], learn_weights=learn_weights, learn_lat_weights=learn_lat_weights, learn_bw_weights=learn_bw_weights)

		# if target has been defined, use that
		elif hasattr(mc, 'target'):
			mc.evolve_system(r0=r0_arr[i], u_tgt=[target[i]], learn_weights=learn_weights, learn_lat_weights=learn_lat_weights, \
				learn_bw_weights=learn_bw_weights, testing=testing, validation=validation)
		else:
			mc.evolve_system(r0=r0_arr[i], learn_weights=learn_weights, learn_lat_weights=learn_lat_weights, learn_bw_weights=learn_bw_weights)

	return mc


def shuffle_dataset(mc, teacher, testing, validation):

	if mc.input_signal in ['step', 'cartpole']:

		if not teacher and hasattr(mc, 'target'):
			logging.debug(f"Shuffling input and output pairs")

			# reshape to equal chunks of Tpres
			# and take last dt of each sample in order to avoid settling phase
			if testing:
				r0_arr = mc.input_testing
				target = mc.target_testing
			elif validation:
				r0_arr = mc.input_validation
				target = mc.target_validation
			else:
				r0_arr = mc.input
				target = mc.target
			
			r0_arr = r0_arr.reshape(-1, int(mc.Tpres / mc.dt), mc.layers[0])[:,-1]
			target = np.array(target).reshape(-1, int(mc.Tpres / mc.dt), mc.layers[-1])[:,-1]

			# shuffle input and target in the same way
			r0_arr, target = unison_shuffled_copies(r0_arr, target, mc.rng)

			# repeat to get original Tpres length
			r0_arr = np.repeat(r0_arr, int(mc.Tpres / mc.dt), axis=0)
			target = np.repeat(target, int(mc.Tpres / mc.dt), axis=0)

			# move target to accomodate for settling using LE
			if mc.mc_model in ['errormc', 'sacramento2018', 'ann', 'dPC']:
				# signal needs time to be passed from error to rep units
				dt_offset = len(mc.layers) - 1
			if dt_offset > 0:
				target = np.append(target[:dt_offset], target[:-dt_offset], axis=0)

			# assign shuffled target to mc
			if testing:
				mc.input_testing = r0_arr.copy()
				mc.target_testing = target.copy()
			elif validation:
				mc.input_validation = r0_arr.copy()
				mc.target_validation = target.copy()
			else:
				mc.input = r0_arr.copy()
				mc.target = target.copy()

		elif not teacher:
			logging.debug(f"Shuffling input")
			# reshape to equal chunks of Tpres
			r0_arr = r0_arr.reshape(-1, int(mc.Tpres / mc.dt), mc.layers[0])
			# shuffle along time axis
			mc.rng.shuffle(r0_arr)
			# reshape to original shape
			r0_arr = r0_arr.reshape(-1, mc.layers[0])
			mc.input = r0_arr.copy()

	return mc
