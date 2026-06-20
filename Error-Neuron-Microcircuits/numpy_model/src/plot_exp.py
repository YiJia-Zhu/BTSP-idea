import numpy as np
from src.microcircuit import *
#import time
import logging
import warnings
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter

try:
	plt.style.use('matplotlib_style.mplstyle')
except:
	plt.rcParams['text.usetex'] = False
	plt.rc('font', size=10,family='serif')

logging.basicConfig(format='Train model -- %(levelname)s: %(message)s',
					level=logging.INFO)


def plot(MC_list, MC_teacher=None, path=None):

	warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)
	logging.info("Warning: ignoring MatplotlibDeprecationWarning")

	# define a variable for path of output files
	if path is not None:
		PATH = path + '/'
	else:
		PATH = ''

	# define the number of recorded time steps which belong to the pre-training
	# and therefore should be skipped in plotting
	TPRE = int(MC_list[0].settling_time / MC_list[0].dt / MC_list[0].rec_per_steps)

	# colors for plotting
	color = cm.rainbow(np.linspace(0, 1, len(MC_list)))
	# color of teacher
	CLR_TEACH = 'k'

	if (MC_list[0].rec_MSE and MC_teacher is not None) or MC_list[0].input_signal == 'genMNIST':

		fig = plt.figure()
		# for mc, c in zip(MC_list, color):
		data = np.array([mc.MSE_time_series for mc in MC_list])
		mean = np.mean(data, axis=0)
		std  = np.std(data, axis=0)
		
		v = int(10*MC_list[0].Tpres/MC_list[0].dt/MC_list[0].rec_per_steps)
		avg_mean = moving_average(mean, v if v>0 else 1)
		avg_std = moving_average(std, v if v>0 else 1)

		plt.plot(avg_mean)
		plt.fill_between(np.arange(len(avg_mean)), avg_mean-avg_std, avg_mean+avg_std, alpha=0.2)
			# plt.plot(mc.MSE_time_series, c=c)

		plt.title("MSE (train + val + test, window over $10\\;T_\\mathrm{pres}$)")
		plt.yscale('log')
		# plt.legend()
		plt.tight_layout()
		plt.savefig(PATH + 'MSE.png', dpi=200)

	if (MC_list[0].rec_MSE and MC_teacher is not None and hasattr(MC_list[0], 'MSE_val_time_series') and MC_list[0].input_signal != 'cartpole') \
		or MC_list[0].input_signal == 'genMNIST':

		fig = plt.figure()
		# for mc, c in zip(MC_list, color):
		data = np.array([mc.MSE_val_time_series for mc in MC_list])
		mean = np.mean(data, axis=0)
		std  = np.std(data, axis=0)
		
		# v = int(10*MC_list[0].Tpres/MC_list[0].dt/MC_list[0].rec_per_steps)
		v = 0
		avg_mean = moving_average(mean, v if v>0 else 1)
		avg_std = moving_average(std, v if v>0 else 1)

		plt.plot(avg_mean)
		plt.fill_between(np.arange(len(avg_mean)), avg_mean-avg_std, avg_mean+avg_std, alpha=0.2)
			# plt.plot(mc.MSE_time_series, c=c)

		# delimit different validation phases
		# range(0, int(10 * MC_list[0].dataset_size * MC_list[0].dt), int(MC_list[0].dataset_size * MC_list[0].dt))
		if MC_list[0].epochs >= 10:
			[plt.axvline(x=x, color='black', ls='dashed', alpha=0.5) for x \
			 in range(0, int(10 * MC_list[0].dataset_size * MC_list[0].Tpres / MC_list[0].dt),\
			             int(MC_list[0].dataset_size * MC_list[0].Tpres / MC_list[0].dt))]
		

		# plt.title("MSE (window over $10\\;T_\\mathrm{pres}$)")
		plt.title(f"validation MSE (every {MC_list[0].epochs // 10} epochs)")
		plt.yscale('log')
		# plt.legend()
		plt.tight_layout()
		plt.savefig(PATH + 'MSE_val.png', dpi=200)

	if (MC_list[0].rec_MSE and MC_teacher is not None and hasattr(MC_list[0], 'MSE_test_time_series') and MC_list[0].input_signal != 'cartpole') \
		or MC_list[0].input_signal == 'genMNIST':

		fig = plt.figure()
		# for mc, c in zip(MC_list, color):
		data = np.array([mc.MSE_test_time_series for mc in MC_list])
		mean = np.mean(data, axis=0)
		std  = np.std(data, axis=0)
		
		# v = int(10*MC_list[0].Tpres/MC_list[0].dt/MC_list[0].rec_per_steps)
		v = 0
		avg_mean = moving_average(mean, v if v>0 else 1)
		avg_std = moving_average(std, v if v>0 else 1)

		plt.plot(avg_mean)
		plt.fill_between(np.arange(len(avg_mean)), avg_mean-avg_std, avg_mean+avg_std, alpha=0.2)
			# plt.plot(mc.MSE_time_series, c=c)

		# plt.title("MSE (window over $10\\;T_\\mathrm{pres}$)")
		plt.title("test MSE")
		plt.yscale('log')
		# plt.legend()
		plt.tight_layout()
		plt.savefig(PATH + 'MSE_test.png', dpi=200)

	if MC_list[0].rec_WPP:

		for i in range(len(MC_list[0].WPP)):
			fig = plt.figure()
			for mc, c in zip(MC_list, color):
				for j in range(len(mc.WPP_time_series[0][i])):
					plt.plot(np.array([vec[i][j] for vec in mc.WPP_time_series[TPRE:]]), c=c)
					if MC_teacher is not None and i <= len(MC_teacher[0].WPP)-1 and j <= len(MC_teacher[0].WPP[i])-1:
						plt.plot(np.array([MC_teacher[0].WPP[i][j] for vec in MC_list[0].WPP_time_series[TPRE:]]), c=CLR_TEACH, ls='--')
			plt.title("$W^\\mathrm{PP}$ layer " + str(i+1))
			# plt.grid()
			# plt.ylim(0,1)
			plt.xlabel(str(MC_list[0].rec_per_steps) + ' dt')
			file_name = 'WPP_layer'+str(i+1)+'.png'
			plt.savefig(PATH + file_name, dpi=200)

	if MC_list[0].rec_WIP:

		for i in range(len(MC_list[0].WIP)):
			fig = plt.figure()
			for mc, c in zip(MC_list, color):
				for j in range(len(mc.WIP_time_series[0][i])):
					plt.plot(np.array([vec[i][j] for vec in mc.WIP_time_series[TPRE:]]), c=c)
					if MC_teacher is not None and i <= len(MC_teacher[0].WPP)-1 and j <= len(MC_teacher[0].WPP[i])-1:
						plt.plot(np.array([MC_teacher[0].WIP[i][j] for vec in MC_list[0].WIP_time_series[TPRE:]]), c=CLR_TEACH, ls='--')
			plt.title("$W^\\mathrm{IP}$ layer " + str(i+1))
			# plt.grid()
			# plt.ylim(0,1)
			plt.xlabel(str(MC_list[0].rec_per_steps) + ' dt')
			file_name = 'WIP_layer'+str(i+1)+'.png'
			plt.savefig(PATH + file_name, dpi=200)

	if MC_list[0].rec_BPP:

		for i in range(len(MC_list[0].BPP)):
			fig = plt.figure()
			for mc, c in zip(MC_list, color):
				for j in range(len(mc.BPP_time_series[0][i])):
					plt.plot(np.array([vec[i][j] for vec in mc.BPP_time_series[TPRE:]]), c=c)
					# if MC_teacher is not None:
					# 	plt.plot(np.array([MC_teacher[0].BPP[i][j] for vec in MC_list[0].BPP_time_series[TPRE:]]), c=CLR_TEACH, ls='--')
			plt.title("$B^\\mathrm{PP}$ layer " + str(i+1))
			# plt.grid()
			# plt.ylim(0,1)
			plt.xlabel(str(MC_list[0].rec_per_steps) + ' dt')
			file_name = 'BPP_layer'+str(i+1)+'.png'
			plt.savefig(PATH + file_name, dpi=200)

	if MC_list[0].rec_BII:

		for i in range(len(MC_list[0].BII)):
			fig = plt.figure()
			for mc, c in zip(MC_list, color):
				for j in range(len(mc.BII_time_series[0][i])):
					plt.plot(np.array([vec[i][j] for vec in mc.BII_time_series[TPRE:]]), c=c)
					# if MC_teacher is not None:
					# 	plt.plot(np.array([MC_teacher[0].BII[i][j] for vec in MC_list[0].BII_time_series[TPRE:]]), c=CLR_TEACH, ls='--')
			plt.title("$B^\\mathrm{II}$ layer " + str(i+1))
			# plt.grid()
			# plt.ylim(0,1)
			plt.xlabel(str(MC_list[0].rec_per_steps) + ' dt')
			file_name = 'BII_layer'+str(i+1)+'.png'
			plt.savefig(PATH + file_name, dpi=200)

	if MC_list[0].rec_BPI:

		for i in range(len(MC_list[0].BPI)):
			fig = plt.figure()
			for mc, c in zip(MC_list, color):
				for j in range(len(mc.BPI_time_series[0][i])):
					plt.plot(np.array([vec[i][j] for vec in mc.BPI_time_series[TPRE:]]), c=c)
					# if MC_teacher is not None:
					# 	plt.plot(np.array([MC_teacher[0].BPI[i][j] for vec in MC_list[0].BPI_time_series[TPRE:]]), c=CLR_TEACH, ls='--')
			plt.title("$B^\\mathrm{PI}$ layer " + str(i+1))
			# plt.grid()
			# plt.ylim(0,1)
			plt.xlabel(str(MC_list[0].rec_per_steps) + ' dt')
			file_name = 'BPI_layer'+str(i+1)+'.png'
			plt.savefig(PATH + file_name, dpi=200)

	if MC_list[0].rec_BPI and MC_list[0].rec_BPP and MC_list[0].mc_model == 'sacramento2018':

		for i in range(len(MC_list[0].BPI)):
				fig = plt.figure()
				for mc, c in zip(MC_list, color):
					for j in range(len(mc.BPI_time_series[0][i])):
						vec1 = np.array([vec[i][j] for vec in mc.BPI_time_series[TPRE:]])
						vec2 = np.array([vec[i][j] for vec in mc.BPP_time_series[TPRE:]])
						plt.plot(vec1+vec2, c=c)
						# if MC_teacher is not None:
						# 	plt.plot(np.array([MC_teacher[0].BPI[i][j] for vec in MC_list[0].BPI_time_series[TPRE:]]), c=CLR_TEACH, ls='--')
				plt.title("$B^\\mathrm{PI}+B^\\mathrm{PP}$ layer " + str(i+1))
				# plt.grid()
				# plt.ylim(0,1)
				plt.xlabel(str(MC_list[0].rec_per_steps) + ' dt')
				file_name = 'BPP+BPI_layer'+str(i+1)+'.png'
				plt.savefig(PATH + file_name, dpi=200)



	if MC_list[0].rec_uP:

		for i in range(len(MC_list[0].layers)-1):
			fig = plt.figure()
			for mc, c in zip(MC_list, color):
				for j in range(len(mc.uP_time_series[0][i])):
					plt.plot(np.array([vec[i][j] for vec in mc.uP_time_series[TPRE:]]), c=c)
					# if MC_teacher is not None:
					# 	data = np.array([vec[i][j] for vec in MC_teacher[0].uP_time_series[TPRE:]])
					# 	data = np.tile(data, MC_list[0].epochs)
					# 	plt.plot(data, c=CLR_TEACH, ls='--')
			plt.title("$u^\\mathrm{P}$ layer " + str(i+1))
			# plt.grid()
			# plt.ylim(0,1)
			plt.xlabel(str(MC_list[0].rec_per_steps) + ' dt')
			file_name = 'uP_layer'+str(i+1)+'.png'
			plt.savefig(PATH + file_name, dpi=200)

	if MC_list[0].rec_rP_breve:

		for i in range(len(MC_list[0].layers)-1):
			fig = plt.figure()
			for mc, c in zip(MC_list, color):
				for j in range(len(mc.rP_breve_time_series[0][i])):
					plt.plot(np.array([vec[i][j] for vec in mc.rP_breve_time_series[TPRE:]]), c=c)
					if hasattr(mc, "target_time_series"):
						if i == len(MC_list[0].layers)-2 and MC_list[0].rate_target:
							plt.plot(np.array([vec[-1][j] for vec in mc.target_time_series]), c=CLR_TEACH, ls='--')
					# if MC_teacher is not None:
					# 	data = np.array([vec[i][j] for vec in MC_teacher[0].rP_breve_time_series[TPRE:]])
					# 	data = np.tile(data, MC_list[0].epochs)
					# 	plt.plot(data, c=CLR_TEACH, ls='--')
			plt.title("$\\breve{r}^\\mathrm{P}$ layer " + str(i+1))
			# plt.grid()
			# plt.ylim(0,1)
			plt.xlabel(str(MC_list[0].rec_per_steps) + ' dt')
			file_name = 'rP_breve_layer'+str(i+1)+'.png'
			plt.savefig(PATH + file_name, dpi=200)


	if MC_list[0].rec_uP_breve:

		for i in range(len(MC_list[0].layers)-1):
			fig = plt.figure()
			for mc, c in zip(MC_list, color):
				for j in range(len(mc.uP_breve_time_series[0][i])):
					plt.plot(np.array([vec[i][j] for vec in mc.uP_breve_time_series[TPRE:]]), c=c)
					if hasattr(mc, "target_time_series"):
						if i == len(MC_list[0].layers)-2 and not MC_list[0].rate_target:
							plt.plot(np.array([vec[-1][j] for vec in mc.target_time_series]), c=CLR_TEACH, ls='--')
					# if MC_teacher is not None:
					# 	data = np.array([vec[i][j] for vec in MC_teacher[0].uP_breve_time_series[TPRE:]])
					# 	data = np.tile(data, MC_list[0].epochs)
					# 	plt.plot(data, c=CLR_TEACH, ls='--')
			plt.title("$\\breve{u}^\\mathrm{P}$ layer " + str(i+1))
			# plt.grid()
			# plt.ylim(0,1)
			plt.xlabel(str(MC_list[0].rec_per_steps) + ' dt')
			file_name = 'uP_breve_layer'+str(i+1)+'.png'
			plt.savefig(PATH + file_name, dpi=200)


	# the following three variables are only relevant in the last layer
	if MC_list[0].rec_rP_breve_HI:

		for i in range(-1,0):
			fig = plt.figure()
			for mc, c in zip(MC_list, color):
				for j in range(len(mc.rP_breve_HI_time_series[0][i])):
					plt.plot(np.array([vec[i][j] for vec in mc.rP_breve_HI_time_series[TPRE:]]), c=c)
			plt.title("$\\breve{r}^\\mathrm{P}_\\mathrm{HI}$ layer " + str(i+1))
			# plt.grid()
			# plt.ylim(0,1)
			plt.xlabel(str(MC_list[0].rec_per_steps) + ' dt')
			file_name = 'rP_breve_HI_layer'+str(i+1)+'.png'
			plt.savefig(PATH + file_name, dpi=200)

	if MC_list[0].rec_uI:

		for i in range(-1,0):
			fig = plt.figure()
			for mc, c in zip(MC_list, color):
				for j in range(len(mc.uI_time_series[0][i])):
					plt.plot(np.array([vec[i][j] for vec in mc.uI_time_series[TPRE:]]), c=c)
					# if MC_teacher is not None:
					# 	data = np.array([vec[i][j] for vec in MC_teacher[0].uI_time_series[TPRE:]])
					# 	data = np.tile(data, MC_list[0].epochs)
					# 	plt.plot(data, c=CLR_TEACH, ls='--')
			plt.title("$u^\\mathrm{I}$ layer " + str(i+1))
			# plt.grid()
			# plt.ylim(0,1)
			plt.xlabel(str(MC_list[0].rec_per_steps) + ' dt')
			file_name = 'uI_layer'+str(i+1)+'.png'
			plt.savefig(PATH + file_name, dpi=200)

	if MC_list[0].rec_rI_breve:

		for i in range(-1,0):
			fig = plt.figure()
			for mc, c in zip(MC_list, color):
				for j in range(len(mc.rI_breve_time_series[0][i])):
					plt.plot(np.array([vec[i][j] for vec in mc.rI_breve_time_series[TPRE:]]), c=c)
					# if MC_teacher is not None:
					# 	data = np.array([vec[i][j] for vec in MC_teacher[0].rI_breve_time_series[TPRE:]])
					# 	data = np.tile(data, MC_list[0].epochs)
					# 	plt.plot(data, c=CLR_TEACH, ls='--')
			plt.title("$\\breve{r}^\\mathrm{I}$ layer " + str(i+1))
			# plt.grid()
			# plt.ylim(0,1)
			plt.xlabel(str(MC_list[0].rec_per_steps) + ' dt')
			file_name = 'rI_breve_layer'+str(i+1)+'.png'
			plt.savefig(PATH + file_name, dpi=200)

	if MC_list[0].rec_vbas:

		for i in range(len(MC_list[0].layers)-1):
			fig = plt.figure()
			for mc, c in zip(MC_list, color):
				for j in range(len(mc.vbas_time_series[0][i])):
					plt.plot(np.array([vec[i][j] for vec in mc.vbas_time_series[TPRE:]]), c=c)
					# if MC_teacher is not None:
					# 	data = np.array([vec[i][j] for vec in MC_teacher[0].vbas_time_series[TPRE:]])
					# 	data = np.tile(data, MC_list[0].epochs)
					# 	plt.plot(data, c=CLR_TEACH, ls='--')
			plt.title("$v^\\mathrm{bas}$, layer " + str(i+1))
			# plt.grid()
			# plt.ylim(0,1)
			plt.xlabel(str(MC_list[0].rec_per_steps) + ' dt')
			file_name = 'vbas_layer'+str(i+1)+'.png'
			plt.savefig(PATH + file_name, dpi=200)

	if MC_list[0].rec_vapi:

		for i in range(len(MC_list[0].layers)-2):
			fig = plt.figure()
			for mc, c in zip(MC_list, color):
				for j in range(len(mc.vapi_time_series[0][i])):
					plt.plot(np.array([vec[i][j] for vec in mc.vapi_time_series[TPRE:]]), c=c)
					# if MC_teacher is not None:
					# 	data = np.array([vec[i][j] for vec in MC_teacher[0].vapi_time_series[TPRE:]])
					# 	data = np.tile(data, MC_list[0].epochs)
					# 	plt.plot(data, c=CLR_TEACH, ls='--')
			plt.title("$v^\\mathrm{api}$ before noise injection, layer " + str(i+1))
			# plt.grid()
			# plt.ylim(0,1)
			plt.xlabel(str(MC_list[0].rec_per_steps) + ' dt')
			file_name = 'vapi_layer'+str(i+1)+'.png'
			plt.savefig(PATH + file_name, dpi=200)

	if MC_list[0].rec_lat_mismatch:

		for i in range(len(MC_list[0].layers)-2):
			fig = plt.figure()
			for mc, c in zip(MC_list, color):
				for j in range(len(mc.lat_mismatch_time_series[0][i])):
					plt.plot(np.array([vec[i][j] for vec in mc.lat_mismatch_time_series[TPRE:]]), c=c)
					# if MC_teacher is not None:
					# 	data = np.array([vec[i][j] for vec in MC_teacher[0].lat_mismatch_time_series[TPRE:]])
					# 	data = np.tile(data, MC_list[0].epochs)
					# 	plt.plot(data, c=CLR_TEACH, ls='--')
			plt.title("$\\breve{u}^\\mathrm{I} - \\widehat{v}^\\mathrm{bas}$, layer " + str(i+1))
			# plt.grid()
			# plt.ylim(0,1)
			plt.xlabel(str(MC_list[0].rec_per_steps) + ' dt')
			file_name = 'lat_mismatch_layer'+str(i+1)+'.png'
			plt.savefig(PATH + file_name, dpi=200)

	if MC_list[0].rec_vapi_noise:

		for i in range(len(MC_list[0].layers)-2):
			fig = plt.figure()
			for mc, c in zip(MC_list, color):
				for j in range(len(mc.vapi_noise_time_series[0][i])):
					plt.plot(np.array([vec[i][j] for vec in mc.vapi_noise_time_series[TPRE:]]), c=c)
			plt.title("$v^\\mathrm{api}$ after noise injection, layer " + str(i+1))
			# plt.grid()
			# plt.ylim(0,1)
			plt.xlabel(str(MC_list[0].rec_per_steps) + ' dt')
			file_name = 'vapi_noise_layer'+str(i+1)+'.png'
			plt.savefig(PATH + file_name, dpi=200)

	if MC_list[0].rec_noise:

		for i in range(len(MC_list[0].layers)-2):
			fig = plt.figure()
			for mc, c in zip(MC_list, color):
				for j in range(len(mc.noise_time_series[0][i])):
					plt.plot(np.array([vec[i][j] for vec in mc.noise_time_series[TPRE:]]), c=c)
			plt.title("injected noise layer " + str(i+1))
			# plt.grid()
			# plt.ylim(0,1)
			plt.xlabel(str(MC_list[0].rec_per_steps) + ' dt')
			file_name = 'noise_layer'+str(i+1)+'.png'
			plt.savefig(PATH + file_name, dpi=200)

	if MC_list[0].rec_epsilon:

		for i in range(len(MC_list[0].layers)-2):
			fig = plt.figure()
			for mc, c in zip(MC_list, color):
				plt.plot([vec[i] for vec in mc.epsilon_time_series[TPRE:]], c=c)
			plt.title("$\\epsilon$ layer " + str(i+1))
			plt.grid()
			plt.ylim(0,1)
			plt.xlabel(str(MC_list[0].rec_per_steps) + ' dt')
			file_name = 'epsilon_layer'+str(i+1)+'.png'
			plt.savefig(PATH + file_name, dpi=200) 


	if MC_list[0].rec_epsilon_LO:

		for i in range(len(MC_list[0].layers)-2):
			fig = plt.figure()
			for mc, c in zip(MC_list, color):
				plt.plot([vec[i] for vec in mc.epsilon_LO_time_series[TPRE:]], c=c)
			plt.title("$\\epsilon_\\mathrm{LO}$ layer " + str(i+1))
			plt.grid()
			plt.ylim(0,1)
			plt.xlabel(str(MC_list[0].rec_per_steps) + ' dt')
			file_name = 'epsilon_LO_layer'+str(i+1)+'.png'
			plt.savefig(PATH + file_name, dpi=200) 

		for i in range(len(MC_list[0].layers)-2):
			fig = plt.figure()
			for mc, c in zip(MC_list, color):
				plt.plot([1-2*np.array(vec[i]) for vec in mc.epsilon_LO_time_series[TPRE:]], label="$\\cos(B^\\mathrm{PP}r^P, \\breve{u}_i^P)$", c=c)

			lab = str(MC_list[0].noise_deg) + "$^\\circ$"
			plt.plot([np.cos(MC_list[0].noise_deg * np.pi/180) for vec in MC_list[0].epsilon_LO_time_series[TPRE:]], label=lab, ls='--')

			plt.title("$1 - 2 \\;\\epsilon_\\mathrm{LO} \\sim$ cos, layer " + str(i+1))
			# plt.yscale("log")
			plt.legend()
			plt.grid()
			plt.ylim(-1.1,1.1)
			plt.xlabel(str(MC_list[0].rec_per_steps) + ' dt')
			file_name = 'cos_layer'+str(i+1)+'.png'
			plt.savefig(PATH + file_name, dpi=200)

	if hasattr(MC_list[0], "angle_dWPP_time_series"):

		for i in range(len(MC_list[0].angle_dWPP_time_series[0])):
			fig = plt.figure()
			for mc, c in zip(MC_list, color):
				plt.plot([vec[i] for vec in mc.angle_dWPP_time_series], c=c)
			plt.title("$\\angle (\\Delta W^\\mathrm{PP}, \\mathrm{BP})$ layer " + str(i+1))
			plt.ylabel("deg")
			plt.grid()
			plt.xlabel(str(MC_list[0].rec_per_steps) + ' dt')
			file_name = 'angle_BP_dWPP'+str(i+1)+'.png'
			plt.tight_layout()
			plt.savefig(PATH + file_name, dpi=200) 

	if hasattr(MC_list[0], "angle_WPPT_BPP_time_series"):

		for i in range(len(MC_list[0].angle_WPPT_BPP_time_series[0])):
			fig = plt.figure()
			for mc, c in zip(MC_list, color):
				plt.plot([vec[i] for vec in mc.angle_WPPT_BPP_time_series[TPRE:]], c=c)
			plt.title("$\\angle (B^\\mathrm{PP}, (W^\\mathrm{PP})^T )$ layer " + str(i+1))
			plt.ylabel("deg")
			plt.grid()
			plt.xlabel(str(MC_list[0].rec_per_steps) + ' dt')
			file_name = 'angle_WPPT_BPP'+str(i+1)+'.png'
			plt.tight_layout()
			plt.savefig(PATH + file_name, dpi=200) 

			# create a mean + std plot
			fig = plt.figure()
			data = []
			for mc, c in zip(MC_list, color):
				data.append([vec[i] for vec in mc.angle_WPPT_BPP_time_series[TPRE:]])
			mean = np.mean(data, axis=0)
			std = np.std(data, axis=0)
			x = np.arange(len(mean))

			plt.plot(x, mean, c='k')
			plt.title("$\\angle (B^\\mathrm{PP},(W^\\mathrm{PP})^T)$ layer " + str(i+1))
			plt.ylabel("deg")
			plt.fill_between(x, mean+std, mean-std, color='gray', alpha=.5)
			plt.grid()
			plt.xlabel(str(MC_list[0].rec_per_steps) + ' dt')
			file_name = 'angle_WPPT_BPP'+str(i+1)+'_mean.png'
			plt.tight_layout()
			plt.savefig(PATH + file_name, dpi=200)

	if hasattr(MC_list[0], "angle_WPPT_BII_time_series"):

		for i in range(len(MC_list[0].angle_WPPT_BII_time_series[0])):
			fig = plt.figure()
			for mc, c in zip(MC_list, color):
				plt.plot([vec[i] for vec in mc.angle_WPPT_BII_time_series[TPRE:]], c=c)
			plt.title("$\\angle (B^\\mathrm{II}, (W^\\mathrm{PP})^T )$ (with padding zeros) layer " + str(i+1))
			plt.ylabel("deg")
			plt.grid()
			plt.xlabel(str(MC_list[0].rec_per_steps) + ' dt')
			file_name = 'angle_WPPT_BII'+str(i+1)+'.png'
			plt.tight_layout()
			plt.savefig(PATH + file_name, dpi=200) 

			# create a mean + std plot
			fig = plt.figure()
			data = []
			for mc, c in zip(MC_list, color):
				data.append([vec[i] for vec in mc.angle_WPPT_BII_time_series[TPRE:]])
			mean = np.mean(data, axis=0)
			std = np.std(data, axis=0)
			x = np.arange(len(mean))

			plt.plot(x, mean, c='k')
			plt.title("$\\angle (B^\\mathrm{II}, (W^\\mathrm{PP})^T )$ (with padding zeros) ayer " + str(i+1))
			plt.ylabel("deg")
			plt.fill_between(x, mean+std, mean-std, color='gray', alpha=.5)
			plt.grid()
			plt.xlabel(str(MC_list[0].rec_per_steps) + ' dt')
			file_name = 'angle_WPPT_BII'+str(i+1)+'_mean.png'
			plt.tight_layout()
			plt.savefig(PATH + file_name, dpi=200) 

	if hasattr(MC_list[0], "angle_jacobians_BP_time_series"):

		for i in range(len(MC_list[0].angle_jacobians_BP_time_series[0])):
			fig = plt.figure()
			for mc, c in zip(MC_list, color):
				plt.plot([vec[i] for vec in mc.angle_jacobians_BP_time_series[TPRE:]], c=c)
			plt.title("$\\angle (J_g, J_f^T)$ layer " + str(i+1))
			plt.ylabel("deg")
			plt.grid()
			plt.xlabel(str(MC_list[0].rec_per_steps) + ' dt')
			file_name = 'angle_jacobians_BP'+str(i+1)+'.png'
			plt.tight_layout()
			plt.savefig(PATH + file_name, dpi=200) 

			# create a mean + std plot
			fig = plt.figure()
			data = []
			for mc, c in zip(MC_list, color):
				data.append([vec[i] for vec in mc.angle_jacobians_BP_time_series[TPRE:]])
			mean = np.mean(data, axis=0)
			std = np.std(data, axis=0)
			x = np.arange(len(mean))

			plt.plot(x, mean, c='k')
			plt.title("$\\angle (J_g, J_f^T)$ layer " + str(i+1))
			plt.ylabel("deg")
			plt.fill_between(x, mean+std, mean-std, color='gray', alpha=.5)
			plt.grid()
			plt.xlabel(str(MC_list[0].rec_per_steps) + ' dt')
			file_name = 'angle_jacobians_BP'+str(i+1)+'_mean.png'
			plt.tight_layout()
			plt.savefig(PATH + file_name, dpi=200) 

	if hasattr(MC_list[0], "angle_BPP_RHS_time_series"):

		for i in range(len(MC_list[0].angle_BPP_RHS_time_series[0])):
			fig = plt.figure()
			for mc, c in zip(MC_list, color):
				plt.plot([vec[i] for vec in mc.angle_BPP_RHS_time_series[TPRE:]], c=c)
			plt.title("$\\angle (B^\\mathrm{PP}, \\varphi' (W^\\mathrm{PP})^T \\varphi')$ layer " + str(i+1))
			plt.ylabel("deg")
			plt.grid()
			plt.xlabel(str(MC_list[0].rec_per_steps) + ' dt')
			file_name = 'angle_BPP_RHS'+str(i+1)+'.png'
			plt.tight_layout()
			plt.savefig(PATH + file_name, dpi=200) 

			# create a mean + std plot
			fig = plt.figure()
			data = []
			for mc, c in zip(MC_list, color):
				data.append([vec[i] for vec in mc.angle_BPP_RHS_time_series[TPRE:]])
			mean = np.mean(data, axis=0)
			std = np.std(data, axis=0)
			x = np.arange(len(mean))

			plt.plot(x, mean, c='k')
			plt.title("$\\angle (B^\\mathrm{PP}, \\varphi' (W^\\mathrm{PP})^T \\varphi')$ layer " + str(i+1))
			plt.ylabel("deg")
			plt.fill_between(x, mean+std, mean-std, color='gray', alpha=.5)
			plt.grid()
			plt.xlabel(str(MC_list[0].rec_per_steps) + ' dt')
			file_name = 'angle_BPP_RHS'+str(i+1)+'_mean.png'
			plt.tight_layout()
			plt.savefig(PATH + file_name, dpi=200)

	if hasattr(MC_list[0], "angle_BII_RHS_time_series"):

		for i in range(len(MC_list[0].angle_BII_RHS_time_series[0])):
			fig = plt.figure()
			for mc, c in zip(MC_list, color):
				plt.plot([vec[i] for vec in mc.angle_BII_RHS_time_series[TPRE:]], c=c)
			plt.title("$\\angle (B^\\mathrm{II}, [\\varphi' ... W^\\mathrm{PP} ... \\varphi']^T)$ (full PAL) layer " + str(i+1))
			plt.ylabel("deg")
			plt.grid()
			plt.xlabel(str(MC_list[0].rec_per_steps) + ' dt')
			file_name = 'angle_BII_RHS'+str(i+1)+'.png'
			plt.tight_layout()
			plt.savefig(PATH + file_name, dpi=200) 

			# create a mean + std plot
			fig = plt.figure()
			data = []
			for mc, c in zip(MC_list, color):
				data.append([vec[i] for vec in mc.angle_BII_RHS_time_series[TPRE:]])
			mean = np.mean(data, axis=0)
			std = np.std(data, axis=0)
			x = np.arange(len(mean))

			plt.plot(x, mean, c='k')
			plt.title("$\\angle (B^\\mathrm{II}, [\\varphi' ... W^\\mathrm{PP} ... \\varphi']^T)$ (full PAL) layer " + str(i+1))
			plt.ylabel("deg")
			plt.fill_between(x, mean+std, mean-std, color='gray', alpha=.5)
			plt.grid()
			plt.xlabel(str(MC_list[0].rec_per_steps) + ' dt')
			file_name = 'angle_BII_RHS'+str(i+1)+'_mean.png'
			plt.tight_layout()
			plt.savefig(PATH + file_name, dpi=200)

	if hasattr(MC_list[0], "angle_BP_updates_time_series"):

		for i in range(len(MC_list[0].angle_BP_updates_time_series[0])):
			fig = plt.figure()
			for mc, c in zip(MC_list, color):
				plt.plot([vec[i] for vec in mc.angle_BP_updates_time_series], c=c)
			plt.title("$\\angle (\\Delta W^\\mathrm{BP}, \\Delta W^\\mathrm{PP})$ layer " + str(i+1))
			plt.ylabel("deg")
			plt.grid()
			plt.xlabel('trainset * epochs')
			plt.ylim(-5, 100)
			file_name = 'angle_MC_BP_ANN'+str(i+1)+'.png'
			plt.tight_layout()
			plt.savefig(PATH + file_name, dpi=200) 

			# create a mean + std plot
			fig = plt.figure()
			data = []
			for mc, c in zip(MC_list, color):
				data.append([vec[i] for vec in mc.angle_BP_updates_time_series])
			mean = np.mean(data, axis=0)
			std = np.std(data, axis=0)
			x = np.arange(len(mean))

			plt.plot(x, mean, c='k')
			plt.title("$\\angle (\\Delta W^\\mathrm{BP}, \\Delta W^\\mathrm{PP})$ layer " + str(i+1))
			plt.ylabel("deg")
			plt.fill_between(x, mean+std, mean-std, color='gray', alpha=.5)
			plt.grid()
			plt.xlabel('trainset * epochs')
			plt.ylim(-5, 100)
			file_name = 'angle_MC_BP_ANN'+str(i+1)+'_mean.png'
			plt.tight_layout()
			plt.savefig(PATH + file_name, dpi=200)

	if hasattr(MC_list[0], "angle_FA_updates_time_series"):

		for i in range(len(MC_list[0].angle_FA_updates_time_series[0])):
			fig = plt.figure()
			for mc, c in zip(MC_list, color):
				plt.plot([vec[i] for vec in mc.angle_FA_updates_time_series[TPRE:]], c=c)
			plt.title("$\\angle (\\Delta W^\\mathrm{BP}, \\Delta W^\\mathrm{PP})$ layer " + str(i+1))
			plt.ylabel("deg")
			plt.grid()
			plt.xlabel('T_pres * epochs')
			file_name = 'angle_MC_FA_ANN'+str(i+1)+'.png'
			plt.tight_layout()
			plt.savefig(PATH + file_name, dpi=200) 

			# create a mean + std plot
			fig = plt.figure()
			data = []
			for mc, c in zip(MC_list, color):
				data.append([vec[i] for vec in mc.angle_FA_updates_time_series[TPRE:]])
			mean = np.mean(data, axis=0)
			std = np.std(data, axis=0)
			x = np.arange(len(mean))

			plt.plot(x, mean, c='k')
			plt.title("$\\angle (\\Delta W^\\mathrm{BP}, \\Delta W^\\mathrm{PP})$ layer " + str(i+1))
			plt.ylabel("deg")
			plt.fill_between(x, mean+std, mean-std, color='gray', alpha=.5)
			plt.grid()
			plt.xlabel('T_pres * epochs')
			file_name = 'angle_MC_FA_ANN'+str(i+1)+'_mean.png'
			plt.tight_layout()
			plt.savefig(PATH + file_name, dpi=200)

	if MC_list[0].input_signal == 'cartpole':
		state_names = [r"$x$", r"$\dot{x}$", r"$\theta$", r"$\dot{\theta}$"]
		for mc in MC_list:
			plt.figure()
			for i in [0, 2]:
				plt.plot(mc.state_log[:, i], label=state_names[i])
			plt.plot(mc.action_log, label="reset", alpha=0.5, ls="--")
			plt.ylim(-1.5, 1.5)
			plt.title("Cartpole dynamics with error-mc controller")
			file_name = 'cartpole_dynamics.png'
			plt.legend()
			plt.tight_layout()
			plt.savefig(PATH + file_name, dpi=200)
