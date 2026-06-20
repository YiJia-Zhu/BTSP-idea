import numpy as np
from src.microcircuit import *

def init_r0(MC_list, form="step", seed=123):

	rng = np.random.RandomState(seed)

	if form == "step":
		r0 = rng.uniform(0, 1, size=(3 * MC_list[0].dataset_size, MC_list[0].layers[0]))
		# split into train and test
		r0_train, r0_val, r0_test = np.split(r0, 3, axis=0)
		# extend by Tpres
		r0_train = np.repeat(r0_train, int(MC_list[0].Tpres / MC_list[0].dt), axis=0)
		r0_val = np.repeat(r0_val, int(MC_list[0].Tpres / MC_list[0].dt), axis=0)
		r0_test = np.repeat(r0_test, int(MC_list[0].Tpres / MC_list[0].dt), axis=0)

		targets = None

	elif form == "cartpole":
		r0 = rng.uniform(-1.5, 1.5, size=(3 * MC_list[0].dataset_size, MC_list[0].layers[0]))
		# split into train and test
		r0_train, r0_val, r0_test = np.split(r0, 3, axis=0)
		# extend by Tpres
		r0_train = np.repeat(r0_train, int(MC_list[0].Tpres / MC_list[0].dt), axis=0)
		r0_val = np.repeat(r0_val, int(MC_list[0].Tpres / MC_list[0].dt), axis=0)
		r0_test = np.repeat(r0_test, int(MC_list[0].Tpres / MC_list[0].dt), axis=0)

		targets = None

	elif form == "genMNIST":
		import torch
		import torchvision
		import torchvision.transforms as transforms

		# load MNIST dataset and
		# normalize to [-1, 1]
		# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
		# normalize to [0, 1]
		transform = transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               # torchvision.transforms.Normalize(
                               #   (0.1307,), (0.3081,))
                             ])
		mnist_full = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

		# assign one set of inputs (digits 0-9)
		# and one (!) set of targets (images)

		samples_train = get_each_MNIST_digit(mnist_full)

		# sort from 0 to 9
		samples_sorted = sorted(samples_train,key=lambda x: x[1])

		# # show images
		# import matplotlib.pyplot as plt

		# def imshow(img):
		# 	# img = img / 2 + 0.5     # unnormalize
		# 	npimg = img.numpy()
		# 	plt.imshow(np.transpose(npimg, (1, 2, 0)))
		# 	plt.show()

		# for (img, label) in samples:
		# 	print(f"label {label}")
		# 	print(img.min(), img.max(), img.mean(), img.std())
		# 	imshow(torchvision.utils.make_grid(img))

		r0_train = [label_to_onehot(label, classes=10) for (img, label) in samples_train]
		target_train = [img.ravel() for (img, label) in samples_train]

		# extend by Tpres
		r0_train = np.repeat(r0_train, int(MC_list[0].Tpres / MC_list[0].dt), axis=0)
		target_train = np.repeat(target_train, int(MC_list[0].Tpres / MC_list[0].dt), axis=0)

		# assign same dataset as train/val/test
		r0_val = r0_train
		target_val = target_train

		r0_test = r0_train
		target_test = target_train

		targets = True


	for mc in MC_list:
		mc.input = r0_train.copy()
		mc.input_validation = r0_val.copy()
		mc.input_testing = r0_test.copy()

		if targets is not None:
			mc.target = target_train.copy()
			mc.target_validation = target_val.copy()
			mc.target_testing = target_test.copy()

	return MC_list

def get_each_MNIST_digit(mnist_full):
	# find one sample per digit
	samples = []
	labels_found = set()
	for img, label in mnist_full:
		if label not in labels_found:
			samples.append((img, label))
			labels_found.add(label)
		if len(labels_found) == 10:
			break

	return samples
