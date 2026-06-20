# Numpy implementation of error neuron microcircuits

This folder contains all files to reproduce all figures, simulating our error microcircuit (`errormc`), the model by Sacramento et al., 2018 (`sacramento2018`), the one of Mikulasch et al., 2022 (`dPC`), and a layered ANN (`ann`).

## Using a parameter file template

You can create a custom experiment using the [template](https://github.com/kma-code/Error-Neuron-Microcircuits/tree/main/numpy_model/example/params_template.json).
Modify it, and run using `python runner.py  --params example/params_template.json`

- if comparison plots with Backprop (e.g. angle between WPP^T and BPP) are required, add the flag `--compare BP`. This will run additional evaluations after the simulation. This can also be run after simulation by using `--load model.pkl`, see below. 
- all plots are saved in `example`, together with a `model.pkl` file of the microcircuit class objects after training
- models are saved after training and once again after plotting. To load and re-plot a saved model, run `python runner.py --params .../params.json  --load .../model.pkl`.


### General hints

All parameters are explained in the [params.json](https://github.com/kma-code/Error-Neuron-Microcircuits/tree/main/numpy_model/example/params_template.json) files.

Some more tips:
- for learning rate `eta_fw`: errormc has `fw_connection_mode` `layered` and `skip`. `eta_fw` must be a list.
-- if `layered`, entries of `eta_fw` correspond to layer-wise learning rates; example: `"eta_fw": [1, 0.2, 0.05]` for a 3-layer network.
-- if `skip`, you need to specify how the list is used to determine learning rates of skip connections. Options are: 'fill_diag' (do not train skip connections), 'fill_all' (each efferent weight has same learning rate), 'fill_scaled' (same as fill_all, but using `realistic_connectivity` to scale learning rates).
- `model_type` denotes how backward weights are set; implemented models are `BP`, `FA`
- *seeds* in `params.json` is an array of numpy random seeds (not a number of seeds)
- *input_signal* in `params.json` defines the signal fed into teacher and students. Currently implemented options: `step, cartpole, genMNIST`
- setting *rec_per_steps* to anything below 1/dt (standard: 1000) slows down training and generates large .pkl files
- recording too many variables slows down training significantly

Data is recorded in lists such as `uP_breve_time_series`. Every class object (i.e. every microcircuit model) saves its own time series, which can be called with e.g. `mc1.uP_breve_time_series`. Every time series has the index structure `uP_breve_time_series[recorded time step][layer][neuron index]` for voltages and rates; weight time series are of the form `WPP_breve_time_series[recorded time step][layer][weight index]`.

To load a saved .pkl-file in an interactive Python session, go to the folder where `runner.py` is located, run `python`, and

```
import src.save_exp
import dill
with open('model.pkl', 'rb') as f: input = dill.load(f)
```
After loading this, `input[0]` represents the teacher model (if it was initiated) and other elements are student networks.




### Nomenclature:

For historical reasons, the neuron populations have variable names `uP` and `uI`. For each model, this corresponds to:
- errormc: `uP` are representation neurons, `uI` are error neurons
- ann: `uP` are neurons, `uI` are only auxiliary variables
- sacramento2018: `uP` are pyramidal neurons, `uI` are interneurons
- dPC: `uP` are pyramidal neurons, `uI` are only auxiliary variables

Variables (based on variable names of errormc):
- base variable `uP` is the array of vectors of somatic potentials of representation units
- base variable `uI` is the array of vectors of somatic potentials of error neurons
- variables with `_breve` are the lookahead of base variables
- `rX_breve` is the instantaneous rate based on a the corresponding lookahead voltage `uX_breve`
- `WPP` are weights connecting representation neurons in one layer to the next (including input to first layer)
- `BII` are weights connecting error neurons in one layer to error cells in layer below
- `WIP` are lateral weights from representation cells to error neurons
- `BPI` are lateral weights form error neurons to representation cells


Keep in mind that the `layer` variable always starts from zero. So e.g. for the interneuron recordings, `uI_time_series[-1][0][1]` returns the voltage of the second neuron in the final (and only) layer of interneurons at the end of training.






# Commands to reproduce plots

Note that the code provided here runs on the local machine for compatibility. Due to long simulation times, it is best to run jobs as slurm jobs.
To do so, modify the slurm file template in this folder, and replace any instance of `python runner.py` with `sbatch slurm_script.sh python runner.py` (and similarly for `DMS_task.py` etc).

## Figure 2a: cartpole

`python runner.py --params experiments/Fig2a_cartpole_ideal_emc/params_errormc.json`

`python runner.py --params experiments/Fig2a_cartpole_tanh_emc/params_errormc.json`

For plots, open `cart-pole plot.ipynb` in jupyter lab.

## Figure 2b: DMS task

`python DMS_task.py`

Results are saved in `experiments/Fig2b_DMS_plots/`

For plots, open `generate figs from runs.ipynb` in jupyter lab.

## Figure 3: multilayer comparison

This is a large simulation, so we run a batch command:

`bash run_Fig3_multilayer_comparison.sh`

> [!CAUTION]
> This will start many jobs and can freeze your system. Use slurm to manage jobs (see above). If you want to quit all running jobs, run `killall Python` (make sure no other python scripts are running which you don't want to kill).

Results are saved in `experiments/Fig3_multilayer_comparison/`.

For plots, open `generate figs from runs.ipynb` in jupyter lab.

For appendix runs:

`bash run_FigA1_multilayer_comparison_hierarchical.sh`

`bash run_FigA2_multilayer_comparison_ideal_lat_inh.sh`

For non-linear implementation of dendritic hierarchical PC:

`bash run_FigC1_nonlinear_dPC.sh`

## Figure 4: populations

This is a large simulation, so we run a batch command:

`bash run_Fig4_populations.sh`

For plots, open `generate figs from runs.ipynb` in jupyter lab.

## Figure 5: noise

This is a large simulation, so we run a batch command:

`bash run_Fig5_noise.sh`

For plots, open `generate figs from runs.ipynb` in jupyter lab.

## Figure 8: varphi' transfer

Open `varphi' transfer.ipynb` in jupyter lab.

For plots, open `generate figs from runs.ipynb` in jupyter lab.


