# Code repository for "‘Backpropagation and the brain’ realized in cortical error neuron microcircuits"

This repo contains experiments for error neuron microcircuits on .
The paper "‘Backpropagation and the brain’ realized in cortical error neuron microcircuits" by Kevin Max, Ismael Jaras, Arno Granier, Katharina A. Wilmes, Mihai A. Petrovici is available at https://www.biorxiv.org/content/10.1101/2025.07.11.664263v1


To get started, open a terminal and run:
```
git clone https://github.com/kma-code/Error-Neuron-Microcircuits
cd Error-Neuron-Microcircuits/numpy_model
python3 -m venv MCenv
source MCenv/bin/activate
pip3 install -r requirements.txt
python -m ipykernel install --user --name=MCenv 
```
Go to [https://github.com/kma-code/Error-Neuron-Microcircuits/numpy_model](https://github.com/kma-code/Error-Neuron-Microcircuits/tree/main/numpy_model#commands-to-reproduce-plots) to run simulations reproducing the figures in the paper.
