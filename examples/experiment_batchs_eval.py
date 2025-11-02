from confidence_sampling import pre_train, sampler, simulate_train
from confidence_sampling import utils as cs_utils
import numpy as np
import torch
import matplotlib.pyplot as plt
import scanpy as sc
import scipy as sp
import pandas as pd
from scipy.stats import kendalltau
import requests, json
import argparse

# input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config_experiment_file', type=str, required=True)
args = parser.parse_args()
config_experiment_file = args.config_experiment_file

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data preparation
## Read data
datasets_folder = '/cs/labs/mornitzan/yam.arieli/datasets/'

# Input
with open(config_experiment_file, 'r') as f:
    config_exp =  json.load(f)
train_file = config_exp['train_file']
test_file = config_exp['test_file']
results_file = config_exp['results_file']
label_column = config_exp['label_column']

mislabeled = config_exp['mislabeled']
genes_std_noise = config_exp['gene_noise']
pre_train_params = config_exp['pre_train_params']
iterations_full = config_exp['iterations_full']
actual_train_params = config_exp['actual_train_params']

# Read adata_train adata_test
adata_train = sc.read_h5ad(datasets_folder + train_file)
adata_test = sc.read_h5ad(datasets_folder + test_file)

# Clean trainset
sc.pp.filter_cells(adata_train, min_genes=200)
sc.pp.filter_genes(adata_train, min_cells=3)

sc.pp.highly_variable_genes(
    adata_train,
    n_top_genes=2500,
    flavor="cell_ranger"
)
adata_train = adata_train[:, adata_train.var["highly_variable"]].copy()

## Add noise
adata_train = cs_utils.mislabel_trainset(adata_train, label_column, mislabeled)
adata_train = cs_utils.add_noise_to_genes(adata_train, genes_std_noise)

## Add **`y`** column
# Get the unique categories and sort them (optional)
categories = adata_train.obs[label_column].unique()

# Create a mapping: category -> integer
cat2num = {cat: i for i, cat in enumerate(categories)}

# Apply mapping to create a numeric column
adata_train.obs['y'] = adata_train.obs[label_column].map(cat2num)

# Meta data
n_obs = adata_train.shape[0]
n_genes = adata_train.shape[1]
n_classes = adata_train.obs[label_column].nunique()

# Actual data
X = adata_train.X
if not isinstance(X, np.ndarray):
    X = X.toarray()
y = np.array(adata_train.obs['y'].values)
X_dtype = torch.tensor(X[0,0]).dtype

# Define pre_train model
layer_sizes = [n_genes, int(n_genes / 2), int(n_genes / 4), int(n_genes / 8), n_classes]
pre_train_params['layer_sizes'] = layer_sizes
pre_train_model = pre_train.LoRANet(layer_sizes=layer_sizes)
pre_train_model = pre_train_model.type(dst_type=X_dtype).to(device)

# Execute the pre_train
confidence, final_probs, forgetting_counts, train_losses = pre_train.pretrain_and_get_signals(
    pre_train_model, X, y, device, batch_size=pre_train_params['batch_size'], lr=pre_train_params['lr'], epochs=pre_train_params['epochs'], weighted_sampler=True)

# Prepare test set
adata_test = cs_utils.fix_test_genes(adata_train, adata_test, label_column)
adata_test = cs_utils.add_target_y_to_test(adata_train, adata_test, label_column)

##################################################################################
############################### Actual Train #####################################
##################################################################################
sample_size = int(n_obs * actual_train_params['sample_frac'])
specific_settings = {}

from confidence_sampling.simulate_train import *

class ComplexNet3(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, droupout_p=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.ln1 = nn.LayerNorm(64)

        self.fc2 = nn.Linear(64, 32)
        self.ln2 = nn.LayerNorm(32)

        self.dropout = nn.Dropout(droupout_p)
        self.fc_out = nn.Linear(32, output_dim)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)

        x = F.relu(self.ln2(self.fc2(x)))
        x = self.dropout(x)

        return self.fc_out(x)


def do_train(X_tensor, y_tensor, num_classes,
             X_test_tensor, y_test_labels, device,
             hidden_dim=1024, epochs=100, batch_size=16, lr=1e-4, droupout_p=0.1, model=None, do_print=False):
    input_dim = X_tensor.shape[1]
    
    if not model:
        model = ComplexNet3(input_dim, hidden_dim, num_classes, droupout_p=droupout_p).to(device)

    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=lr/10)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=lr/10)
    criterion = nn.CrossEntropyLoss()

    losses = []
    test_metrics = []
    
    for epoch in range(epochs):
        epoch_losses = train_one_epoch(model, optimizer, criterion, X_tensor, y_tensor, batch_size)

        # Predict probabilities for adata_test
        with torch.no_grad():
            test_probs = torch.softmax(model(X_test_tensor), dim=1)
        
        acc_test, recall_test, f1_per_class, f1_macro, f1_weighted = eval_result(test_probs, y_test_labels, do_print=False)
        test_metrics.append([acc_test, recall_test, f1_per_class, f1_macro, f1_weighted])
        
        if do_print and ((epoch+1) % 10 == 0 or epoch == 0):
            print(f"Epoch {epoch+1}/{epochs}, Loss: {np.mean(epoch_losses):.4f}")
        
        losses.append(epoch_losses)

    # Get probabilities
    with torch.no_grad():
        probs = torch.softmax(model(X_tensor), dim=1)
    
    return model, probs, np.array(losses), test_metrics

def simulate_train(adata_train, adata_test, device, epochs=80, batch_size=16, lr=1e-4, hidden_dim=1024, droupout_p=0.1):
    adata_temp = adata_train.copy()
    X_tensor, y_tensor, num_classes = prepare_train_tensors(adata_temp, device)
    X_test_tensor, y_test_tensor, num_classes_test = prepare_train_tensors(adata_test, device)
    y_test_labels = adata_test.obs["y"].astype("category").cat.codes.values
    model, probs, losses, test_metrics = do_train(X_tensor, y_tensor, num_classes,
                                                  X_test_tensor, y_test_labels,
                                                  device, hidden_dim=hidden_dim, epochs=epochs,
                                                  batch_size=batch_size, lr=lr, droupout_p=droupout_p, do_print=True)

    return model, probs, losses, test_metrics
"""
# Bootstrap differnt methods
test_metrics_lrdy_all_03 = []

power = 0.3
for iteration in range(actual_train_params['iterations']):
    print(f'iteration: {iteration}')
    sample_indeces = sampler.stratified_weighted_sample(y, confidence, M=sample_size, sample_func=sampler.sample_by_conf, power=power)
    adata_lrdy_sample = adata_train[sample_indeces, :].copy()
    model, probs, losses, test_metrics_lrdy = simulate_train.simulate_train(
                                            adata_lrdy_sample, adata_test, device, epochs=actual_train_params['epochs'],
                                            lr=actual_train_params['lr'], batch_size=actual_train_params['batch_size'], hidden_dim=actual_train_params['hidden_dim'])
    accuracies = [epoch_results[0] for epoch_results in test_metrics_lrdy]
    test_metrics_lrdy_all_03.append(test_metrics_lrdy)

    print(accuracies[-1], end='\n-------------------------\n\n')

specific_settings['test_metrics_lrdy_all_03'] = test_metrics_lrdy_all_03
"""
test_metrics_lrdy_all_04 = []
power = 0.4
for iteration in range(actual_train_params['iterations']):
    print(f'iteration: {iteration}')
    sample_indeces = sampler.stratified_weighted_sample(y, confidence, M=sample_size, sample_func=sampler.sample_by_conf, power=power)
    adata_lrdy_sample = adata_train[sample_indeces, :].copy()
    model, probs, losses, test_metrics_lrdy = simulate_train(
                                            adata_lrdy_sample, adata_test, device, epochs=actual_train_params['epochs'],
                                            lr=actual_train_params['lr'], batch_size=actual_train_params['batch_size'], hidden_dim=actual_train_params['hidden_dim'])
    accuracies = [epoch_results[0] for epoch_results in test_metrics_lrdy]
    test_metrics_lrdy_all_04.append(test_metrics_lrdy)

    print(accuracies[-1], end='\n-------------------------\n\n')

specific_settings['test_metrics_lrdy_all_04'] = test_metrics_lrdy_all_04

test_metrics_lrdy_all_05 = []

power = 0.5
for iteration in range(actual_train_params['iterations']):
    print(f'iteration: {iteration}')
    sample_indeces = sampler.stratified_weighted_sample(y, confidence, M=sample_size, sample_func=sampler.sample_by_conf, power=power)
    adata_lrdy_sample = adata_train[sample_indeces, :].copy()
    model, probs, losses, test_metrics_lrdy = simulate_train(
                                            adata_lrdy_sample, adata_test, device, epochs=actual_train_params['epochs'],
                                            lr=actual_train_params['lr'], batch_size=actual_train_params['batch_size'], hidden_dim=actual_train_params['hidden_dim'])
    accuracies = [epoch_results[0] for epoch_results in test_metrics_lrdy]
    test_metrics_lrdy_all_05.append(test_metrics_lrdy)

    print(accuracies[-1], end='\n-------------------------\n\n')

specific_settings['test_metrics_lrdy_all_05'] = test_metrics_lrdy_all_05

## Random
test_accs_rand = []
test_metrics_rand_all = []
for iteration in range(actual_train_params['iterations']):
    print(f'iteration: {iteration}')
    sample_indeces = sampler.stratified_random_sample(y, sample_size)
    adata_rand_sample = adata_train[sample_indeces, :].copy()
    model, probs, losses, test_metrics_rand = simulate_train(
                                            adata_rand_sample, adata_test, device, epochs=actual_train_params['epochs'],
                                            lr=actual_train_params['lr'], batch_size=actual_train_params['batch_size'], hidden_dim=actual_train_params['hidden_dim'])
    accuracies = [epoch_results[0] for epoch_results in test_metrics_rand]
    test_accs_rand.append(accuracies)
    test_metrics_rand_all.append(test_metrics_rand)
    print(accuracies[-1], end='\n-------------------------\n\n')

specific_settings['test_metrics_rand_all'] = test_metrics_rand_all

## full train set
test_accs_full = []
test_metrics_full_all = []

for iteration in range(iterations_full):
    print(f'iteration: {iteration}')
    model, probs, losses, test_metrics_full = simulate_train(
                                            adata_train, adata_test, device, epochs=actual_train_params['epochs'],
                                            lr=actual_train_params['lr'], batch_size=actual_train_params['batch_size'], hidden_dim=actual_train_params['hidden_dim'])
    accuracies = [epoch_results[0] for epoch_results in test_metrics_full]
    test_accs_full.append(accuracies)
    test_metrics_full_all.append(test_metrics_full)
    print(accuracies[-1], end='\n-------------------------\n\n')
specific_settings['test_metrics_full_all'] = test_metrics_full_all

## Save
# Convert numpy objects to Python objects
def convert_np_to_json(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.int64, np.int32)):
        return int(o)
    if isinstance(o, (np.float32, np.float64)):
        return float(o)
    raise TypeError

with open(f'/cs/labs/mornitzan/yam.arieli/confident_sampling_experiment/results/{results_file}.json', 'w') as f:
    json.dump(specific_settings, f, indent=4, default=convert_np_to_json)
