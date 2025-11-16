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
y = torch.tensor(np.array(adata_train.obs['y'].values), dtype=torch.long)
X_dtype = torch.tensor(X[0,0]).dtype

# Define pre_train model
layer_sizes = [n_genes, int(n_genes / 2), int(n_genes / 4), int(n_genes / 8), n_classes]
pre_train_params['layer_sizes'] = layer_sizes
pre_train_model = pre_train.LoRANet(layer_sizes=layer_sizes)
pre_train_model = pre_train_model.type(dst_type=X_dtype).to(device)

# Execute the pre_train
confidence, final_probs, forgetting_counts, train_losses = pre_train.pretrain_and_get_signals(
    pre_train_model, X, y, device, batch_size=pre_train_params['batch_size'], lr=pre_train_params['lr'], epochs=pre_train_params['epochs'], weighted_sampler=True)

####### Add Ambiguous Bias ##### BETA !!!!
confidence = np.power(0.5-np.abs(confidence - 0.5), 2) + confidence

# Prepare test set
adata_test = cs_utils.fix_test_genes(adata_train, adata_test, label_column)
adata_test = cs_utils.add_target_y_to_test(adata_train, adata_test, label_column)

##################################################################################
############################### Actual Train #####################################
##################################################################################
sample_size = int(n_obs * actual_train_params['sample_frac'])
specific_settings = {}

def run_sampling_experiment(
    sampler_func,
    sample_args,
    label,
    adata_train,
    adata_test,
    device,
    actual_train_params,
    iterations,
):
    """Run repeated sampling + training experiments and collect test metrics."""
    all_metrics = []

    for iteration in range(iterations):
        print(f"iteration: {iteration}")
        sample_indices = sampler_func(**sample_args) if sampler_func else None
        adata_sample = adata_train[sample_indices, :].copy() if sample_indices is not None else adata_train

        model, probs, losses, test_metrics = simulate_train.simulate_train(
            adata_sample,
            adata_test,
            device,
            epochs=actual_train_params["epochs"],
            lr=actual_train_params["lr"],
            batch_size=actual_train_params["batch_size"],
            hidden_dim=actual_train_params["hidden_dim"],
        )

        accuracies = [epoch_results[0] for epoch_results in test_metrics]
        all_metrics.append(test_metrics)

        print(accuracies[-1], end="\n-------------------------\n\n")

    return all_metrics


# === Run all your experiments ===
iterations = actual_train_params["iterations"]

# 1️⃣ Conf-based sampling (power=0.4)
specific_settings["test_metrics_lrdy_all_04"] = run_sampling_experiment(
    sampler_func=sampler.stratified_weighted_sample,
    sample_args=dict(y=y, confidence=confidence, M=sample_size, sample_func=sampler.sample_by_conf, power=0.4),
    label="conf_0.4",
    adata_train=adata_train,
    adata_test=adata_test,
    device=device,
    actual_train_params=actual_train_params,
    iterations=iterations,
)

# 2️⃣ Conf-based sampling (power=0.5)
specific_settings["test_metrics_lrdy_all_05"] = run_sampling_experiment(
    sampler_func=sampler.stratified_weighted_sample,
    sample_args=dict(y=y, confidence=confidence, M=sample_size, sample_func=sampler.sample_by_conf, power=0.5),
    label="conf_0.5",
    adata_train=adata_train,
    adata_test=adata_test,
    device=device,
    actual_train_params=actual_train_params,
    iterations=iterations,
)

# 3️⃣ Random sampling
specific_settings["test_metrics_rand_all"] = run_sampling_experiment(
    sampler_func=sampler.stratified_random_sample,
    sample_args=dict(y=y, M=sample_size),
    label="random",
    adata_train=adata_train,
    adata_test=adata_test,
    device=device,
    actual_train_params=actual_train_params,
    iterations=iterations,
)

# 4️⃣ Full training set (no sampling)
specific_settings["test_metrics_full_all"] = run_sampling_experiment(
    sampler_func=None,  # no sampling
    sample_args={},
    label="full",
    adata_train=adata_train,
    adata_test=adata_test,
    device=device,
    actual_train_params=actual_train_params,
    iterations=iterations_full,
)

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