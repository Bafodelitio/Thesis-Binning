import os
import pandas as pd
import torch
import copy
import numpy as np
#import torch.nn.functional as F
from torch_geometric.transforms import NormalizeFeatures
from util import AssemblyDataset, apply_pca_to_node_features
#from sklearn.metrics import silhouette_score
#from sklearn.manifold import TSNE
from SAGE_unsup_aux import create_pyg_graph_from_assembly, hyperparameter_combination_tuning, hyperparameter_tuning, calculate_avg_edges_per_node

# Baseline Cluster        

# DATA LOADING & PRE-PROCESSING


dataset = AssemblyDataset(
    name="strong100", #"aale", 
    data_dir="strong100", #"aale",
    fastafile="assembly.fasta",       
    graphfile="assembly_graph.gfa", 
    depthfile="assembly_depth.txt",
    scgfile="marker_gene_stats.tsv",
    labelsfile=None,
    featuresfile="features.tsv",
    cache_dir="strong100_cache" #"aale_cache" 
)

if(dataset.check_cache()):
    dataset.read_cache()
else:
    dataset.read_assembly()
                
#coverage = dataset.calculate_coverage_threshold()

data = create_pyg_graph_from_assembly(dataset)

#calculate_avg_edges_per_node(data)
#print(f"THIS IS THE SHAPE: n nodes: {data.num_nodes}, n edges: {data.num_edges}")
    #print(f"These are the attributes: {data.keys()}")
    #print(f"These are the sequence lengths: {dataset.node_lengths}")
#print(f"These are the k-mers: {dataset.node_kmers[350]}")
#print(f"These are the sequences: {np.shape(dataset.node_seqs)}")
#print(f"NUM_NODES {type(data.num_nodes)}")
#print(f"NUMBER OF NODES: GRAPH {data.num_nodes} VS DATASET {len(dataset.graph_nodes)}")
#print(f"NUMBER OF EDGES: GRAPH {data.num_edges} VS DATASET {len(dataset.graph_paths)}")
#print(f"SHAPE - NODE NAMES {dataset.node_names}")
#print(f"SHAPE - GRAPH NODES {data.node_names}")
print(f"NODE DEPTHS {np.shape(dataset.node_depths)}")

#print(f"SHAPE - Graph Nodes {np.shape(dataset.graph_nodes)}")
#print(f"NODE NAME: {dataset.node_names[50]} vs GRAPH NODE: {dataset.graph_nodes[50]}")
    #print(f"SHAPE - NODE KMERS {np.shape(dataset.node_kmers)}")
    
#dataset.print_stats()

# Add node features if not present (using "kmers" and "length" features)
# data.x = torch.eye(data.num_nodes)
# Assuming "kmers" and "length" are tensors of the same length (num_nodes)
#kmers = data.kmers  # Assuming this is a tensor of shape [num_nodes, kmer_feature_dim]
#length = data.length.unsqueeze(1)  # Reshape "length" to be [num_nodes, 1] if it's a 1D tensor

# Concatenate "kmers" and "length" along the feature dimension
#data.x = torch.cat([kmers, length], dim=1)   
    
# Normalize features
transform = NormalizeFeatures()
data = transform(data)

# Reduce kmer dimensionality
data = apply_pca_to_node_features(data)

# Ensure the Data object has edge_index attribute
assert data.edge_index is not None

base_config = {
    'batch_size': 256,
    'hidden_channels': 256,
    'out_channels': 32,
    'num_layers': 4,
    'dropout': 0.1,  
    'lr': 0.005,
    'weight_decay': 0.01,
    'margin': 0.07,
    'num_neighbors': 3,
    'epochs': 50, 
    'num_components': None
} 

"""
base_config = {
    'batch_size': 256,
    'hidden_channels': 128,  # Reduced hidden channels
    'out_channels': 32,
    'num_layers': 3,  # Reduced number of layers
    'dropout': 0,  # Increased dropout
    'lr': 0.0001,  # Reduced learning rate
    'weight_decay': 1e-4,  # Increased weight decay
    'margin': 0.08,  # Increased margin
    'num_neighbors': 4,
    'epochs': 50,
    'num_components': None
}

base_config = {
    'batch_size': 256,
    'hidden_channels': 128,  # Reduced hidden channels
    'out_channels': 32,
    'num_layers': 3,  # Reduced number of layers
    'dropout': 0.5,  # Increased dropout
    'lr': 0.001,  # Reduced learning rate
    'weight_decay': 1e-4,  # Increased weight decay
    'margin': 0.1,  # Increased margin
    'num_neighbors': 4,
    'epochs': 50,
    'num_components': None
}

 BASE STRONG100
base_config = {
    'batch_size': 256,
    'hidden_channels': 256,
    'out_channels': 32,
    'num_layers': 4,
    'dropout': 0,  
    'lr': 0.005,
    'weight_decay': 0,
    'margin': 0.08,
    'num_neighbors': 3,
    'epochs': 50, 
    'num_components': None
} 


# AALE
base_config = {
    'batch_size': 512,           # Smaller batch sizes for better embedding quality
    'hidden_channels': 128,      # Reduce to prevent overfitting and encourage generalisation
    'out_channels': 64,          # Increase output dimension for better separation
    'num_layers': 3,             # Fewer layers for computational stability
    'dropout': 0.2,              # Introduce regularisation to improve model robustness
    'lr': 0.001,                 # Moderate learning rate to balance convergence speed
    'weight_decay': 0.0001,      # Small weight decay to prevent overfitting
    'margin': 0.1,               # Slightly increase the margin for better contrast
    'num_neighbors': 5,          # Increase neighbours for better structural context
    'epochs': 50,                # Increase epochs to allow more gradual convergence
    'num_components': None
}
"""

# Define the range of values for each parameter
#batch_size_values = [32, 64, 128, 256, 512]
#hidden_channels_values = [32, 64, 128, 256, 512]
#out_channels_values = [8, 16, 32, 64, 128, 256]
#num_layers_values = [2, 3, 4, 5] 
#lr_values = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
#margin_values = [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.12]
#num_neighbor_values = [3, 4, 5, 6]
#epoch_values = [50, 100, 150, 200, 300] 
#weight_decay_values = [0, 0.01, 0.02, 0.05, 0.08, 0.1, 0.15]
#dropout_values = [0, 0.2, 0.25, 0.3, 0.35, 0.4]
#num_components_values = [10, 25, 30, 40, 50, 60, 75, 100]

# Define the range of values for each parameter
batch_size_values = [64, 128, 256, 512] # Focus on medium to large values for smoother gradients.
hidden_channels_values = [64, 128, 256, 512]  # Narrowed to commonly effective powers of 2.
out_channels_values = [16, 32, 64, 100, 128]  # Removed extreme values; focus on middle range.
num_layers_values = [2, 3, 4]  # Simplified to a practical range for most GNNs.
lr_values = [0.0001, 0.0005, 0.001, 0.005]  # Coarse search around typical learning rates.
margin_values = [0.05, 0.06, 0.07, 0.08, 0.09]  # Slightly narrowed range for clustering tasks.
num_neighbor_values = [3, 4, 5]  # Focused on smaller neighbourhoods for efficiency.
epoch_values = [50, 100, 200]  # Removed overly high values for initial tuning.
weight_decay_values = [0.01, 0.02, 0.05]  # Prioritised smaller values for regularisation.
dropout_values = [0.1, 0.2, 0.3, 0.4]  # Reduced to common values for preventing overfitting.



# Specify the base directory for saving model results
base_path = "E:/FCT/Tese/Code/checkm2/checkm2/tests/s100_2"
#base_path = "E:/FCT/Tese/Code/checkm2/checkm2/tests/temp"


other_params = {
    'batch_size': batch_size_values,
    'hidden_channels': hidden_channels_values,
    'out_channels': out_channels_values,
    'num_layers': num_layers_values,
    'margin': margin_values,
    'num_neighbors': num_neighbor_values,
    'epochs': epoch_values,
    'weight_decay': weight_decay_values,
    'dropout': dropout_values  
}

"""x
# Run hyperparameter tuning for each parameter
for value in num_components_values:
    # Reduce kmer dimensionality    
    data_copy = copy.deepcopy(data)
    data_copy = apply_pca_to_node_features(data_copy)
    hyperparameter_tuning(dataset, data_copy, 'num_components', [value], base_config, base_path, update_edges=False)
"""

#data = apply_pca_to_node_features(data)
#for _ in range(4):
#hyperparameter_tuning(dataset, data, 'lr', None, base_config, base_path, update_edges=False)#, coverage=coverage)



#hyperparameter_tuning(dataset, data, 'lr', None, base_config, base_path, update_edges=False)



hyperparameter_tuning(dataset, data, 'lr', lr_values, base_config, base_path, update_edges=False)
hyperparameter_tuning(dataset, data, 'margin', margin_values, base_config, base_path, update_edges=False)
hyperparameter_tuning(dataset, data, 'batch_size', batch_size_values, base_config, base_path, update_edges=False)
hyperparameter_tuning(dataset, data, 'num_neighbors', num_neighbor_values, base_config, base_path, update_edges=False)
hyperparameter_tuning(dataset, data, 'num_layers', num_layers_values, base_config, base_path, update_edges=False)
hyperparameter_tuning(dataset, data, 'hidden_channels', hidden_channels_values, base_config, base_path, update_edges=False)
hyperparameter_tuning(dataset, data, 'out_channels', out_channels_values, base_config, base_path, update_edges=False)
hyperparameter_tuning(dataset, data, 'epochs', epoch_values, base_config, base_path, update_edges=False)
hyperparameter_tuning(dataset, data, 'weight_decay', weight_decay_values, base_config, base_path, update_edges=False)
hyperparameter_tuning(dataset, data, 'dropout', dropout_values, base_config, base_path, update_edges=False)


"""
import seaborn as sns
import matplotlib.pyplot as plt

# Access depth values
depth_values = data.depths.cpu().numpy().squeeze()  # Shape: (num_nodes,)

# Check depth values
print("Depth values:", depth_values)
print("Min depth:", np.min(depth_values))
print("Max depth:", np.max(depth_values))
print("Mean depth:", np.mean(depth_values))
print("Standard deviation:", np.std(depth_values))
print("Unique depth values:", np.unique(depth_values))

# Clip outliers (optional but recommended)
depth_values = np.clip(depth_values, -3, 3)  # Clip values to [-3, 3]

# Compute pairwise depth differences
depth_diff_matrix = np.abs(depth_values[:, None] - depth_values[None, :])  # Shape: (num_nodes, num_nodes)

# Compute pairwise depth similarities (Gaussian kernel)
sigma = 1.0  # Bandwidth parameter (adjust as needed)
depth_similarity_matrix = np.exp(-depth_diff_matrix**2 / (2 * sigma**2))  # Shape: (num_nodes, num_nodes)

depth_similarities = []
for u in range(len(data.depths)):
    for v in range(len(data.depths)):
        if u != v:
            similarity = depth_similarity(u, v, depth_values)
            depth_similarities.append(similarity)
sns.histplot(depth_similarities, bins=50, kde=True)
plt.xlabel("Depth Similarity")
plt.ylabel("Frequency")
plt.title("Distribution of Depth Similarities")
plt.show()
# Plot pairwise depth differences
sns.histplot(depth_diff_matrix.flatten(), bins=50, kde=True)
plt.xlabel("Pairwise Depth Difference")
plt.ylabel("Frequency")
plt.title("Distribution of Pairwise Depth Differences")
plt.show()

# Plot pairwise depth similarities
sns.histplot(depth_similarity_matrix.flatten(), bins=50, kde=True)
plt.xlabel("Pairwise Depth Similarity")
plt.ylabel("Frequency")
plt.title("Distribution of Pairwise Depth Similarities")
plt.show()"""











"""
# Reduce kmer dimensionality
if(base_config['num_components'] != None):
    data = apply_pca_to_node_features(data)
for i in range(2):
    if(i == 1):
        base_config['num_components'] = 50
        hyperparameter_tuning(dataset, data, 'num_components', num_components_values, base_config, base_path, update_edges=True)
    hyperparameter_tuning(dataset, data, 'batch_size', batch_size_values, base_config, base_path, update_edges=True)
    hyperparameter_tuning(dataset, data, 'hidden_channels', hidden_channels_values, base_config, base_path, update_edges=True)
    hyperparameter_tuning(dataset, data, 'out_channels', out_channels_values, base_config, base_path, update_edges=True)
    hyperparameter_tuning(dataset, data, 'num_layers', num_layers_values, base_config, base_path, update_edges=True)
    hyperparameter_tuning(dataset, data, 'lr', lr_values, base_config, base_path, update_edges=True)
    hyperparameter_tuning(dataset, data, 'margin', margin_values, base_config, base_path, update_edges=True)
    hyperparameter_tuning(dataset, data, 'num_neighbors', num_neighbor_values, base_config, base_path, update_edges=True)
    hyperparameter_tuning(dataset, data, 'epochs', epoch_values, base_config, base_path, update_edges=True)
    hyperparameter_tuning(dataset, data, 'weight_decay', weight_decay_values, base_config, base_path, update_edges=True)
    hyperparameter_tuning(dataset, data, 'dropout', dropout_values, base_config, base_path, update_edges=True)
"""
"""
hyperparameter_combination_tuning(
    dataset,
    data,
    main_parameter='lr',
    main_values=lr_values,
    other_parameters=other_params,
    base_config=base_config,
    base_path=base_path
)

other_params.pop('hidden_channels', None)
other_params['lr'] = lr_values

hyperparameter_combination_tuning(
    dataset,
    data,
    main_parameter='hidden_channels',
    main_values=hidden_channels_values,
    other_parameters=other_params,
    base_config=base_config,
    base_path=base_path
)

other_params.pop('num_layers', None)
other_params['hidden_channels'] = hidden_channels_values

hyperparameter_combination_tuning(
    dataset,
    data,
    main_parameter='num_layers',
    main_values=num_layers_values,
    other_parameters=other_params,
    base_config=base_config,
    base_path=base_path
)
"""