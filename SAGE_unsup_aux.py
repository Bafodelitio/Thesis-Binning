import time
import random
import torch
import os
from collections import Counter
#import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GraphSAGE
from torch_geometric.utils import to_undirected, negative_sampling
from sklearn.model_selection import train_test_split
from util import plot_and_save_loss, plot_and_save_loss_wval, get_last_iteration_number, compare_graphs, plot_pca_clusters, plot_pca_embeddings_simple, evaluate_contig_sets
#from sklearn.metrics import silhouette_score
#from sklearn.manifold import TSNE
from vamb_clustering import cluster
import numpy as np
#import networkx as nx 
from collections import defaultdict
#import pickle
#import matplotlib.pyplot as plt
import pandas as pd
#G, kmer_to_id = open_gfa_file("strong100/assembly_graph.gfa")
#print("K-mer ID's: "+ str(kmer_to_id))

def encode(node_names):
    return [torch.tensor(name.encode('utf-8') for name in node_names)]

def decode(node_ids):
    return [bytes(tensor.tolist()).decode('utf-8') for tensor in node_ids]  

def create_pyg_graph_from_assembly(dataset):
    # Prepare node features separately
    
    node_length = torch.tensor(dataset.node_lengths, dtype=torch.float).unsqueeze(1)
    node_depths = torch.tensor(dataset.node_depths, dtype=torch.float)  # New feature
    node_kmers = torch.tensor(dataset.node_kmers, dtype=torch.float)
    # Prepare node ids as a tensor (assuming node_names are unique identifiers)
    node_ids = torch.tensor([dataset.node_names.index(name) for name in dataset.node_names], dtype=torch.long)
    
    # Prepare edge index and edge weights
    edge_index = to_undirected(torch.tensor([dataset.edges_src, dataset.edges_dst], dtype=torch.long))
    
    # Create the PyTorch Geometric data object
    data = Data(edge_index=edge_index)

    # Assign the node features to the data object
    data.length = node_length
    data.depths = node_depths
    data.kmers = node_kmers
    data.node_names = node_ids #encode(dataset.node_names)
    #data.num_nodes = node_ids.size(0)
    
    if 'x' not in data:
        data.x = torch.cat([data.kmers, data.depths], dim=1)
    
    return data

# Convert NetworkX graph to PyTorch Geometric
#data = from_networkx(G)

def calculate_avg_edges_per_node(data):
    # Initialize a dictionary to count edges per node
    edge_count = defaultdict(int)

    # Iterate through edge_index to count edges for each node
    for source, target in data.edge_index.t().tolist():
        edge_count[source] += 1
        edge_count[target] += 1

    # Convert the counts to a list
    edge_counts = list(edge_count.values())

    # Calculate the average number of edges per node
    avg_edges_per_node = np.mean(edge_counts)

    # Print the result
    print(f"Average number of edges per node: {avg_edges_per_node:.2f}")

    # Calculate and print the distribution of neighbors per node
    distribution = np.bincount(edge_counts)
    print(f"Distribution of neighbors per node: {distribution}")
    
#calculate_avg_edges_per_node(data)
                

def create_train_val_split(data, val_ratio=0.2):
    """
    Create a train/validation node split without altering the edge structure.

    Parameters:
    - data (Data): The original PyTorch Geometric Data object.
    - val_ratio (float): The proportion of nodes to use for validation.

    Returns:
    - train_mask (torch.Tensor): Mask for training nodes.
    - val_mask (torch.Tensor): Mask for validation nodes.
    """
    # Split node indices into training and validation
    train_idx, val_idx = train_test_split(np.arange(data.num_nodes), test_size=val_ratio, random_state=42)
    
    # Create boolean masks for training and validation
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    
    return train_mask, val_mask


def generate_pairs(data, device):
    edge_index = data.edge_index.cpu().numpy()
    num_nodes = data.num_nodes

    pos_pairs = []
    for i in range(edge_index.shape[1]):
        pos_pairs.append((edge_index[0, i], edge_index[1, i]))
    
    pos_pairs = torch.tensor(pos_pairs, dtype=torch.long, device=device)
    
    neg_pairs = []
    for _ in range(len(pos_pairs)):
        while True:
            u = random.randint(0, num_nodes - 1)
            v = random.randint(0, num_nodes - 1)
            if u != v and not any((u == edge_index[0, j] and v == edge_index[1, j]) 
                               or (u == edge_index[1, j] and v == edge_index[0, j]) 
                               for j in range(edge_index.shape[1])):
                neg_pairs.append((u, v))
                break
    
    neg_pairs = torch.tensor(neg_pairs, dtype=torch.long, device=device)
    
    return pos_pairs, neg_pairs

"""

def depth_similarity(u, v, depth_values, sigma=1.0):
    #Compute depth similarity between two nodes using a Gaussian kernel.
    depth_diff = np.abs(depth_values[u] - depth_values[v])
    return np.exp(-depth_diff**2 / (2 * sigma**2))

def generate_pairs(data, device, depth_threshold=0.8, neg_depth_threshold=0.8):  # Adjusted thresholds
    edge_index = data.edge_index.cpu().numpy()
    num_nodes = data.num_nodes
    depth_values = data.x[:, -1].cpu().numpy()  # Assuming depth is the last feature in data.x

    # Generate positive pairs
    pos_pairs = []
    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i], edge_index[1, i]
        similarity = depth_similarity(u, v, depth_values)  # Compute depth similarity
        if similarity >= depth_threshold:  # Ensure high similarity
            pos_pairs.append((u, v))
    
    pos_pairs = torch.tensor(pos_pairs, dtype=torch.long, device=device)

    # If no positive pairs, return empty tensors
    if len(pos_pairs) == 0:
        return pos_pairs, torch.tensor([], dtype=torch.long, device=device)

    # Generate negative pairs
    # Precompute a set of "safe" negative candidates
    neg_candidates = []
    for u in range(num_nodes):
        for v in range(num_nodes):
            if u != v:
                similarity = depth_similarity(u, v, depth_values)  # Compute depth similarity
                if similarity <= neg_depth_threshold:  # Ensure low similarity
                    # Relaxed condition: Allow some overlap with positive pairs
                    if random.random() < 0.5:  # Randomly include some pairs
                        neg_candidates.append((u, v))
    
    # Sample negative pairs from the precomputed candidates
    num_neg_samples = len(pos_pairs)  # Match the number of positive pairs
    if len(neg_candidates) > num_neg_samples:
        sampled_indices = random.sample(range(len(neg_candidates)), num_neg_samples)
        neg_pairs = [neg_candidates[i] for i in sampled_indices]
    else:
        neg_pairs = neg_candidates  # Use all candidates if there are fewer than required
    
    neg_pairs = torch.tensor(neg_pairs, dtype=torch.long, device=device)
    
    return pos_pairs, neg_pairs
"""

def contrastive_loss(h, pos_pairs, neg_pairs, margin):
    if pos_pairs.size(0) > 0:  # If there are positive pairs
        pos_loss = (h[pos_pairs[:, 0]] - h[pos_pairs[:, 1]]).pow(2).sum(1).mean()
    else:  # No positive pairs, so the loss contribution is 0
        print("ALERT - NO POSITIVE PAIRS CREATED")
        pos_loss = torch.tensor(0.0, device=h.device, requires_grad=True)

    if neg_pairs.size(0) > 0:  # If there are negative pairs
        neg_loss = (margin - (h[neg_pairs[:, 0]] - h[neg_pairs[:, 1]]).pow(2).sum(1)).clamp(min=0).mean()
    else:  # No negative pairs, so the loss contribution is 0
        print("ALERT - NO NEGATIVE PAIRS CREATED")
        neg_loss = torch.tensor(0.0, device=h.device, requires_grad=True)

    # Total loss is the sum of positive and negative losses
    return pos_loss + neg_loss

"""
def contrastive_loss(h, pos_pairs, neg_pairs, margin):

    if pos_pairs.size(0) > 0:  # If there are positive pairs
        pos_diff = h[pos_pairs[:, 0]] - h[pos_pairs[:, 1]]
        pos_loss = pos_diff.pow(2).sum(1).mean()
    else:  # No positive pairs
        print("ALERT - NO POSITIVE PAIRS CREATED")
        pos_loss = torch.zeros(1, device=h.device)

    if neg_pairs.size(0) > 0:  # If there are negative pairs
        neg_diff = h[neg_pairs[:, 0]] - h[neg_pairs[:, 1]]
        neg_loss = (margin - neg_diff.pow(2).sum(1)).clamp(min=0).mean()
    else:  # No negative pairs
        print("ALERT - NO NEGATIVE PAIRS CREATED")
        neg_loss = torch.zeros(1, device=h.device)

    return pos_loss + neg_loss
""" 

@torch.no_grad()
def validate(data, val_loader, model, device, margin):
    model.eval()
    total_loss = 0
    for batch in val_loader:
        batch = batch.to(device)
        h = model(batch.x, batch.edge_index)
        
        # Generate positive and negative pairs
        pos_pairs, neg_pairs = generate_pairs(batch, device)
        loss = contrastive_loss(h, pos_pairs, neg_pairs, margin)
        total_loss += loss.item() * batch.num_nodes

    # Return average loss over the validation set
    return total_loss / data.num_nodes

def train(data, train_loader, model, optimizer, device, margin):#, coverage):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        h = model(batch.x, batch.edge_index)
        
        # Generate positive and negative pairs
        pos_pairs, neg_pairs = generate_pairs(batch, device)#, coverage_threshold=coverage)
        
        loss = contrastive_loss(h, pos_pairs, neg_pairs, margin)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_nodes
    # Return average loss over the entire training set
    return total_loss / data.num_nodes

def train_with_validation(data, train_loader, val_loader, model, optimizer, device, margin, epochs, early_stopping=None):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Training
        train_loss = train(data, train_loader, model, optimizer, device, margin)
        train_losses.append(train_loss)

        # Validation
        val_loss = validate(data, val_loader, model, device, margin)
        val_losses.append(val_loss)

        print(f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Early stopping based on validation loss
        if early_stopping:
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

        # Save the model with the best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

    return train_losses, val_losses


@torch.no_grad()
def generate_embeddings(data, model):
    model.eval()

    # Debug edge_index before passing it to the model
    print(f"Edge index type: {type(data.edge_index)}")
    if not isinstance(data.edge_index, torch.Tensor):
        raise ValueError("edge_index must be a torch.Tensor")
    
    if data.edge_index.dtype != torch.long:
        data.edge_index = data.edge_index.long()  # Ensure it's of the correct type
    
    # Perform embedding computation
    embed = model(data.x, data.edge_index).cpu()
    print(f"EMBED TYPE IS {type(embed)}")
    
    # Get node features and embeddings
    node_features = data.x.cpu().numpy().astype(np.float32)
    embeddings = embed.numpy().astype(np.float32)
    combined = np.concatenate((embeddings, node_features), axis=1)
    return combined

import csv
def save_cluster_to_contig_to_csv(cluster_to_contig, filepath):
    """Creates a new CSV file at the given filepath and writes the cluster_to_contig dictionary into it."""
    # Ensure the directory exists
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)  # Create the directory if it doesn't exist

    # Now, create and write to the CSV file
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Cluster_ID', 'Contig_IDs'])  # Write header row
        for cluster_id, contigs in cluster_to_contig.items():
            writer.writerow([cluster_id, ','.join(contigs)])  # Write each cluster and its contigs


def write_bins(path, dataset, embeds, minbin):
    clustered_contigs = set()
    multi_contig_clusters = 0
    short_contigs = set()
    #skipped_clusters = 0
    clusters = cluster(embeds, np.array(dataset.node_names), normalized=True, cuda=True)
    cluster_to_contig = {i: c for (i, (n, c)) in enumerate(clusters)}
    # Print the number of clusters and the number of nodes per cluster
    # Save cluster_to_contig to CSV using the auxiliary function
    with open(f"{path}/cluster_to_contig.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Cluster_ID', 'Contig_IDs'])  # Write header row
        for cluster_id, contigs in cluster_to_contig.items():
            writer.writerow([cluster_id, ','.join(contigs)])  # Write each cluster and its contigs
    
    n_clusters = len(cluster_to_contig)+1
    print(f"Embeddings Clusteres resulted in {n_clusters} clusters")
    for c in cluster_to_contig:
        cluster_size = sum([dataset.node_lengths[dataset.node_names.index(contig)] for contig in cluster_to_contig[c]])
        if cluster_size < minbin:
            # print("skipped small cluster", len(cluster_to_contig[c]), "contig")
            for contig in cluster_to_contig[c]:
                short_contigs.add(contig)
            #skipped_clusters += 1
            continue
        multi_contig_clusters += 1
        with open(f"{path}/{c}.fa", "w") as binfile:
            for contig in cluster_to_contig[c]:
                binfile.write(">" + contig + "\n")
                binfile.write(dataset.node_seqs[contig] + "\n")
                clustered_contigs.add(contig)
        # print("multi cluster", c, "size", cluster_size, "contigs", len(cluster_to_contig[c]))
    single_clusters = multi_contig_clusters
    left_over = set(dataset.node_names) - clustered_contigs - short_contigs
    for c in left_over:
        if c not in clustered_contigs and len(dataset.node_seqs[c]) > minbin:

            with open(f"{path}/{single_clusters}.fna", "w") as binfile:
                binfile.write(">" + c + "\n")
                binfile.write(dataset.node_seqs[c] + "\n")
                single_clusters += 1
            # print("contig", single_clusters, "size", len(dataset.contig_seqs[c]))
    return n_clusters, cluster_to_contig
    
def eval_cluster(dataset, best_cluster_to_contig):
    if len(dataset.ref_marker_sets) == 0:
        print("NO ref MARKERS")
        
    if len(dataset.contig_markers) == 0:
        print("NO CONTIG MARKERS")

    total_hq = 0
    total_mq = 0
    results = evaluate_contig_sets(dataset.ref_marker_sets, dataset.contig_markers, best_cluster_to_contig)
    hq_bins = set()
    for binid in results:
        if results[binid]["comp"] > 90 and results[binid]["cont"] < 5:
            contig_labels = [dataset.node_to_label.get(node, 0) for node in best_cluster_to_contig[binid]]
            labels_count = Counter(contig_labels)
            hq_bins.add(binid)
            total_hq += 1
        if results[binid]["comp"] > 50 and results[binid]["cont"] < 10:
            total_mq += 1
    print(f"#### Total HQ {total_hq} ####")
    print(f"#### Total MQ {total_mq} ####")    
    return total_hq, total_mq
    
    
def test_value(dataset, data, parameter, value, config, base_path, results_df, counter,
               comparison, first_time=False, current_iter_suffix="", ):#coverage,
    """
    Test a specific value for the given parameter and log the results.
    
    Parameters:
    - dataset: The dataset object.
    - data: The graph data object (torch geometric).
    - parameter: The parameter being tested (e.g., batch_size, hidden_channels).
    - value: The value of the parameter to test.
    - config: The base configuration dictionary.
    - base_path: Path where the results will be stored.
    - results_df: DataFrame where results will be logged.
    - counter: The current iteration counter.
    - first_time: Boolean indicating if this is the first run.
    - current_iter_suffix: Suffix to add to the folder name, used for edge-updated reruns.
    
    Returns:
    - data: The updated data object (if modified during the test).
    - results_df: The updated results DataFrame.
    """
    print(f"CURRENTLY TESTING {parameter}={value}")

    # Set up the configuration for the current run
    config = config.copy()
    config[parameter] = value

    # Ensure out_channels is not larger than hidden_channels
    if config['out_channels'] > config['hidden_channels']:
        print(f"Adjusting out_channels from {config['out_channels']} to {config['hidden_channels']} (matching hidden_channels)")
        config['out_channels'] = config['hidden_channels']
    
    # Create a unique folder name based on the parameter and value
    current_iter = f"cluster_output_{counter}{current_iter_suffix}" if not first_time else "Baseline"
    output_path = os.path.join(base_path, current_iter)
    os.makedirs(output_path, exist_ok=True)

    # Initialize data loader with the current batch size
    train_loader = NeighborLoader(
        data,
        num_neighbors=[config['num_neighbors']] * config['num_layers'],
        batch_size=config['batch_size'],
        shuffle=True
    )
    
    """
    train_loader = NeighborSampler(
        edge_index=data.edge_index,
        sizes=[config['num_neighbors']] * config['num_layers'],
        batch_size=config['batch_size'],
        shuffle=True,
        num_nodes=data.num_nodes
    )
    """
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device, 'x', 'edge_index')
    print(f"THIS IS THE NUMBER OF NODE FEATURES: {data.num_node_features}")
    
    # Initialize the model with the current hidden_channels, out_channels, and num_layers
    model = GraphSAGE(
        data.num_node_features,
        hidden_channels=config['hidden_channels'],
        out_channels=config['out_channels'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)

    # Initialize the optimizer with the current learning rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    # Training loop
    times = []
    epoch_losses = []
    if not first_time:
        for epoch in range(0, config['epochs']):
            start = time.time()
            loss = train(data, train_loader, model, optimizer, device, margin=config['margin'])#, coverage=coverage)
            epoch_losses.append(loss)
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
            times.append(time.time() - start)
        
        median_time = torch.tensor(times).median().item()
        print(f"Median time per epoch: {median_time:.4f}s")
        print(f"Final Loss: {loss}")
        
        # Plot and save loss curve
        plot_and_save_loss(epoch_losses, current_iter, parameter, os.path.join(base_path, "losses"))
    else:
        median_time = None
        epoch_losses = []

    # Generate embeddings and cluster them
    embeddings = generate_embeddings(data, model)
    print("Embeddings Generated")
    
    # Write Bins
    n_clusters, cluster_to_contig = write_bins(output_path, dataset, embeddings, 200000)
    print(f"Clusters Saved in {output_path}")

    # Evaluate Clusters
    hq_clusters, mq_clusters = eval_cluster(dataset, cluster_to_contig)
    #Scatter_plot Clusters
    plot_pca_clusters(dataset, data, embeddings, cluster_to_contig, os.path.join(base_path, "plots"), current_iter, value)
    #plot_pca_embeddings_simple(data, embeddings, os.path.join(base_path, "plots"), current_iter, value)
    # Log the results
    new_row = {
        'Folder Name': current_iter, 'Parameter': parameter, 'Value': value,
        'Median Time Per Epoch': median_time, 'Final Loss': epoch_losses[-1] if epoch_losses else None,
        'Batch Size': config['batch_size'], 'Hidden Channels': config['hidden_channels'], 
        'Out Channels': config['out_channels'], 'N. Layers': config['num_layers'],
        'Learning Rate': config['lr'], 'Margin': config['margin'],
        'N. Neighbors': config['num_neighbors'], 'N. Clusters': n_clusters, 
        'Epochs': config['epochs'], 'Weight Decay': config['weight_decay'],
        'Dropout': config['dropout'], 'N. Components': config['num_components']
        , 'HQ Eval': hq_clusters, 'MQ Eval': mq_clusters
    }
    
    if comparison != None:
        new_row.update({'Isolated Nodes': comparison['isolated'], 
                        'Added Edges': comparison['added'],
                        'Removed Edges': comparison['removed']})
        
    results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)

    return data, results_df    

def hyperparameter_tuning(dataset, data, parameter, values, base_config, base_path, 
                          res_filename='hyperparameter_tuning_results.xlsx', 
                          directory = 'E:/FCT/Tese/Code/checkm2/checkm2/tests/data', 
                          first=False, update_edges=False):#, coverage=coverage):
    """
    Run hyperparameter tuning for the specified parameter, optionally updating graph edges.

    Parameters:
    - dataset: The dataset object.
    - parameter (str): The parameter to tune (e.g., 'batch_size', 'hidden_channels').
    - values (list): The list of values to try for the specified parameter.
    - base_config (dict): The base configuration containing default values for all parameters.
    - base_path (str): The base directory where each iteration's results will be saved.
    - results_filename (str): The filename of the Excel file to store results.
    - update_edges (bool): If True, updates the graph edges and re-runs the model for comparison.
    """
    first_time = first
    results_filename = os.path.join(base_path, res_filename)

    # Initialize the results DataFrame if it doesn't exist
    if os.path.exists(results_filename):
        results_df = pd.read_excel(results_filename)
        last_iteration_number = get_last_iteration_number(results_filename)
    else:
        last_iteration_number = 0
        results_df = pd.DataFrame(columns=['Folder Name', 'Parameter', 'Value', 'Median Time Per Epoch', 
                                           'Final Loss', 'Batch Size', 'Hidden Channels', 'Out Channels', 
                                           'N. Layers', 'Dropout', 'Learning Rate', 'Weight Decay', 'Margin', 
                                           'N. Neighbors', 'Epochs', 'N. Components', 'N. Clusters', "HQ Eval", "MQ Eval"])
    # Start counter from the last iteration number
    counter = last_iteration_number + 1
    if(base_config['num_components'] == None):
        suffix = ""
    else: 
        suffix = "_PCA"
    # If the values list is empty, use the baseline value
    if not values:
        values = [base_config[parameter]]

    for value in values:
        # Test the value without updating edges
        data, results_df = test_value(dataset, data, parameter, value, config=base_config, 
                                      base_path=base_path, results_df=results_df, 
                                      counter=counter, comparison=None,# coverage = coverage,
                                      first_time=first_time, current_iter_suffix=suffix)
        first_time = False  # After the first iteration, set first_time to False

        # Update graph and re-run the model if update_edges is True
        if update_edges:
            print("Updating edges and re-running the model for comparison.")
            
            # Update edges of the graph
            data_updated = update_graph_edges(data, directory)

            # Re-run the model with updated edges
            data_updated, results_df = test_value(dataset, data_updated, parameter, value, 
                                                  config=base_config, base_path=base_path, 
                                                  results_df=results_df, counter=counter,
                                                  comparison= compare_graphs(data, data_updated), 
                                                  #coverage = coverage, 
                                                  current_iter_suffix=suffix+"_updated_edges")
            
            # Update edges of the graph and remove other connections
            data_updated_removed = update_graph_edges(data, directory,remove_existing_edges=True)

            # Re-run the model with updated edges
            data_updated_removed, results_df = test_value(dataset, data_updated_removed, parameter, value, 
                                                  config=base_config, base_path=base_path, 
                                                  results_df=results_df, counter=counter,
                                                  comparison= compare_graphs(data, data_updated_removed), 
                                                  #coverage = coverage, 
                                                  current_iter_suffix=suffix+"_updated_&_removed_edges")
            
        # Increment the counter for the next iteration
        counter += 1

    # Save results to an Excel file
    results_df.to_excel(results_filename, index=False)
    print(f"Results saved to {results_filename}")

def hyperparameter_tuning_with_validation(dataset, data, parameter, values, base_config, base_path, res_filename='hyperparameter_tuning_results.xlsx', first=False):
    """
    Run hyperparameter tuning with validation for the specified parameter.

    Parameters:
    - dataset: The dataset object.
    - train_loader: DataLoader for training set.
    - val_loader: DataLoader for validation set.
    - parameter (str): The parameter to tune (e.g., 'batch_size', 'hidden_channels').
    - values (list): The list of values to try for the specified parameter.
    - base_config (dict): The base configuration containing default values for all parameters.
    - base_path (str): The base directory where each iteration's results will be saved.
    - results_filename (str): The filename of the Excel file to store results.
    """
    first_time = first
    results_filename = os.path.join(base_path, res_filename)
    
    # Initialize the results DataFrame if it doesn't exist
    if os.path.exists(results_filename):
        results_df = pd.read_excel(results_filename)
        last_iteration_number = get_last_iteration_number(results_filename)
    else:
        last_iteration_number = 0
        results_df = pd.DataFrame(columns=['Folder Name','Parameter', 'Value', 'Median Time Per Epoch', 
                                           'Final Loss', 'Final Val Loss', 'Batch Size','Hidden Channels', 'Out Channels', 
                                           'N. Layers', 'Dropout', 'Learning Rate', 'Weight Decay', 'Margin', 
                                           'N. Neighbors','Epochs','N. Clusters', 'Complete', 'Contaminated'])
    # Start counter from the last iteration number
    counter = last_iteration_number + 1

    # If the values list is empty, use the baseline value
    if not values:
        values = [base_config[parameter]]
        
    for value in values:
        print(f"CURRENTLY TESTING {parameter}={value}")
        config = base_config.copy()
        config[parameter] = value
        if config[parameter] == base_config[parameter] and values != []:
            print(f"Current value for {parameter} same as base_config")
        else:
            # Ensure out_channels is not larger than hidden_channels
            
            if config['out_channels'] > config['hidden_channels']:
                print(f"Adjusting out_channels from {config['out_channels']} to {config['hidden_channels']} (matching hidden_channels)")
                config['out_channels'] = config['hidden_channels']
                
            # Create a unique folder name based on the parameter and value
            if not first_time:
                current_iter = f"cluster_output_{counter}"
            else:
                current_iter = "Baseline"
            output_path = os.path.join(base_path, current_iter)
            os.makedirs(output_path, exist_ok=True)
            # Create NeighborLoaders for training and validation sets
            
            # *** ADD VALIDATION SPLIT ***
            # Assuming 'data' is your PyTorch Geometric Data object
            train_mask, val_mask = create_train_val_split(data)

            # Assign masks to the data object
            data.train_mask = train_mask
            data.val_mask = val_mask
            
            train_loader = NeighborLoader(
                data,
                num_neighbors=[config['num_neighbors']] * config['num_layers'],
                batch_size=config['batch_size'],
                shuffle=True,
                input_nodes=data.train_mask
            )

            val_loader = NeighborLoader(
                data,
                num_neighbors=[config['num_neighbors']] * config['num_layers'],
                batch_size=config['batch_size'],
                shuffle=False,
                input_nodes=data.val_mask
            )
            # Set device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            data = data.to(device, 'x', 'edge_index')

            # Initialize the model with the current hidden_channels, out_channels, and num_layers
            model = GraphSAGE(
                data.num_node_features,
                hidden_channels=config['hidden_channels'],
                out_channels=config['out_channels'],
                num_layers=config['num_layers'],
                dropout=config['dropout']
            ).to(device)

            optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
            
            if not first_time:
                times = []
                train_losses = []
                val_losses = []
                for epoch in range(0, config['epochs']):
                    start = time.time()
                    
                    # Train
                    train_loss = train(data, train_loader, model, optimizer, device, margin=config['margin'])
                    train_losses.append(train_loss)
                    print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}')
                    
                    # Validate
                    val_loss = validate(data, val_loader, model, device, margin=config['margin'])
                    val_losses.append(val_loss)
                    print(f'Epoch: {epoch:03d}, Val Loss: {val_loss:.4f}')
                    
                    times.append(time.time() - start)
                
                median_time = torch.tensor(times).median().item()
                print(f"Median time per epoch: {median_time:.4f}s")

                # Plot and save both training and validation loss curves
                plot_and_save_loss_wval(train_losses, val_losses, current_iter, parameter, os.path.join(base_path,"losses"))
            else:
                median_time = None
                train_losses = []
                val_losses = []
                first_time = False
            
            # Generate embeddings and cluster them
            embeddings = generate_embeddings(data, model)
            print("Embeddings Generated")

            # Write Bins
            n_clusters = write_bins(output_path, dataset, embeddings, 200000)
            print(f"Clusters Saved in {output_path}")

            counter += 1

            new_row = {
                'Folder Name': current_iter, 'Parameter': parameter, 'Value': value,
                'Median Time Per Epoch': median_time, 'Final Loss': train_losses[-1] if train_losses else None,
                'Final Val Loss': val_losses[-1] if val_losses else None, 'Batch Size': config['batch_size'],
                'Hidden Channels': config['hidden_channels'], 'Out Channels': config['out_channels'], 'N. Layers': config['num_layers'],
                'Learning Rate': config['lr'], 'Margin': config['margin'], 'N. Neighbors': config['num_neighbors'],
                'N. Clusters': n_clusters, 'Epochs': config['epochs'], 'Weight Decay': config['weight_decay'], 'Dropout': config['dropout']
            }
            results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)

    # Save results to an Excel file
    results_df.to_excel(results_filename, index=False)
    print(f"Results saved to {results_filename}")


def hyperparameter_combination_tuning(dataset, data, main_parameter, main_values, other_parameters, base_config, base_path, saved_filename = "saved.xlsx"):
    """
    Run hyperparameter tuning for the main parameter against each other parameter one by one, avoiding redundant tests.

    Parameters:
    - dataset: The dataset object.
    - main_parameter (str): The main parameter to tune (e.g., 'lr').
    - main_values (list): The list of values to try for the main parameter.
    - other_parameters (dict): A dictionary where keys are other parameters and values are lists of possible values for those parameters.
    - base_config (dict): The base configuration containing default values for all parameters.
    - base_path (str): The base directory where each iteration's results will be saved.
    - results_filename (str): The filename of the Excel file to store results.
    """

    # Load existing results if they exist
    results_filename = os.path.join(base_path, saved_filename)
    if os.path.exists(results_filename):
        results_df = pd.read_excel(results_filename)
    else:
        results_df = pd.DataFrame(columns=['Main Parameter', 'Main Value', 'Other Parameter', 'Other Value', 'Tested'])

    for main_value in main_values:
        # Set the main parameter in the configuration
        config = base_config.copy()
        config[main_parameter] = main_value
        
        for other_param, other_values in other_parameters.items():
            for other_value in other_values:
                # Check if this combination has already been tested
                already_tested = (
                    (results_df['Main Parameter'] == main_parameter) &
                    (results_df['Main Value'] == main_value) &
                    (results_df['Other Parameter'] == other_param) &
                    (results_df['Other Value'] == other_value)
                ).any()

                if already_tested:
                    print(f"Skipping already tested configuration: {main_parameter}={main_value}, {other_param}={other_value}")
                    continue

                # Create a specific config for the current combination
                specific_config = config.copy()
                specific_config[other_param] = other_value

                # Ensure out_channels is not larger than hidden_channels
                if specific_config['out_channels'] > specific_config['hidden_channels']:
                    print(f"Adjusting out_channels from {specific_config['out_channels']} to {specific_config['hidden_channels']} (matching hidden_channels)")
                    specific_config['out_channels'] = specific_config['hidden_channels']
                
                # Display the current combination being tested
                print(f"Testing {main_parameter}={main_value} with {other_param}={other_value}")

                # Run the hyperparameter tuning for this specific combination
                hyperparameter_tuning(dataset, data, main_parameter, [main_value], specific_config, base_path)

                # Log this combination as tested
                new_row = {
                    'Main Parameter': main_parameter,
                    'Main Value': main_value,
                    'Other Parameter': other_param,
                    'Other Value': other_value,
                    'Tested': True
                }
                results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)

                # Save the results after each iteration to ensure progress is saved
                results_df.to_excel(results_filename, index=False)
"""
def update_graph_edges(data, base_directory, remove_existing_edges=False):
    Update graph edges by fully connecting nodes from high-quality clusters (HQ clusters) in the data object.
    High-quality clusters are defined by completeness > 90 and contamination < 5.

    Parameters:
    - data: A torch geometric data object containing the graph (with edge_index).
    - quality_report_file: Path to the quality_report.tsv file.
    - contig2bin_file: Path to the graphmb_best_contig2bin.tsv file.
    - remove_existing_edges: If True, remove all edges of the nodes in the HQ cluster before adding new connections.

    Returns:
    - data: The modified data object with updated edges for HQ clusters.
    

    # Step 1: Load quality report and save clusters with Completeness > 90 and Contamination < 5
    quality_report_file = os.path.join(base_directory, 'quality_report.tsv')
    quality_report = pd.read_csv(quality_report_file, sep='\t')
    hq_clusters = quality_report[
        (quality_report['Completeness'] > 90) & 
        (quality_report['Contamination'] < 5)
    ]['Name'].tolist()

    # Step 2: Load graphmb_best_contig2bin file
    contig2bin_file = os.path.join(base_directory, 'graphmb_best_contig2bin.tsv')
    contig2bin = pd.read_csv(contig2bin_file, sep='\t', skiprows=2, names=['SEQUENCEID', 'BINID'])

    # Step 3: Iterate over high-quality clusters
    for cluster_id in hq_clusters:
        # Find all nodes that belong to the current high-quality cluster
        cluster_nodes = contig2bin[contig2bin['BINID'] == cluster_id]['SEQUENCEID'].tolist()
        cluster_node_indices = [int(n) for n in cluster_nodes]

        if remove_existing_edges:
            # Step 4: If the flag is True, remove all existing edges of nodes in this cluster
            mask = ~(torch.isin(data.edge_index[0], cluster_node_indices) | torch.isin(data.edge_index[1], cluster_node_indices))
            data.edge_index = data.edge_index[:, mask]

        # Step 5: Add missing connections between all nodes in this HQ cluster
        for i in range(len(cluster_node_indices)):
            for j in range(i + 1, len(cluster_node_indices)):
                node_i = cluster_node_indices[i]
                node_j = cluster_node_indices[j]

                # Check if there is already an edge between node_i and node_j
                existing_edges = ((data.edge_index[0] == node_i) & (data.edge_index[1] == node_j)) | \
                                 ((data.edge_index[0] == node_j) & (data.edge_index[1] == node_i))
                
                if not existing_edges.any():
                    # Add the edge in both directions for an undirected graph
                    new_edges = torch.tensor([[node_i, node_j], [node_j, node_i]], dtype=torch.long)
                    data.edge_index = torch.cat([data.edge_index, new_edges], dim=1)

    return data

import torch
"""

def update_graph_edges(data, base_directory, remove_existing_edges=False):
    """
    Update graph edges by fully connecting nodes from high-quality clusters (HQ clusters) in the data object.
    High-quality clusters are defined by completeness > 90 and contamination < 5.
    """
    # Step 1: Load quality report and save clusters with Completeness > 90 and Contamination < 5
    quality_report_file = os.path.join(base_directory, 'quality_report.tsv')

    quality_report = pd.read_csv(quality_report_file, sep='\t')

    # Step 2: Load graphmb_best_contig2bin file, skipping metadata rows and fixing headers
    contig2bin_file = os.path.join(base_directory, 'graphmb_best_contig2bin.tsv')
    contig2bin = pd.read_csv(contig2bin_file, sep='\t', skiprows=3, names=['SEQUENCEID', 'BINID'])

    # Step 3: For each high-quality cluster, get the nodes belonging to that cluster
    hq_clusters = quality_report[(quality_report['Completeness'] > 90) & (quality_report['Contamination'] < 5)]
    cluster_ids = hq_clusters['Name'].tolist()

    for cluster_id in cluster_ids:
        cluster_nodes = contig2bin[contig2bin['BINID'] == cluster_id]['SEQUENCEID'].tolist()
        cluster_node_indices = torch.tensor([int(node.split('_')[-1]) for node in cluster_nodes], dtype=torch.long)

        # Ensure cluster_node_indices is on the same device as data.edge_index
        cluster_node_indices = cluster_node_indices.to(data.edge_index.device)

        if remove_existing_edges:
            # Step 4: Create a mask to remove existing edges
            mask = ~(torch.isin(data.edge_index[0], cluster_node_indices) | 
                     torch.isin(data.edge_index[1], cluster_node_indices))

            data.edge_index = data.edge_index[:, mask]

    return data
