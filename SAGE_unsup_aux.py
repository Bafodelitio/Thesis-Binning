import time
import random
import torch
import os
from collections import Counter
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GraphSAGE
from torch_geometric.utils import to_undirected, negative_sampling
from sklearn.model_selection import train_test_split
from vamb_clustering import cluster
import numpy as np
from collections import defaultdict
import pandas as pd
from util import plot_and_save_loss, get_last_iteration_number, compare_graphs, evaluate_contig_sets, is_undirected, get_undirected_edge_set, plot_pca_hq_clusters

def encode(node_names):
    return [torch.tensor(name.encode('utf-8') for name in node_names)]

def decode(node_ids):
    return [bytes(tensor.tolist()).decode('utf-8') for tensor in node_ids]  

def create_pyg_graph_from_assembly(dataset):
    # Prepare node features separately
    
    node_length = torch.tensor(dataset.node_lengths, dtype=torch.float).unsqueeze(1)
    node_depths = torch.tensor(dataset.node_depths, dtype=torch.float)#.squeeze(ONLY FOR AALE)  # New feature
    node_kmers = torch.tensor(dataset.node_kmers, dtype=torch.float)
    # Prepare node ids as a tensor (assuming node_names are unique identifiers)
    node_ids = torch.tensor([dataset.node_names.index(name) for name in dataset.node_names], dtype=torch.long)
    
    # Prepare edge index and edge weights
    edges = np.array([dataset.edges_src, dataset.edges_dst])
    edge_index = to_undirected(torch.from_numpy(edges).long())    
    # Create the PyTorch Geometric data object
    data = Data(edge_index=edge_index)

    # Assign the node features to the data object
    data.length = node_length
    data.depths = node_depths
    data.kmers = node_kmers
    data.node_names = node_ids #encode(dataset.node_names)
    #data.num_nodes = node_ids.size(0)
    
    if 'x' not in data:
        #data.x = torch.tensor(data.kmers, dtype=torch.float)
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
    
def gen_save_clusters(data, model, dataset, output_path):
    # Generate embeddings and cluster them
    embeddings = generate_embeddings(data, model)
    print("Embeddings Generated")
    
    # Write Bins
    clusters = cluster(embeddings, np.array(dataset.node_names), normalized=True, cuda=True)
    cluster_to_contig = {i: c for (i, (n, c)) in enumerate(clusters)}
    n_clusters = len(cluster_to_contig)+1
    print(f"Clustered into {n_clusters} bins")
    #_clusters, cluster_to_contig = write_bins(output_path, dataset, embeddings, 200000)
    #print(f"Clusters Saved in {output_path}")

    # Evaluate Clusters
    hq_clusters, mq_clusters = eval_cluster(dataset, cluster_to_contig)
    
    return n_clusters, hq_clusters, mq_clusters
    
def test_value(dataset, data, parameter, value, config, base_path, results_df, counter,
               comparison, first_time=False, current_iter_suffix="", epoch_cluster=0):
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
    - epoch_cluster: Number of epochs after which to generate and save clusters. If 0, clustering is done only at the end.
    
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

    """ 
    num_neighbors = [config['num_neighbors']]  # Start with the first hop
    for i in range(1, config['num_layers']):
        num_neighbors.append(int(num_neighbors[-1] * 1.5))  # Multiply by 1.5 for each subsequent hop
    """
    
    # Initialize data loader with the current batch size
    #num_neighbors = [config['num_neighbors'] * (2 ** i) for i in range(config['num_layers'])]
    
    train_loader = NeighborLoader(
        data,
        num_neighbors=[config['num_neighbors']] * config['num_layers'],
        batch_size=config['batch_size'],
        shuffle=True
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

            # Generate and save clusters every epoch_cluster epochs
            if epoch_cluster > 0 and (epoch + 1) % epoch_cluster == 0:
                cluster_output_path = os.path.join(output_path, f"epoch_{epoch + 1}")
                os.makedirs(cluster_output_path, exist_ok=True)
                n_clusters, hq_clusters, mq_clusters = gen_save_clusters(data, model, dataset, cluster_output_path)
                
                # Log the intermediate results
                new_row = {
                    'Folder Name': f"{current_iter}_epoch_{epoch + 1}", 'Parameter': parameter, 'Value': value,
                    'Median Time Per Epoch': None, 'Final Loss': loss,
                    'Batch Size': config['batch_size'], 'Hidden Channels': config['hidden_channels'], 
                    'Out Channels': config['out_channels'], 'N. Layers': config['num_layers'],
                    'Learning Rate': config['lr'], 'Margin': config['margin'],
                    'N. Neighbors': config['num_neighbors'], 'N. Clusters': n_clusters, 
                    'Epochs': epoch + 1, 'Weight Decay': config['weight_decay'],
                    'Dropout': config['dropout'], 'N. Components': config['num_components']
                    , 'HQ Eval': hq_clusters, 'MQ Eval': mq_clusters
                }
                
                if comparison != None:
                    new_row.update({'Isolated Nodes': comparison['isolated'], 
                                    'Added Edges': comparison['added'],
                                    'Removed Edges': comparison['removed']})
                
                results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
        
        median_time = torch.tensor(times).median().item()
        print(f"Median time per epoch: {median_time:.4f}s")
        print(f"Final Loss: {loss}")
        
        # Plot and save loss curve
        plot_and_save_loss(epoch_losses, current_iter, parameter, os.path.join(base_path, "losses"))
    else:
        median_time = None
        epoch_losses = []

    # Generate and save clusters at the end of training
    n_clusters, hq_clusters, mq_clusters = gen_save_clusters(data, model, dataset, output_path)
    
    # Log the final results
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
                          directory = 'E:/FCT/Tese/Code/GNN', 
                          first=False, update_edges=False, epoch_cluster=0):
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
                                      counter=counter, comparison=None,
                                      first_time=first_time, current_iter_suffix=suffix, epoch_cluster=epoch_cluster)
        first_time = False  # After the first iteration, set first_time to False
        
        # Update graph and re-run the model if update_edges is True
        if update_edges:
            node_depths = dataset.node_depths
            print("Updating edges and re-running the model for comparison.")
            
            # Update edges of the graph
            data_updated, changes = load_or_update_graph(data, dataset, 
                                                         directory, 
                                                         node_depths, 
                                                         add_hq_edges=True, 
                                                         remove_existing_edges=False, 
                                                         depth_pruning=False,
                                                         depth_linking=False)
            
            
            # Re-run the model with updated edges
            _, results_df = test_value(dataset, data_updated, parameter, value, 
                                                  config=base_config, base_path=base_path, 
                                                  results_df=results_df, counter=counter,
                                                  comparison=changes, 
                                                  current_iter_suffix=suffix+"_updated_edges",
                                                  epoch_cluster=epoch_cluster)
            
            # Update edges of the graph and remove other connections
            data_updated_removed, changes = load_or_update_graph(data, dataset,
                                                                directory, 
                                                                node_depths,  
                                                                add_hq_edges=False, 
                                                                remove_existing_edges=True, 
                                                                depth_pruning=False,
                                                                depth_linking=False, 
                                                                depth_diff_threshold=2.0)

            # Re-run the model with updated edges
            _, results_df = test_value(dataset, data_updated_removed, parameter, value, 
                                                  config=base_config, base_path=base_path, 
                                                  results_df=results_df, counter=counter,
                                                  comparison=changes, 
                                                  current_iter_suffix=suffix+"_removed_edges",
                                                  epoch_cluster=epoch_cluster)
            
            # Update edges of the graph and remove other connections
            data_updated_removed, changes = load_or_update_graph(data, dataset,
                                                                directory, 
                                                                node_depths,  
                                                                add_hq_edges=True, 
                                                                remove_existing_edges=True, 
                                                                depth_pruning=False,
                                                                depth_linking=False)

            # Re-run the model with updated edges
            _, results_df = test_value(dataset, data_updated_removed, parameter, value, 
                                                  config=base_config, base_path=base_path, 
                                                  results_df=results_df, counter=counter,
                                                  comparison=changes, 
                                                  current_iter_suffix=suffix+"_updated_&_removed_edges",
                                                  epoch_cluster=epoch_cluster)
            """
            # Update edges of the graph and add depth_pruning
            data_updated_pruned, changes = load_or_update_graph(data, dataset,
                                                                directory, 
                                                                node_depths,  
                                                                add_hq_edges=False, 
                                                                remove_existing_edges=False, 
                                                                depth_pruning=True,
                                                                depth_linking=False)

            # Re-run the model with updated depths
            _, results_df = test_value(dataset, data_updated_pruned, parameter, value, 
                                                  config=base_config, base_path=base_path, 
                                                  results_df=results_df, counter=counter,
                                                  comparison=changes, 
                                                  current_iter_suffix=suffix+"_pruned",
                                                  epoch_cluster=epoch_cluster) 
            
            # Update edges of the graph and add depth_pruning
            data_updated_pruned, changes = load_or_update_graph(data, dataset,
                                                                directory, 
                                                                node_depths,  
                                                                add_hq_edges=False, 
                                                                remove_existing_edges=False, 
                                                                depth_pruning=False,
                                                                depth_linking=True,
                                                                similarity_threshold= 0.9)

            # Re-run the model with updated depths
            _, results_df = test_value(dataset, data_updated_pruned, parameter, value, 
                                                  config=base_config, base_path=base_path, 
                                                  results_df=results_df, counter=counter,
                                                  comparison=changes, 
                                                  current_iter_suffix=suffix+"_depth_linked",
                                                  epoch_cluster=epoch_cluster)
            
            # Update edges of the graph and add depth_pruning
            data_updated_pruned, changes = load_or_update_graph(data, dataset,
                                                                directory, 
                                                                node_depths,  
                                                                add_hq_edges=False, 
                                                                remove_existing_edges=False, 
                                                                depth_pruning=True,
                                                                depth_linking=True,
                                                                similarity_threshold= 0.9)

            # Re-run the model with updated depths
            _, results_df = test_value(dataset, data_updated_pruned, parameter, value, 
                                                  config=base_config, base_path=base_path, 
                                                  results_df=results_df, counter=counter,
                                                  comparison=changes, 
                                                  current_iter_suffix=suffix+"_prune_linked",
                                                  epoch_cluster=epoch_cluster)
            """
        # Increment the counter for the next iteration
        counter += 1

        # Save results to an Excel file
        results_df.to_excel(results_filename, index=False)
        print(f"Results saved to {results_filename}")


def validate_cluster_nodes(cluster_ids, contig2bin, dataset):
    """
    Validates and maps cluster nodes, checking for:
    - Nodes existing in dataset.node_names
    - Nodes belonging to multiple clusters
    - Empty clusters
    
    Args:
        cluster_ids: List of cluster IDs to process
        contig2bin: DataFrame mapping contigs to bins
        dataset: Dataset object containing node_names
        
    Returns:
        dict: {cluster_id: list_of_node_indices} mapping
    """
    from collections import defaultdict
    
    # Initialize data structures
    cluster_to_nodes = defaultdict(list)
    node_to_clusters = defaultdict(list)
    missing_nodes = []
    problematic_clusters = set()

    # Validate each cluster
    for cluster_id in cluster_ids:
        # Get all contigs for this cluster
        cluster_contigs = contig2bin[contig2bin['BINID'] == cluster_id]['SEQUENCEID']
        
        # Skip empty clusters
        if len(cluster_contigs) == 0:
            print(f"Warning: Cluster {cluster_id} has no contigs assigned")
            problematic_clusters.add(cluster_id)
            continue
            
        # Map contigs to node indices
        cluster_nodes = []
        for contig in cluster_contigs:
            if contig not in dataset.node_names:
                missing_nodes.append(contig)
                continue
                
            node_idx = dataset.node_names.index(contig)
            cluster_nodes.append(node_idx)
            node_to_clusters[node_idx].append(cluster_id)
            
        # Store validated nodes
        if cluster_nodes:
            cluster_to_nodes[cluster_id] = cluster_nodes
        else:
            print(f"Warning: Cluster {cluster_id} has no valid nodes")
            problematic_clusters.add(cluster_id)

    # Report missing nodes (first 5 examples)
    if missing_nodes:
        print(f"\nError: {len(missing_nodes)} nodes missing from dataset.node_names")
        print("Sample missing nodes:", missing_nodes[:5])

    # Check for nodes in multiple clusters
    duplicate_nodes = {node: clusters for node, clusters in node_to_clusters.items() 
                      if len(clusters) > 1}
    if duplicate_nodes:
        print(f"\nError: {len(duplicate_nodes)} nodes belong to multiple clusters")
        for node, clusters in list(duplicate_nodes.items())[:5]:
            print(f"  Node {node} appears in clusters: {clusters}")

    # Final validation
    valid_clusters = set(cluster_to_nodes.keys())
    invalid_clusters = problematic_clusters - valid_clusters
    
    print(f"\nValidation results:")
    print(f"- Valid clusters: {len(valid_clusters)}/{len(cluster_ids)}")
    print(f"- Problematic clusters: {len(problematic_clusters)}")
    print(f"- Nodes assigned: {sum(len(nodes) for nodes in cluster_to_nodes.values())}")
    
    if invalid_clusters:
        print(f"\nWarning: {len(invalid_clusters)} clusters have no valid nodes:")
        print(list(invalid_clusters)[:5], "...")

    return cluster_to_nodes


def add_hq_cluster_edges(data, cluster_to_nodes):
    data = data.clone()
    changes = {'added': 0}
    
    # Convert to set for faster lookups (sorted tuples)
    existing_pairs = {tuple(sorted((i.item(), j.item()))) 
                     for i, j in data.edge_index.t()}
    
    new_edges = []
    for cluster_id, node_indices in cluster_to_nodes.items():
        if len(node_indices) < 2:
            continue
            
        # Generate all possible pairs (both directions)
        for i, u in enumerate(node_indices):
            for v in node_indices[i+1:]:  # Avoid self-loops
                pair = tuple(sorted((u, v)))
                if pair not in existing_pairs:
                    new_edges.extend([[u, v], [v, u]])  # Add both directions
                    existing_pairs.add(pair)
                    changes['added'] += 2
    
    if new_edges:
        new_edges = torch.tensor(new_edges).t().to(data.edge_index.device)
        data.edge_index = torch.cat([data.edge_index, new_edges], dim=1)
        
        # Verify
        expected = sum(len(n)*(len(n)-1) for n in cluster_to_nodes.values())
        print(f"Added {changes['added']}/{expected} undirected pairs")
    
    return data, changes


def remove_edges(data, cluster_to_nodes):
    data = data.clone()
    changes = {'removed': 0}
    
    # Get all HQ nodes as a tensor
    hq_nodes = torch.tensor(
        sorted({n for nodes in cluster_to_nodes.values() for n in nodes}),
        device=data.edge_index.device
    )
    
    src, dst = data.edge_index
    to_remove = torch.zeros_like(src, dtype=torch.bool)
    
    # Vectorized check for HQ↔non-HQ edges
    is_hq_src = torch.isin(src, hq_nodes)
    is_hq_dst = torch.isin(dst, hq_nodes)
    to_remove = (is_hq_src != is_hq_dst)  # XOR operation
    
    # Apply removal
    data.edge_index = data.edge_index[:, ~to_remove]
    changes['removed'] = to_remove.sum().item()
    
    # Final verification
    src, dst = data.edge_index
    leaks = torch.isin(src, hq_nodes) != torch.isin(dst, hq_nodes)
    print(f"Removed {changes['removed']} edges | "
          f"Remaining leaks: {leaks.sum().item()}")
    
    return data, changes

"""
def depth_based_edge_pruning(data, dataset, node_depths, cluster_ids, contig2bin, depth_diff_threshold=1.305):
    
    #Prunes edges based on depth differences between nodes, but preserves edges between nodes in the same HQ cluster.
    #Args:
    #    data: PyG Data object with edge_index.
    #    node_depths: Tensor of node depths (shape: [num_nodes]).
    #    cluster_ids: List of high-quality cluster IDs.
    #    contig2bin: DataFrame mapping contigs to bins.
    #    dataset: Dataset object containing node_names.
    #    depth_diff_threshold: Maximum allowed depth difference for edge pruning.
    #Returns:
    #    data: Updated copy of the PyG Data object with pruned edges.
    #    changes: Dictionary of changes made to the graph.
    changes = {'added': 0, 'removed': 0, 'isolated': 0}
    
    # Ensure node_depths is on the same device as data.edge_index    
    node_depths = node_depths.to(data.edge_index.device)
    
    # Normalize node depths
    node_depths = (node_depths - node_depths.mean()) / node_depths.std()

    # Get all HQ cluster node indices
    hq_node_indices = []
    for cluster_id in cluster_ids:
        cluster_nodes = contig2bin[contig2bin['BINID'] == cluster_id]['SEQUENCEID'].tolist()
        
        # Convert node names to node indices using dataset.node_names
        cluster_node_indices = []
        for node_name in cluster_nodes:
            if node_name in dataset.node_names:
                cluster_node_indices.append(dataset.node_names.index(node_name))
            else:
                print(f"Warning: Node {node_name} not found in dataset.node_names. Skipping.")
        
        # Skip if no valid nodes are found in the cluster
        if not cluster_node_indices:
            print(f"Warning: Cluster {cluster_id} has no valid nodes. Skipping.")
            continue

        # Convert to tensor and ensure it's on the correct device
        cluster_node_indices = torch.tensor(cluster_node_indices, dtype=torch.long, device=data.edge_index.device)
        hq_node_indices.append(cluster_node_indices)

    # Concatenate all HQ node indices
    if hq_node_indices:
        hq_node_indices = torch.cat(hq_node_indices)
    else:
        print("Warning: No valid HQ nodes found. No edges will be preserved.")
        hq_node_indices = torch.tensor([], dtype=torch.long, device=data.edge_index.device)

    # Apply depth-based edge pruning
    src, dst = data.edge_index
    depth_diff = torch.abs(node_depths[src] - node_depths[dst])

    # Create a mask to preserve edges between nodes in the same HQ cluster
    hq_mask = torch.isin(src, hq_node_indices) & torch.isin(dst, hq_node_indices)

    # Create a mask to prune edges based on depth difference, but preserve HQ cluster edges
    prune_mask = (depth_diff <= depth_diff_threshold) | hq_mask

    # Apply the mask
    removed_edges = data.edge_index.size(1) - prune_mask.sum().item()
    data.edge_index = data.edge_index[:, prune_mask]
    print(f"Removed {removed_edges} edges based on depth pruning (threshold: {depth_diff_threshold}).")

    changes['removed'] = removed_edges
    return data, changes


from sklearn.neighbors import NearestNeighbors
def add_edges_based_on_depth_similarity(data, node_depths, similarity_threshold=0.98):
    
    #Adds edges between nodes with similar depth values (optimized for large graphs).
    #Args:
    #    data: PyG Data object with edge_index.
    #    node_depths: Tensor of node depths (shape: [num_nodes] for 1D or [num_nodes, 4] for 4D).
    #    similarity_threshold: Minimum similarity (as a fraction of max depth difference) to add an edge.
    #Returns:
    #    data: Updated copy of the PyG Data object with added edges.
    #    changes: Dictionary of changes made to the graph.
    
    changes = {'added': 0, 'removed': 0, 'isolated': 0}
    print(f"SELECTED SIMILARITY THRESHOLD: {similarity_threshold}")
    # Ensure node_depths is on the same device as data.edge_index    
    node_depths = node_depths.to(data.edge_index.device)
    
    # Normalize depth values
    if node_depths.dim() == 1:  # 1D depth values
        node_depths = (node_depths - node_depths.mean()) / node_depths.std()
        node_depths_np = node_depths.cpu().numpy().reshape(-1, 1)  # Reshape to 2D for sklearn
    elif node_depths.dim() == 2:  # 2D depth values (e.g., [num_nodes, 4])
        node_depths = (node_depths - node_depths.mean(dim=0)) / node_depths.std(dim=0)
        node_depths_np = node_depths.cpu().numpy()
    else:
        raise ValueError("node_depths must be 1D or 2D.")

    # Use Approximate Nearest Neighbor (ANN) search to find similar nodes
    n_neighbors = min(100, node_depths_np.shape[0])  # Number of neighbors to consider
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', metric='euclidean').fit(node_depths_np)
    distances, indices = nbrs.kneighbors(node_depths_np)

    # Convert distances to similarities
    max_distance = np.max(distances)
    similarities = 1 - (distances / max_distance)

    # Add edges based on similarity threshold
    new_edges = []
    for i in range(indices.shape[0]):
        for j, sim in zip(indices[i], similarities[i]):
            if i < j and sim >= similarity_threshold:  # Avoid duplicate edges
                new_edges.append((i, j))
                new_edges.append((j, i))  # Add reverse edge for undirected graph

    # Convert new edges to tensor
    if new_edges:
        new_edges = torch.tensor(new_edges, dtype=torch.long, device=data.edge_index.device).t()
        # Add new edges to the graph
        data.edge_index = torch.cat([data.edge_index, new_edges], dim=1)
        print(f"Added {new_edges.size(1)} edges based on depth similarity (threshold: {similarity_threshold}).")
        changes['added'] = new_edges.size(1)
    else:
        print("No edges added based on depth similarity.")

    return data, changes
"""

import faiss

def add_edges_based_on_depth_similarity(data, node_depths, similarity_threshold=0.98, n_neighbors=100):
    """
    Adds edges between nodes with similar depth values (optimized for large graphs using FAISS).
    Args:
        data: PyG Data object with edge_index.
        node_depths: Tensor of node depths (shape: [num_nodes] for 1D or [num_nodes, 4] for 4D).
        similarity_threshold: Minimum similarity (as a fraction of max depth difference) to add an edge.
        n_neighbors: Number of nearest neighbors to consider for each node.
    Returns:
        data: Updated copy of the PyG Data object with added edges.
        changes: Dictionary of changes made to the graph.
    """
    changes = {'added': 0, 'removed': 0, 'isolated': 0}
    print(f"SELECTED SIMILARITY THRESHOLD: {similarity_threshold}")
    
    # Ensure node_depths is on the same device as data.edge_index    
    node_depths = node_depths.to(data.edge_index.device)
    
    # Normalize depth values
    if node_depths.dim() == 1:  # 1D depth values
        node_depths = (node_depths - node_depths.mean()) / node_depths.std()
        node_depths_np = node_depths.cpu().numpy().reshape(-1, 1)  # Reshape to 2D for FAISS
    elif node_depths.dim() == 2:  # 2D depth values (e.g., [num_nodes, 4])
        node_depths = (node_depths - node_depths.mean(dim=0)) / node_depths.std(dim=0)
        node_depths_np = node_depths.cpu().numpy()
    else:
        raise ValueError("node_depths must be 1D or 2D.")

    # Use FAISS for efficient nearest neighbor search
    dimension = node_depths_np.shape[1]  # Dimensionality of depth values
    index = faiss.IndexFlatL2(dimension)  # L2 distance (Euclidean)
    index.add(node_depths_np)  # Add depth values to the index

    # Search for nearest neighbors
    k = min(n_neighbors, node_depths_np.shape[0])  # Number of neighbors to consider
    distances, indices = index.search(node_depths_np, k)

    # Convert distances to similarities
    max_distance = np.max(distances)
    similarities = 1 - (distances / max_distance)

    # Add edges based on similarity threshold
    new_edges = []
    for i in range(indices.shape[0]):
        for j, sim in zip(indices[i], similarities[i]):
            if i < j and sim >= similarity_threshold:  # Avoid duplicate edges
                new_edges.append((i, j))
                new_edges.append((j, i))  # Add reverse edge for undirected graph

    # Convert new edges to tensor
    if new_edges:
        new_edges = torch.tensor(new_edges, dtype=torch.long, device=data.edge_index.device).t()
        # Add new edges to the graph
        data.edge_index = torch.cat([data.edge_index, new_edges], dim=1)
        size= new_edges.size(1)
        print(f"Added {size} edges based on depth similarity (threshold: {similarity_threshold}).")
        changes['added'] = size
    else:
        print("No edges added based on depth similarity.")

    print(f"Total edges added: {changes['added']}")  # Print total edges added
    return data, changes


def update_graph_edges(data, dataset, 
                       base_directory, node_depths, 
                       add_hq_edges=False, 
                       remove_existing_edges=False, 
                       depth_pruning=False, 
                       depth_linking=False, 
                       depth_diff_threshold=2.0, 
                       similarity_threshold=0.98):
    """
    Updates the graph edges based on high-quality clusters and node depths.
    Args:
        data: PyG Data object with edge_index.
        base_directory: Directory containing quality_report.tsv and graphmb_best_contig2bin.tsv.
        node_depths: Tensor of node depths (shape: [num_nodes]).
        remove_existing_edges: If True, removes edges connected to high-quality clusters.
        depth_pruning: If True, applies depth-based edge pruning.
        weighted_edges: If True, applies weighted edges based on depth similarity.
        depth_diff_threshold: Maximum allowed depth difference for edge pruning.
        similarity_threshold: Minimum depth similarity for edge weighting.
    Returns:
        data_copy: Updated copy of the PyG Data object with modified edges and edge weights.
    """
    # Create a deep copy of the data object to avoid modifying the original
    data_copy = data.clone()
    changes = {'added': 0, 'removed': 0, 'isolated': 0}
    
    # Load quality report and save clusters with Completeness > 90 and Contamination < 5
    quality_report_file = os.path.join(base_directory, 'quality_report.tsv')
    quality_report = pd.read_csv(quality_report_file, sep='\t')

    # Load graphmb_best_contig2bin file, skipping metadata rows and fixing headers
    contig2bin_file = os.path.join(base_directory, 'graphmb_best_contig2bin.tsv')
    contig2bin = pd.read_csv(contig2bin_file, sep='\t', skiprows=3, names=['SEQUENCEID', 'BINID'])

    # Identify high-quality clusters
    hq_clusters = quality_report[(quality_report['Completeness'] > 90) & (quality_report['Contamination'] < 5)]
    cluster_ids = hq_clusters['Name'].tolist()
    print(f"Found {len(cluster_ids)} high-quality clusters.")
    
    # After loading contig2bin and dataset, but before graph modifications
    missing_nodes = [name for name in contig2bin['SEQUENCEID'] if name not in dataset.node_names]
    print(f"Missing nodes in dataset.node_names: {len(missing_nodes)}")
    if missing_nodes:
        print("First 5 missing nodes:", missing_nodes[:5])
    
    # Step 1: Add edges to HQ clusters
    cluster_to_nodes = validate_cluster_nodes(cluster_ids, contig2bin, dataset)
    if add_hq_edges:
        data_copy, edge_changes = add_hq_cluster_edges(data_copy, cluster_to_nodes)
        changes['added'] += edge_changes['added']
    # Step 2: Remove existing edges connected to HQ clusters (if enabled)
    if remove_existing_edges:
        data_copy, edge_changes = remove_edges(data_copy, cluster_to_nodes)
        changes['removed'] += edge_changes['removed']
    # Step 3: Apply depth-based edge pruning (if enabled)
    # if depth_pruning:
    #    data_copy, edge_changes = depth_based_edge_pruning(data_copy, dataset, node_depths, cluster_ids, contig2bin, depth_diff_threshold)
    #    changes['removed'] += edge_changes['removed']
    # Step 4: Apply weighted edges based on depth similarity (if enabled)
    # if depth_linking:
    #    data_copy, edge_changes = add_edges_based_on_depth_similarity(data_copy, node_depths, similarity_threshold)
        
    # Calculate node degrees using torch.bincount
    node_degrees = torch.bincount(data_copy.edge_index[0], minlength=data_copy.num_nodes)
    # If the graph is undirected, add degrees from the reverse edges
    if is_undirected(data_copy.edge_index):
        node_degrees += torch.bincount(data_copy.edge_index[1], minlength=data_copy.num_nodes)
    # Calculate the number of isolated nodes
    isolated = (node_degrees == 0).sum().item()
    changes['isolated'] = isolated
    print(f"ISOLATED NODES FOUND: {isolated}")
    return data_copy, changes

def load_or_update_graph(data, dataset, base_directory, node_depths, add_hq_edges=False, remove_existing_edges=False, depth_pruning=False, depth_linking=False, depth_diff_threshold=2.0, similarity_threshold=0.98):
    """
    Wrapper function to load or update the graph based on methodologies.
    Args:
        data: PyG Data object with edge_index.
        base_directory: Directory containing quality_report.tsv and graphmb_best_contig2bin.tsv.
        node_depths: Tensor or array of node depths.
        dataset_name: Name of the dataset ("strong100" or "aale").
        add_hq_edges: If True, adds fully connected edges within HQ clusters.
        remove_existing_edges: If True, removes edges connected to high-quality clusters.
        depth_pruning: If True, applies depth-based edge pruning.
        weighted_edges: If True, applies weighted edges based on depth similarity.
        depth_diff_threshold: Maximum allowed depth difference for edge pruning.
        similarity_threshold: Minimum depth similarity for edge weighting.
    Returns:
        data: Updated or loaded PyG Data object.
    """
    changes = {'added': 0, 'removed': 0, 'isolated': 0}
    
    # Step 1: Convert node_depths to a tensor and ensure it has the correct shape
    if not isinstance(node_depths, torch.Tensor):
        node_depths = torch.tensor(node_depths, dtype=torch.float)

    # Debugging: Print node_depths shape
    print(f"Node depths shape before reduction: {node_depths.shape}")

    # Reduce node_depths to a single value per node if it has more than one dimension
    if node_depths.dim() > 1:
        node_depths = node_depths.mean(dim=1)  # Take the mean of the depth values

    # Debugging: Print node_depths shape after reduction
    print(f"Node depths shape after reduction: {node_depths.shape}")
    
    # Ensure node_depths is on the same device as data.edge_index
    node_depths = node_depths.to(data.edge_index.device)
    
    # Step 2: Determine the cache subfolder
    if dataset.name not in ["strong100", "aale"]:
        raise ValueError("dataset_name must be either 'strong100' or 'aale'.")
    cache_dir = os.path.join(base_directory, f"{dataset.name}_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    depth_threshold = 0
    if dataset.name == "strong100":
        depth_threshold = 2.11
    else:
        depth_threshold = 1.305
    # Step 3: Generate the filename based on methodologies
    filename = "graph_data"
    if add_hq_edges:
        filename += "_add_edges"
    if remove_existing_edges:
        filename += "_remove_edges"
    if depth_pruning:
        filename += "_depth_prune"
    if depth_linking:
        filename += "_depth_linking"
    filename += ".pt"

    filepath = os.path.join(cache_dir, filename)

    # Step 4: Check if the graph is already saved
    if os.path.exists(filepath):
        print(f"Loading saved graph from {filepath}")
        data = torch.load(filepath)
        # Compare the loaded graph with the original to compute changes
        original_edges = get_undirected_edge_set(data.edge_index)
        altered_edges = get_undirected_edge_set(torch.load(filepath).edge_index)
        changes['added'] = len(altered_edges - original_edges)
        changes['removed'] = len(original_edges - altered_edges)
        # Calculate node degrees using torch.bincount
        node_degrees = torch.bincount(data.edge_index[0], minlength=data.num_nodes)
        # If the graph is undirected, add degrees from the reverse edges
        if is_undirected(data.edge_index):
            node_degrees += torch.bincount(data.edge_index[1], minlength=data.num_nodes)
        # Calculate the number of isolated nodes
        isolated = (node_degrees == 0).sum().item()
        changes['isolated'] = isolated
        print(f"ISOLATED NODES FOUND: {isolated}")
    else:
        print(f"Updating graph and saving to {filepath}")
        # Step 5: Apply methodologies using update_graph_edges
        data, changes = update_graph_edges(
            data=data,
            dataset=dataset,
            base_directory=os.path.join(base_directory,dataset.name),
            node_depths=node_depths,
            add_hq_edges=add_hq_edges,
            remove_existing_edges=remove_existing_edges,
            depth_pruning=depth_pruning,
            depth_linking=depth_linking,
            depth_diff_threshold=depth_threshold,
            similarity_threshold=similarity_threshold
        )
        # Step 6: Save the updated graph
        torch.save(data, filepath)
    
    return data, changes