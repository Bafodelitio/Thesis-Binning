from torch_geometric.transforms import NormalizeFeatures
from util import AssemblyDataset, apply_pca_to_node_features
from SAGE_unsup_aux import create_pyg_graph_from_assembly, hyperparameter_tuning, calculate_avg_edges_per_node,load_or_update_graph
from pathlib import Path
# Baseline Cluster        

# DATA LOADING & PRE-PROCESSING

def main():
    
    # choose dataset in use
    #name = "aale"
    name = "strong100"
    
    dataset = AssemblyDataset(
        name=name,
        data_dir=name,
        fastafile="assembly.fasta",       
        graphfile="assembly_graph.gfa", 
        depthfile="assembly_depth.txt",
        scgfile="marker_gene_stats.tsv",
        labelsfile=None,
        featuresfile="features.tsv",
        cache_dir=name+"_cache"
    )

    if(dataset.check_cache()):
        dataset.read_cache()
    else:
        dataset.read_assembly()
                    
    data = create_pyg_graph_from_assembly(dataset)

    
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
        'out_channels': 64,          
        'num_layers': 4,             
        'dropout': 0.1,             
        'lr': 0.005,                
        'weight_decay': 0.02,        
        'margin': 0.06,              
        'num_neighbors': 3,  
        'epochs': 50,             
        'num_components': None      
    }
    
    
    # Define the range of values for each parameter
    batch_size_values = [64, 128, 256, 512]     
    hidden_channels_values = [64, 128, 256, 512]  
    out_channels_values = [16, 32, 64, 128]  
    num_layers_values = [2, 3, 4, 5, 6]  
    lr_values = [0.0001, 0.0005, 0.001, 0.005] 
    margin_values = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1]  
    num_neighbor_values = [2, 3, 4]  
    epoch_values = [50, 100, 200]  
    weight_decay_values = [0, 0.01, 0.02, 0.05]  
    dropout_values = [0, 0.2, 0.3, 0.4]  



    # Specify the base directory for saving model results
    current_dir = Path(__file__).parent
    base_path = "E:/FCT/Tese/Code/GNN/temp_res"  # Change this to your desired base path"
    print(f"Results will be saved in: {base_path}")


    # Perform hyperparameter tuning
    # Este método treina o modelo com a configuração base de parametros 
    # e iterativamente atualiza o "parameter" para cada valor de "values", 
    # voltando a correr o modelo em cada iteração
    # (neste caso, o parâmetro values=None significa que o método vai correr apenas a configuração base sem alterações)
    
    
    
    hyperparameter_tuning(dataset=dataset, 
                          data=data, 
                          parameter='num_layers', 
                          values=None, 
                          base_config=base_config, 
                          base_path=base_path, 
                          directory = current_dir, 
                          update_edges=False, 
                          epoch_cluster=0)
    
    


    
if __name__ == "__main__":
    main()