from glob import escape
from importlib.resources import files
import itertools
import os
import pickle
import shutil
from typing import Dict, List
from collections import Counter
import numpy as np
import networkx as nx
import scipy.stats as stats
import scipy.sparse
import ast
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
import torch
import re
from torch_geometric.utils import is_undirected
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.stats import percentileofscore
from pathlib import Path

def open_gfa_file(filename, filter=1000, root=False, kmer=4):
    G = nx.Graph()
    if root:
        root_node = G.add_node("root", length=0)
    skipped_contigs = set()
    skipped_edges = set()
    kmer_to_id, canonical_k = get_kmer_to_id(kmer)
    with open(filename, "r") as f:
        for line in f:
            if line.startswith("S"):
                values = line.strip().split()
                contigid = values[1]
                # contiglen = int(values[3].split(":")[-1])
                contiglen = len(values[2])
                contig_kmers = count_kmers(values[2], kmer, kmer_to_id, canonical_k)
                if contiglen < filter:
                    skipped_contigs.add(contigid)
                else:

                    G.add_node(contigid, length=contiglen, kmers=contig_kmers)  # TODO: add coverage and kmer too
                    if root:
                        G.add_edge(contigid, "root")
            if line.startswith("L"):
                values = line.strip().split()
                if values[1] in skipped_contigs or values[3] in skipped_contigs:
                    skipped_edges.add((values[1], values[3]))
                    continue
                G.add_edge(values[1], values[3])

    print("load graph from GFA")
    print("skipped these contigs {} (len<{})".format(len(skipped_contigs), filter))
    print("skipped these edges {} (len<{})".format(len(skipped_edges), filter))
    print("Done Loading")
    return G, kmer_to_id


def read_marker_gene_sets(lineage_file):
    """Open file with gene sets for a taxon

    :param lineage_file: path to marker set file from CheckM (Bacteria.ms)
    :type lineage_file: str
    :return: Marker sets
    :rtype: set
    """
    with open(lineage_file, "r") as f:
        lines = f.readlines()
    # consider only single taxon gene set
    sets = lines[1].strip().split("\t")[-1]
    sets = ast.literal_eval(sets)
    return sets


def read_contig_genes(contig_markers):
    """Open file mapping contigs to genes

    :param contig_markers: path to contig markers (marker stats)
    :type contig_markers: str
    :return: Mapping contig names to markers
    :rtype: dict
    """
    contigs = {}
    with open(contig_markers, "r") as f:
        for line in f:
            values = line.strip().split("\t")
            contig_name = values[0]
            # keep only first two elements
            contig_name = "_".join(contig_name.split("_")[:2])
            contigs[contig_name] = {}
            mappings = ast.literal_eval(values[1])
            for contig in mappings:
                for gene in mappings[contig]:
                    if gene not in contigs[contig_name]:
                        contigs[contig_name][gene] = 0
                    # else:
                    #    breakpoint()
                    contigs[contig_name][gene] += 1
                    if len(mappings[contig][gene]) > 1:
                        breakpoint()
    return contigs

def get_markers_to_contigs(marker_sets, contigs):
    """Get marker to contig mapping

    :param marker_sets: Marker sets from CheckM
    :type marker_sets: set
    :param contigs: Contig to marker mapping
    :type contigs: dict
    :return: Marker to contigs list mapping
    :rtype: dict
    """
    marker2contigs = {}
    for marker_set in marker_sets:
        for gene in marker_set:
            marker2contigs[gene] = []
            for contig in contigs:
                if gene in contigs[contig]:
                    marker2contigs[gene].append(contig)
    return marker2contigs

#BACTERIA_MARKERS = "data/Bacteria.ms"
#kernel = np.load("data/kernel.npz")['arr_0']
BACTERIA_MARKERS = Path(__file__).parent / 'data' / 'Bacteria.ms'
kernel = np.load(Path(__file__).parent / 'data' / 'kernel.npz')['arr_0']

def count_kmers(seq, k, kmer_to_id, canonical_k):
    # Used in case kmers are used as input features
    # https://stackoverflow.com/q/22428020
    #breakpoint()
    kmer_letters = set(["A", "T", "C", "G"])
    kmers = [seq[i : i + k] for i in range(len(seq) - k + 1) if set(seq[i : i + k]).issubset(kmer_letters)]
    kmers = [kmer_to_id[k] for k in kmers]
    kmer_counts = Counter(kmers)
    counts = np.array([kmer_counts[k] for k in range(canonical_k)])
    counts = counts / counts.sum()
    counts += -(1/(4**k))
    counts = np.dot(counts, kernel)
    return counts

def get_kmer_to_id(kmer, combine_revcomp=False):
    kmer_to_ids = {}
    BASE_COMPLEMENT = {"A": "T", "T": "A", "G": "C", "C": "G"}
    all_kmers = itertools.product("ACGT", repeat=kmer)
    all_kmers = ["".join(k) for k in all_kmers]
    new_id = 0
    for kmer in all_kmers:
        if kmer not in kmer_to_ids:
            kmer_to_ids[kmer] = new_id
            if combine_revcomp:
                rev_compl = "".join(tuple([BASE_COMPLEMENT[x] for x in reversed(kmer)]))
                kmer_to_ids[rev_compl] = new_id
            new_id += 1
    return kmer_to_ids, new_id

def process_node_name(name: str, assembly_type: str="flye") -> str:
    contig_name = name.strip().split(" ")[0]
    #if assembly_type == "spades":
    contig_name = "_".join(contig_name.split("_")[:2])
    return contig_name


class AssemblyDataset:
# Read contig names from fasta file and save to disk
# read graph
# other features are loaded by functions
# everything as numpy to make it easier to convert
    def __init__(
        self,
        name,
        data_dir,
        fastafile,
        graphfile,
        depthfile,
        scgfile,
        labelsfile,
        featuresfile,
        cache_dir,
        assemblytype="flye",
        min_contig_length=0,
        contignodes=False
    ):
        self.name = name
        self.data_dir = data_dir
        self.fastafile = fastafile
        self.graphfile = graphfile
        self.depthfile = depthfile
        self.assembly_type = assemblytype
        self.cache_dir = cache_dir
        self.labelsfile = labelsfile
        self.featuresfile = featuresfile
        self.scgfile = scgfile
        self.load_kmer = True
        self.kmer = 4
        self.min_contig_length = min_contig_length
        self.contignodes = contignodes
        if self.load_kmer:
            self.kmer_to_ids, self.canonical_k = get_kmer_to_id(self.kmer)

        # initialize data features which can be cached as numpy or pickle
        self.node_names = []
        self.node_seqs = {}  # should it be saved?
        self.graph_nodes = []  # nodes that are part of the assembly graph
        self.graph_paths = {} # contig paths identified by assembler
        self.node_lengths = []
        self.node_kmers = []
        self.node_depths = []
        self.edges_src = []
        self.edges_dst = []
        self.edge_weights = []
        self.adj_matrix = None
        self.neg_pairs_idx = None  # cached on the main function, only necessary for TF models
        
        # initialize labels
        self.labels = []
        self.node_to_label = {}
        self.label_to_node = {}
        
        self.contig_markers = {}
        self.ref_marker_sets = {}

        #aux
        #self.coverage_threshold = None
        
    def read_assembly(self):
        """Read assembly files, convert to numpy arrays and save to disk"""
        self.read_seqs()
        np.save(os.path.join(self.cache_dir, "node_names.npy"), self.node_names)
        np.save(os.path.join(self.cache_dir, "node_lengths.npy"), self.node_lengths)
        np.save(os.path.join(self.cache_dir, "node_attributes_kmer.npy"), self.node_kmers.astype(float))
        pickle.dump(self.node_seqs, open(os.path.join(self.cache_dir, "node_seqs.pkl"), "wb"))
        if os.path.exists(os.path.join(self.data_dir, self.graphfile)):
            if not self.contignodes:
                self.read_gfa()
            else:
                self.read_gfa_contigs()
            np.save(os.path.join(self.cache_dir, "edge_weights.npy"), self.edge_weights)
            scipy.sparse.save_npz(os.path.join(self.cache_dir, "adj_sparse.npz"), self.adj_matrix)
            np.save(os.path.join(self.cache_dir, "graph_nodes.npy"), self.graph_nodes)
            pickle.dump(self.graph_paths, open(os.path.join(self.cache_dir, "graph_paths.pkl"), 'wb'))
        # self.filter_contigs()
        # self.rename_nodes_to_index()
        # self.nodes_depths = np.array(self.nodes_depths)
        # read lengths
        self.read_depths()
        np.save(os.path.join(self.cache_dir, "node_attributes_depth.npy"), np.array(self.node_depths))
        self.read_labels()
        np.save(os.path.join(self.cache_dir, "node_to_label.npy"), self.node_to_label)
        np.save(os.path.join(self.cache_dir, "label_to_node.npy"), self.label_to_node)
        np.save(os.path.join(self.cache_dir, "labels.npy"), self.labels)
        print("reading SCGs")
        self.read_scgs()
        np.save(os.path.join(self.cache_dir, "contig_markers.npy"), self.contig_markers)
        np.save(os.path.join(self.cache_dir, "ref_marker_sets.npy"), self.ref_marker_sets)
        #self.calculate_coverage_threshold()
        #np.save(os.path.join(self.cache_dir, "coverage_threshold.npy"), self.coverage_threshold)

    def print_stats(self):
        print("==============================")
        print("DATASET STATS:")
        # get:
        #   number of sequences
        #   number of contigs
        #   length of contigs (sum and average and N50)
        #   coverage samples from jgi file
        #   assembly_graph.file exists, number of edges, number of paths
        print("number of sequences: {}".format(len(self.node_names)))
        print("assembly length: {} Gb".format(round(sum(self.node_lengths) / 1000000000, 3)))
        print("assembly N50: {} Mb".format(round(self.calculate_n50()/1000000, 3)))
        print("assembly average length (Mb): {} max: {} min: {}".format(round(np.mean(self.node_lengths)/1000000, 3),
                                                                    round(np.max(self.node_lengths)/1000000, 3),
                                                                    round(np.min(self.node_lengths)/1000000, 3)))
#       print("coverage samples: {}".format(len(self.node_depths[0])))
        if os.path.exists(os.path.join(self.data_dir, self.graphfile)) or \
            len(self.edges_src) > 0:
            print("Graph file found and read")
            print("graph edges: {}".format(len(self.edges_src)))
            print("contig paths: {}".format(len(self.graph_paths)))
        else:
            print("No assembly graph loaded")
        #   contigs with markers on marker_gene_stats
        #   stats with SCGs (max/min # of contigs, etc)

        if len(self.contig_markers) > 0:
            print("total ref markers sets: {}".format(len(self.ref_marker_sets)))
            print("total ref markers: {}".format(len(self.markers)))
            n_of_markers = [len(x) for x in self.contig_markers.values() if len(x) > 0]
            print("contigs with one or more markers: {}/{}".format(len(n_of_markers),
                                                                    len(self.node_names)))
            
            print("max SCGs on one contig: {}, average(excluding 0): {}".format(max(n_of_markers),
                                                            round(np.mean(n_of_markers), 3)))
            self.estimate_n_genomes()
            print("SCG contig count min: {} contigs".format(min(self.scg_counts.values())))
            self.get_edges_with_same_scgs()
        else:
            print("No SCG markers")
        # labels
        if self.labelsfile is not None or len(self.labels) > 1:
            print("number of GS labels: {}".format(len(self.labels)))
            nodes_not_na = [n for n in self.node_names if n in self.node_to_label and \
                                                        self.node_to_label[n] != "NA"]
            print("nodes with labels: {}".format(len(nodes_not_na)))
        print("==============================")
                                                            
    def calculate_n50(self):
        # https://eaton-lab.org/slides/genomics/answers/nb-4.1-numpy.html
        contig_sizes = self.node_lengths.copy()
        contig_sizes.sort()
        contig_sizes = contig_sizes[::-1]
        total_len = contig_sizes.sum()
        half_total_len = total_len / 2
        contig_sum_lens = np.zeros(contig_sizes.size, dtype=int)
        for i in range(contig_sizes.size):
            contig_sum_lens[i] = contig_sizes[i:].sum()
        which_contigs_longer_than_half = contig_sum_lens > half_total_len
        contigs_longer_than_half = contig_sizes[which_contigs_longer_than_half]
        n50 = contigs_longer_than_half.min()
        return n50

    def read_cache(self, load_graph=True):
        prefix = os.path.join(self.cache_dir, "{}")
        self.node_names = list(np.load(prefix.format("node_names.npy")))
        self.node_seqs = pickle.load(open(prefix.format("node_seqs.pkl"), "rb"))
        # graph nodes
        self.node_lengths = np.load(prefix.format("node_lengths.npy"))
        if load_graph:
            self.graph_nodes = np.load(prefix.format("graph_nodes.npy"))
            self.graph_paths = pickle.load(open(prefix.format("graph_paths.pkl"), 'rb'))
            self.adj_matrix = scipy.sparse.load_npz(prefix.format("adj_sparse.npz"))
            self.edge_weights = self.adj_matrix.data
            self.edges_src = self.adj_matrix.row
            self.edges_dst = self.adj_matrix.col
        self.node_kmers = np.load(prefix.format("node_attributes_kmer.npy"))
        self.node_depths = np.load(prefix.format("node_attributes_depth.npy"))
        # self.edge_weights = np.load(prefix.format("edge_weights.npy"))
        
        # labels
        self.node_to_label = np.load("{}/node_to_label.npy".format(self.cache_dir), allow_pickle=True)[()]
        self.label_to_node = np.load("{}/label_to_node.npy".format(self.cache_dir), allow_pickle=True)[()]
        self.labels = list(np.load("{}/labels.npy".format(self.cache_dir)))
        if os.path.exists(prefix.format("true_adj_sparse.npz")):
            self.true_adj_matrix = scipy.sparse.load_npz(prefix.format("true_adj_sparse.npz"))
        self.read_scgs()
        #self.coverage_threshold = np.load("{}/coverage_threshold.npy".format(self.cache_dir), allow_pickle=True)[()]
        print("DATA LOADED")

    def check_cache(self, require_graph=True):
        """check if all necessary files exist in cache"""
        prefix = os.path.join(self.cache_dir, "{}")
        if require_graph:
            return (
                os.path.exists(prefix.format("node_names.npy"))
                and os.path.exists(prefix.format("node_lengths.npy"))
                and os.path.exists(prefix.format("node_attributes_kmer.npy"))
                #and os.path.exists(prefix.format("node_attributes_depth.npy"))
                and os.path.exists(prefix.format("edge_weights.npy"))
                and os.path.exists(prefix.format("adj_sparse.npz"))
                and os.path.exists(prefix.format("graph_nodes.npy"))
                and os.path.exists(prefix.format("graph_paths.pkl"))
                and os.path.exists(prefix.format("node_to_label.npy"))
                and os.path.exists(prefix.format("label_to_node.npy"))
                and os.path.exists(prefix.format("labels.npy"))
            )
        else:
            return (
                os.path.exists(prefix.format("node_names.npy"))
                and os.path.exists(prefix.format("node_lengths.npy"))
                and os.path.exists(prefix.format("node_attributes_kmer.npy"))
                #and os.path.exists(prefix.format("node_attributes_depth.npy"))
                and os.path.exists(prefix.format("node_to_label.npy"))
                and os.path.exists(prefix.format("label_to_node.npy"))
                and os.path.exists(prefix.format("labels.npy"))
            )

    def read_seqs(self):
        """Read sequences from fasta file, write to self.node_seqs"""
        node_lengths = {}
        node_kmers = {}
        contig_name = None
        with open(os.path.join(self.data_dir, self.fastafile), "r") as f:
            for line in f:
                if line.startswith(">"):
                    if len(self.node_names) > 0:
                        # finish last contig
                        node_lengths[contig_name] = len(self.node_seqs[contig_name])
                        if self.load_kmer:
                            kmers = count_kmers(
                                self.node_seqs[contig_name], self.kmer, self.kmer_to_ids, self.canonical_k
                            )
                            node_kmers[contig_name] = kmers
                    contig_name = process_node_name(line[1:], self.assembly_type)
                    self.node_names.append(contig_name)
                    self.node_seqs[contig_name] = ""
                else:
                    self.node_seqs[contig_name] += line.strip()
        # add last
        node_lengths[contig_name] = len(self.node_seqs[contig_name])
        if self.load_kmer:
            kmers = count_kmers(self.node_seqs[contig_name], self.kmer, self.kmer_to_ids, self.canonical_k)
            node_kmers[contig_name] = kmers
            # convert kmers to numpy
            self.node_kmers = np.array(stats.zscore([node_kmers[n] for n in self.node_names], axis=0))
        self.node_lengths = np.array([node_lengths[n] for n in self.node_names])    

    def read_gfa(self):
        """Read graph file from GFA format, save list of start and end nodes to self.edges_src and self.edges_dst
        Check if contig names match the fasta file from self.node_seqs
        """
        skipped_contigs = set()
        with open(os.path.join(self.data_dir, self.graphfile), "r") as f:
            for line in f:
                if line.startswith("S"):  # sequence
                    values = line.strip().split()
                    node_name = process_node_name(values[1], self.assembly_type)
                    if node_name not in self.node_names:
                        skipped_contigs.add(node_name)
                        continue
                    contig_seq = self.node_seqs.get(node_name, "")  # discard missing contigs
                    contiglen = len(contig_seq)

                    if contiglen < self.min_contig_length:
                        skipped_contigs.add(node_name)
                    else:
                        self.graph_nodes.append(node_name)
                        # self.nodes_data.append([])
                elif line.startswith("L"):  # link/edge
                    values = line.strip().split()  # TAG, SRC, SIGN, DEST, SIGN, 0M, RC
                    src_node_name = process_node_name(values[1], self.assembly_type)
                    dst_node_name = process_node_name(values[3], self.assembly_type)
                    if src_node_name in skipped_contigs or dst_node_name in skipped_contigs:
                        # skipped_edges.add((contig_names.index(values[1]), contig_names.index(values[3])))
                        continue
                    src_index = self.node_names.index(src_node_name)
                    dst_index = self.node_names.index(dst_node_name)
                    self.edges_src.append(src_index)
                    self.edges_dst.append(dst_index)
                    if len(values) > 6:
                        rc = int(values[6].split(":")[-1])
                    else:
                        rc = 1
                    self.edge_weights.append(rc)
                    # reverse too
                    if values[1] != values[3]:
                        self.edges_src.append(dst_index)
                        self.edges_dst.append(src_index)
                        self.edge_weights.append(rc)
                elif line.startswith("P"):
                    values = line.strip().split()  # P contig path
                    self.graph_paths[values[1]] = [self.node_names.index(path_node_name[:-1]) \
                        for path_node_name in values[2].split(",") if path_node_name[:-1] in self.node_names]
        self.adj_matrix = scipy.sparse.coo_matrix(
            (self.edge_weights, (self.edges_src, self.edges_dst)), shape=(len(self.node_names), len(self.node_names))
        )
        self.edge_weights = np.array(self.edge_weights)

    def read_gfa_contigs(self):
        skipped_contigs = set()
        edge_edge_links = {}
        contig_edge_links = {}
        with open(os.path.join(self.data_dir, self.graphfile), "r") as f:
            # ignore S entries (these are not contigs)
            # the node seqs are read from the assembly file
            for line in f:
                if line.startswith("L"): # edge links
                    values = line.strip().split()  # TAG, SRC, SIGN, DEST, SIGN, 0M, RC
                    src_node_name = process_node_name(values[1], self.assembly_type)
                    dst_node_name = process_node_name(values[3], self.assembly_type)
                    edge_edge_links.setdefault(src_node_name,set()).add(values[3])
                    edge_edge_links.setdefault(dst_node_name,set()).add(values[1])
                elif line.startswith("P"): # contig to edge links
                    values = line.strip().split()
                    contig_name = process_node_name(values[1], self.assembly_type)
                    if contig_name in skipped_contigs or contig_name not in self.node_names:
                        #skipped_edges.add((contig_names.index(values[1]), contig_names.index(values[3])))
                        continue
                    for edge in values[2].split(","):
                        contig_edge_links.setdefault(contig_name,set()).add(edge[:-1])
                    self.graph_paths[values[1]] = [self.node_names.index(path_node_name[:-1]) \
                        for path_node_name in values[2].split(",") if path_node_name[:-1] in self.node_names]
        if len(contig_edge_links) == 0:
            # load graph paths from assembly_info
            # TODO: check if file exists
            contig_edge_links = self.read_assembly_info()
        print("skipped contigs", len(skipped_contigs), "<", self.min_contig_length)
        # create graph, , and to linked to adjacent edges
        # first create edge to contig index
        edge_contig_links = {}
        for contig in contig_edge_links:
            for e in contig_edge_links[contig]:
                edge_contig_links.setdefault(e, set()).add(contig)
        for contig in contig_edge_links:
            src_index = self.node_names.index(contig)
            for edge in contig_edge_links[contig]:
                # connect contigs linked to same edge
                for contig2 in edge_contig_links[edge]:
                    dst_index = self.node_names.index(contig2)
                    if contig2 != contig:
                        self.edges_src.append(src_index)
                        self.edges_dst.append(dst_index)
                        self.edge_weights.append(1)
                        # inverse is added when we get to contig2
                for edge2 in edge_edge_links.get(edge, []):
                    for contig2 in edge_contig_links.get(edge2, []):
                        dst_index = self.node_names.index(contig2)
                        self.edges_src.append(src_index)
                        self.edges_dst.append(dst_index)
                        self.edge_weights.append(1)
        #.info(f"skipped contigs {len(skipped_contigs)} < {self.min_contig_length}")
        self.adj_matrix = scipy.sparse.coo_matrix(
            (self.edge_weights, (self.edges_src, self.edges_dst)), shape=(len(self.node_names), len(self.node_names))
        )
        self.edge_weights = np.array(self.edge_weights)

    def read_assembly_info(self, get_graph_paths=False, circular_length=100000) -> Dict[str,list]:
        """Read assembly_info.txt file generated by flye
        """
        self.circular_contigs: List[str] = []
        self.repetitive_contigs: List[str] = []
        self.contigs_multiplicity = {}
        contig_edge_links = {}
        with open(os.path.join(self.data_dir, "assembly_info.txt"), "r") as f:
            next(f) # skip header
            for line in f:
                values = line.strip().split("\t")
                contig_name = process_node_name(values[0], self.assembly_type)
                if values[3] == "Y" and int(values[1]) > circular_length:
                    self.circular_contigs.append(contig_name)
                if values[4] == "Y":
                    self.repetitive_contigs.append(contig_name)
                #if contig_name not in self.node_names:
                    #skipped_edges.add((contig_names.index(values[1]), contig_names.index(values[3])))
                #    continue
                self.contigs_multiplicity[contig_name] = int(values[5])
                # get graph_paths
                for edge in values[-1].split(","):
                    if edge != "*":
                        contig_edge_links.setdefault(contig_name,set()).add("edge_" + edge.strip("-"))
                if get_graph_paths:
                    self.graph_paths[contig_name] = ["edge_" + e for e in contig_edge_links[contig_name]]
                #for edge_name in self.graph_paths[contig_name]:
                #    self.contigs_multiplicity[self.node_names.index(edge_name)] = int(values[5])
        return contig_edge_links

    def print_circular_contigs(self):
        """Print circular contigs and info, based on flye
        """
        for contig_name in self.circular_contigs:
            print("found circular contig", contig_name,
                    "mult.", self.contigs_multiplicity[contig_name])
            # get edges of this contig
            contig_edges = self.graph_paths[contig_name]
            print("---", "name", "depth", "markers", "label", "edges")
            for edge_id in contig_edges:
                #edge_id = self.node_names.index("edge_" + edge)
                edge_name = self.node_names[edge_id]
                print("---", edge_id, edge_name, self.node_lengths[edge_id],
                        [round(d, 4) for d in self.node_depths[edge_id]],
                        self.node_to_label[edge_name],
                        len(self.contig_markers.get(edge_name, [])),
                        list(self.adj_matrix.getrow(edge_id).data))

    def read_depths(self) -> None:
        node_depths: Dict[str, np.array] = {}
        if self.depthfile is not None and os.path.exists(os.path.join(self.data_dir, self.depthfile)):
            if self.depthfile.endswith(".npz"):
                self.node_depths = np.load(open(os.path.join(self.data_dir, self.depthfile), "rb"))["arr_0"]
                if self.node_depths.shape[0] != len(self.node_names):
                    print("depth npz file mismatch:")
                    breakpoint()
            else:
                with open(os.path.join(self.data_dir, self.depthfile)) as f:
                    header = next(f)
                    depth_i = [i + 3 for i, n in enumerate(header.split("\t")[3:]) if "-var" not in n]
                    # var_i = [i + 3 for i, n in enumerate(header.split("\t")[3:]) if "-var" in n]
                    for line in f:
                        values = line.strip().split()
                        node_name = process_node_name(values[0], self.assembly_type)
                        if node_name in self.node_names:
                            node_depths[node_name] = np.array([float(values[i]) for i in depth_i])                
                self.node_depths = np.array([node_depths.get(n, [0]*len(depth_i)) for n in self.node_names])
            if len(self.node_depths[0]) > 1:  # normalize depths
                depthssum = self.node_depths.sum(axis=1) + 1e-10
                self.node_depths /= depthssum.reshape((-1, 1))
            else:
                self.node_depths = stats.zscore(self.node_depths, axis=0)
        else:
            self.node_depths = np.ones(len(self.node_names), dtype=np.float32)

    def read_labels(self):
        # logging.info("loading labels from {}".format(args.labels))
        if self.labelsfile is not None:
            node_to_label = {c: "NA" for c in self.node_names}
            labels = set(["NA"])
            with open(os.path.join(self.data_dir, self.labelsfile), "r") as f:
                for line in f:
                    # label, node = line.strip().split()
                    if self.labelsfile.endswith(".csv"):
                        values = line.strip().split(",")
                    elif self.labelsfile.endswith(".tsv"):  # amber format
                        if line.startswith("@") or line.startswith("#"):
                            continue
                        values = line.strip().split("\t")
                    if len(values) < 2:
                        breakpoint()
                    node_name = process_node_name(values[0], self.assembly_type)
                    label = values[1]
                    if node_name in node_to_label:
                        node_to_label[node_name] = label
                        labels.add(label)
                    else:
                        breakpoint()
            labels = list(labels)
            label_to_node = {s: [] for s in labels}
            for n in node_to_label:
                s = node_to_label[n]
                label_to_node[s].append(n)
            self.node_to_label = {n: l for n, l in node_to_label.items()}
            self.labels = labels
            self.label_to_node = label_to_node
            # calculate homophily
            if len(self.edge_weights) > 1:
                self.calculate_homophily()
        else:
            self.labels = ["NA"]
            self.label_to_node = {"NA": self.node_names}
            self.node_to_label = {n: "NA" for n in self.node_names}

    def calculate_homophily(self):
        """
        Calculate percentage of edges that connect nodes with the same label
        """
        positive_edges = 0
        edges_without_label = 0
        for u, v in zip(self.edges_src, self.edges_dst):
            if self.node_to_label.get(self.node_names[u], "NA") == "NA" or \
                self.node_to_label.get(self.node_names[v], "NA") == "NA":
                edges_without_label += 1
            if self.node_to_label[self.node_names[u]] == self.node_to_label[self.node_names[v]] and self.node_to_label[self.node_names[u]] != "NA":
                positive_edges += 1
    def read_scgs(self):
        # Load contig marker genes (Bacteria list)
        if self.scgfile is not None:
            # logging.info("loading checkm results")
            scg_path = os.path.join(self.data_dir, self.scgfile)
            ref_sets = read_marker_gene_sets(BACTERIA_MARKERS)
            contig_markers = read_contig_genes(scg_path)
            self.ref_marker_sets = ref_sets
            self.contig_markers = contig_markers
            print(f"REF_SET LENGTH = {len(ref_sets)} AND CONtiG_MARKER LENGTH = {len(contig_markers)}")
            marker_counts = get_markers_to_contigs(ref_sets, contig_markers)
            self.markers = marker_counts
        else:
            self.ref_marker_sets = {}
            self.contig_markers = {}
            self.run_checkm()

    def get_all_different_idx(self):
        """
        Returns a 2d numpy array where each row
        corresponds to a pairs of node idx whose
        feature must be different as they correspond
        to the same contig (check jargon). This
        should encourage the HQ value to be higher.
        """
        node_names_to_idx = {node_name: i for i, node_name in enumerate(self.node_names)}
        pair_idx = set()
        for n1 in self.contig_markers:
            for gene1 in self.contig_markers[n1]:
                for n2 in self.contig_markers:
                    if n1 != n2 and gene1 in self.contig_markers[n2]:
                        p1 = (node_names_to_idx[n1], node_names_to_idx[n2])
                        p2 = (node_names_to_idx[n2], node_names_to_idx[n1])
                        if (p1 not in pair_idx) and (p2 not in pair_idx):
                            pair_idx.add(p1)
        pair_idx = np.unique(np.array(list(pair_idx)), axis=0)
        print("Number of diff cluster pairs:", len(pair_idx))
        self.neg_pairs_idx = pair_idx

    def run_vamb(self, vamb_outdir, cuda, vambdim):
        from vamb.vamb_run import run as run_vamb
        batchsteps = []
        vamb_epochs = 500
        if len(self.node_depths[0]) == 1:
            vamb_bs = 32
            batchsteps = [25, 75, 150]
        else:
            vamb_bs = 64
            batchsteps = [25, 75, 150, 300]
        nhiddens = [512, 512]

        vamb_logpath = os.path.join(vamb_outdir, "log.txt")
        if os.path.exists(vamb_outdir) and os.path.isdir(vamb_outdir):
            shutil.rmtree(vamb_outdir)
        os.mkdir(vamb_outdir)
        with open(vamb_logpath, "w") as vamb_logfile:
            run_vamb(
                outdir=vamb_outdir,
                fastapath=os.path.join(self.data_dir, self.fastafile),
                jgipath=os.path.join(self.data_dir, self.depthfile),
                logfile=vamb_logfile,
                cuda=cuda,
                batchsteps=batchsteps,
                batchsize=vamb_bs,
                nepochs=vamb_epochs,
                mincontiglength=self.min_contig_length,
                nhiddens=nhiddens,
                nlatent=int(vambdim),
                norefcheck=True,
            )
            if self.data_dir != "" and self.featuresfile.endswith(".tsv"):
                shutil.copyfile(os.path.join(vamb_outdir, "embs.tsv"), self.featuresfile)

    def read_features(self):
        node_embs = {}
        if self.featuresfile.endswith(".tsv"):
            with open(self.featuresfile, "r") as ffile:
                for line in ffile:
                    values = line.strip().split()
                    node_embs[values[0]] = [float(x) for x in values[1:]]
        elif self.featuresfile.endswith(".pickle"):
            with open(self.featuresfile, "rb") as ffile:
                node_embs = pickle.load(ffile)
        self.node_embs = [
                node_embs.get(n, np.random.Generator(10e-5, 1.0, len(node_embs[list(node_embs.keys())[0]]))) for n in self.node_names
            ]  # deal with missing embs
        self.node_embs = np.array(self.node_embs)

    def write_features_tsv(self):
        with open(self.featuresfile, "w") as ffile:
            for i, emb in enumerate(self.node_embs):
                ffile.write(f"{self.node_names[i]} {' '.join(map(str,emb.tolist()))}\n")

    def get_topk_neighbors(self, k, scg_only=False):
        """
        Returns a list of the top k neighbors for each node. Use kmers and abundance

        """
        #breakpoint()
        self.topk_neighbors = []
        features = np.concatenate((self.node_kmers, self.node_depths), axis=1)
        cosine_dists = np.dot(features, features.T)
        for i in range(len(self.node_names)):
            self.topk_neighbors.append(set(np.argsort(cosine_dists[i])[-k:]))

    def estimate_n_genomes(self):
        self.scg_counts = {}
        for marker_set in self.ref_marker_sets:
            for gene in marker_set:
                self.scg_counts[gene] = 0
                for contig in self.contig_markers:
                    if gene in self.contig_markers[contig]:
                        self.scg_counts[gene] += self.contig_markers[contig][gene]

        #print(self.scg_counts)
        quartiles = np.percentile(list(self.scg_counts.values()), [25, 50, 75])
        print("candidate k0s", sorted(set([k for k in self.scg_counts.values() if k >= quartiles[2]])))
        return max(self.scg_counts.values())
        
        
    def run_checkm(self, nthreads=10, tempdir="../temp"):
        # check if checkm is installed
        checkm_is_avail = shutil.which("checkm") is not None
        # if not, print commands only
        commands = [
            "mkdir {}/nodes".format(self.data_dir), #; cd nodes; ",
            "cat {0}/{1} | awk '{{ if (substr($0, 1, 1)=='>') {{filename=(substr($0,2) '.fa')}} print $0 > {0}/filename }}".format(self.data_dir, self.fastafile),
            'find {}/nodes/ -name "* *" -type f | rename "s/ /_/g"'.format(self.data_dir),
            "checkm taxonomy_wf --tmpdir {0} -t {1} -x fa domain Bacteria {2}/nodes/ {2}/checkm_nodes/".format(tempdir, nthreads, self.data_dir)
        ]
        if checkm_is_avail:
            #run commands one by one
            for cmd in commands:
                os.system(cmd)
        else:
            print("Run this on a machine with CheckM installed")
            for cmd in commands:
                print(cmd)


    def read_gtdbtk_files(self, prefix="gtdbtk", suffix_labels="summary.tsv",
                            suffix_markers="markers_summary.tsv", reset=True):
        # use summary to add species level labels
        # also save full lineages
        roots = ["bac120"] #, "ar53"]
        no_species = 0
        no_genus = 0
        species_labels = set()
        # reset labels
        if reset:
            self.node_to_label = {n: "NA" for n in self.node_names}
            self.labels = ["NA"]
        contig_markers = {n: {} for n in self.node_names}
        ref_markers = set()
        for root in roots:
            # read labels
            with open(os.path.join(self.data_dir, f"{prefix}.{root}.{suffix_labels}"), 'r') as f:
                next(f) # skip header
                for line in f:
                    values = line.strip().split("\t")
                    node_name = process_node_name(values[0], self.assembly_type)
                    if values[1].startswith("Unclassified"):
                        continue # do not add unclassified nodes
                    classification = values[1].split(";")
                    node_slabel = classification[-1]
                    node_glabel = classification[-2]
                    #if values[5] != "N/A":
                    #    fast_ani = float(values[5])
                    #    anis.append(fast_ani)
                    if node_slabel == "s__":
                        no_species += 1
                    else:
                        self.node_to_label[node_name] = node_slabel
                        species_labels.add(node_slabel)
                    #if node_slabel == "g__":
                    #    no_genus += 1
                    #else:
                    #    node_to_glabel[node_name] = node_glabel
                    #    genus_labels.add(node_glabel)

            # read markers
            with open(os.path.join(self.data_dir, f"{prefix}.{root}.{suffix_markers}"), 'r') as f:
                next(f) # skip header
                for line in f:
                    values = line.strip().split("\t")
                    node_name = process_node_name(values[0], self.assembly_type)
                    if int(values[1]) > 0:
                        contig_markers[node_name] = {g: 1 for g in values[5].split(",")}
                        ref_markers.update(set(contig_markers[node_name]))
                    if int(values[4]) > 0:
                        ref_markers.update(set(values[-1].split(",")))
        # create ref labels and ref markers
        self.labels += list(species_labels)
        self.label_to_node = {s: [] for s in self.labels}
        for n in self.node_to_label:
            s = self.node_to_label[n]
            self.label_to_node[s].append(n)
        assert len(ref_markers) == 120 # GTDB-tk bac and ar markers
        if reset:
            self.ref_marker_sets = [ref_markers]    
            self.contig_markers = contig_markers
            marker_counts = get_markers_to_contigs(self.ref_marker_sets, contig_markers)
            self.markers = marker_counts

    def get_edges_with_same_scgs(self):
        """Check every edge to see if nodes have SCGs in common

        :return: edges IDs (edges_src, edges_dst, edge_weight)
        :rtype: list
        """
        edge_ids_with_same_scgs = []
        scg_counter = Counter()
        for x, (i, j) in enumerate(zip(self.edges_src, self.edges_dst)):
            if self.node_names[i] in self.contig_markers and self.node_names[j] in self.contig_markers and \
                len(self.contig_markers[self.node_names[i]]) > 0 and len(self.contig_markers[self.node_names[j]]) > 0:
            
                overlap = len(self.contig_markers[self.node_names[i]].keys() & \
                    self.contig_markers[self.node_names[j]].keys())
                if overlap > 0 and i != j:
                    #remove edge
                    scg_counter[overlap] += 1
                    edge_ids_with_same_scgs.append(x)
        print("edges with overlapping scgs (max=20):", scg_counter.most_common(20))
        return edge_ids_with_same_scgs

    def generate_edges_based_on_labels(self, noise=0):
        # link nodes with same labels and create new adj_matrix/edges_src/edges_dst/edge_weights
        #new_edges = []
        new_srcs = []
        new_dsts = []
        for label in self.label_to_node:
            # make circular graph
            for i in range(len(self.label_to_node[label])):
                new_src = self.node_names.index(self.label_to_node[label][i])
                new_srcs.append(new_src)
                if i+1 < len(self.label_to_node[label]):
                    new_dst = self.node_names.index(self.label_to_node[label][i+1])
                else: # connect to element 0
                    new_dst = self.node_names.index(self.label_to_node[label][0])
                #new_edges.append((new_src, new_dst))
                new_dsts.append(new_dst)
        new_weights = [1] * len(new_srcs)
        print("creating new graph with {} edges".format(len(new_srcs)))
        self.adj_matrix = scipy.sparse.coo_matrix(
            (new_weights, (new_srcs, new_dsts)), shape=(len(self.node_names), len(self.node_names))
        )
        self.edges_src = new_srcs
        self.edges_dst = new_dsts
        self.edge_weights = np.array(new_weights)
        
    """
    def calculate_coverage_threshold(self, percentile=90):
        
        Calculate the coverage correlation threshold based on the distribution of coverage correlations.

        Args:
            percentile (int): The percentile to use for determining the threshold (default: 90).

        Returns:
            float: The coverage correlation threshold.
        
        # Ensure coverage data is available
        if not hasattr(self, 'node_depths') or len(self.node_depths) == 0:
            raise ValueError("Coverage data (node_depths) is not available in the dataset.")

        # Compute pairwise coverage correlations
        coverage = np.array(self.node_depths)
        num_nodes = coverage.shape[0]
        correlations = []

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                corr = np.corrcoef(coverage[i], coverage[j])[0, 1]
                if not np.isnan(corr):  # Ignore NaN values
                    correlations.append(corr)

        # Calculate the threshold based on the specified percentile
        threshold = np.percentile(correlations, percentile)
        print(f"Coverage correlation threshold (percentile={percentile}): {threshold:.4f}")
        self.coverage_threshold = threshold
        np.save(os.path.join(self.cache_dir, "coverage_threshold.npy"), self.coverage_threshold)
       """
 
def plot_and_save_loss(epoch_losses, iteration_name, value, save_dir):
    """
    Plot the loss over epochs and save the plot as an image file.

    Parameters:
    - epoch_losses (list): List of loss values for each epoch.
    - iteration_name (str): Name of the model iteration for labeling the plot and saving the file.
    - value: The value currently being evaluated, to be included in the plot title.
    - save_dir (str): Directory where the plot will be saved.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_losses, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss Curve for {iteration_name} = {value}')
    plt.legend()
    plt.grid(True)
    image_filename = os.path.join(save_dir, f"{iteration_name}_{value}_Loss.png")
    plt.savefig(image_filename)
    plt.close()
    print(f"Loss plot saved as {image_filename}")

def plot_and_save_loss_wval(train_losses, val_losses, iteration_name, value, save_dir):
    """
    Plot the train and validation loss over epochs and save the plot as an image file.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss Curve for {iteration_name} = {value}')
    plt.legend()
    plt.grid(True)
    image_filename = os.path.join(save_dir, f"{iteration_name}_{value}_Loss.png")
    plt.savefig(image_filename)
    plt.close()
    print(f"Loss plot saved as {image_filename}")

def get_last_iteration_number(results_filename):
    """
    Retrieve the last iteration number from the results Excel file.

    Parameters:
    - results_filename (str): The filename of the Excel file storing results.

    Returns:
    - int: The last iteration number. Returns 0 if no prior entries exist.
    """
    if os.path.exists(results_filename):
        results_df = pd.read_excel(results_filename)
        if not results_df.empty:
            last_name = results_df['Folder Name'].iloc[-1]
            if isinstance(last_name, str):  # Ensure it's a string
                try:
                    # Use regular expression to find the first integer in the string
                    match = re.search(r'\d+', last_name)
                    if match:
                        last_iteration = int(match.group())
                        return last_iteration
                    else:
                        return 0
                except (IndexError, ValueError):
                    return 0
            else:
                print(f"Warning: Last entry in 'Folder Name' is not a string: {last_name}")
                return 0
    return 0


def process_quality_reports(
    base_directory="E:/FCT/Tese/Code/checkm2/checkm2/model_results",
    excel_filename="hyperparameter_tuning_results.xlsx",
    recount_all=False
):
    # Load or create the Excel file
    excel_directory = os.path.join(base_directory,excel_filename)
    if os.path.exists(excel_directory):
        results_df = pd.read_excel(excel_directory)
    else:
        print(f"Excel file '{excel_filename}' not found.")
        return

    # Add columns for "Complete" and "Contaminated" if they do not exist
    if "Complete" not in results_df.columns:
        results_df["Complete"] = 0

    if "Contaminated" not in results_df.columns:
        results_df["Contaminated"] = 0
    
    # Iterate over each subdirectory in the base directory
    for folder_name in os.listdir(base_directory):
        if folder_name != excel_filename:
            folder_path = os.path.join(base_directory, folder_name, "checkm2_result")
            quality_report_path = os.path.join(folder_path, "quality_report.tsv")

            # Check if the quality_report.tsv file exists
            if not os.path.isfile(quality_report_path):
                print(f"quality_report.tsv not found in {folder_path}")
                continue

            # Find the row in the Excel file that matches the folder name
            row_index = results_df[results_df["Folder Name"] == folder_name].index
            print(f"COMPLETENESS VALUE BEFORE = {results_df.at[row_index[0], 'Complete']}")
            print(f"CONTAMINATION VALUE BEFORE = {results_df.at[row_index[0], 'Contaminated']}")

            # Check if we should recount everything or only empty columns
            if not results_df[results_df["Folder Name"] == folder_name].empty:
                if not recount_all and results_df.at[row_index[0], "Complete"] != 0 and results_df.at[row_index[0], "Contaminated"] != 0:
                    print(f"Skipping folder '{folder_name}' as it already has counts.")
                    continue

            # Read the quality_report.tsv file
            try:
                quality_report_df = pd.read_csv(quality_report_path, sep='\t')
            except Exception as e:
                print(f"Error reading {quality_report_path}: {e}")
                continue

            # Count the "Complete" and "Contaminated" rows
            complete_count = quality_report_df[quality_report_df['Completeness'] > 90].shape[0]
            contaminated_count = quality_report_df[quality_report_df['Contamination'] > 5].shape[0]

            # If matching row found, update the values
            if not row_index.empty:
                results_df.at[row_index[0], "Complete"] = complete_count
                results_df.at[row_index[0], "Contaminated"] = contaminated_count
                print(f"COMPLETENESS VALUE AFTER = {results_df.at[row_index[0], 'Complete']}")
                print(f"CONTAMINATION VALUE AFTER = {results_df.at[row_index[0], 'Contaminated']}")
            else:
                print(f"No matching folder name '{folder_name}' found in the Excel file.")

    # Save the updated Excel file
    results_df.to_excel(excel_directory, index=False)
    print(f"Updated Excel file saved as '{excel_directory}'")

def apply_pca_to_node_features(data, n_components=50):
    """
    Apply PCA to the node features (kmers) to reduce dimensionality.

    Parameters:
    - data: The PyTorch Geometric Data object containing the graph.
    - n_components: Number of principal components to retain.

    Returns:
    - data: The modified Data object with reduced-dimensionality node features.
    """
    # Convert node features to numpy array for PCA
    node_features = data.x.numpy()
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(node_features)
    
    # Convert reduced features back to torch tensor and update the data object
    data.x = torch.tensor(reduced_features, dtype=torch.float)
    
    return data

# Convert edge indices to sets of tuples for comparison
def get_undirected_edge_set(edge_index):
    # Ensure all edges are stored in a consistent order
    undirected_edges = {tuple(sorted(edge)) for edge in edge_index.t().tolist()}
    return undirected_edges

def compare_graphs(original_graph, altered_graph):
    changes = {'Isolated Nodes': 0, 'Added Edges': 0, 'Removed Edges': 0}

    # Convert edge indices to sets of tuples for comparison
    original_edges = get_undirected_edge_set(original_graph.edge_index)
    altered_edges = get_undirected_edge_set(altered_graph.edge_index)

    # Calculate added and removed edges
    added_edges = altered_edges - original_edges
    removed_edges = original_edges - altered_edges
    
    # Check the results
    print("Added edges:", len(added_edges))
    print("Removed edges:", len(removed_edges))
    
    changes['added'] = len(added_edges)
    changes['removed'] = len(removed_edges)

    # Check for isolated nodes in the altered graph (no connections)
    node_degrees = torch.zeros(altered_graph.num_nodes)
    for edge in altered_graph.edge_index.t():
        node_degrees[edge[0]] += 1
        node_degrees[edge[1]] += 1 if is_undirected(altered_graph.edge_index) else 0
    
    isolated_nodes = (node_degrees == 0).sum().item()
    changes['isolated'] = isolated_nodes

    return changes

def plot_pca_hq_clusters(dataset, data, embeddings, cluster_to_contig, output_path, iteration_name, value):
    """
    Applies PCA to reduce embeddings to 2D and generates a scatter plot coloured by high-quality (HQ) clusters.
    Nodes in HQ clusters are plotted with distinct colors and shapes, ensuring that each HQ cluster has a unique
    combination of color and shape. All other nodes are plotted in black. HQ clusters are plotted last to ensure
    they appear in the foreground.

    Parameters:
    - dataset: The original dataset containing node names.
    - data (torch_geometric.data.Data): The data object containing node indices.
    - embeddings (np.ndarray): The embeddings of the nodes.
    - cluster_to_contig (dict): Dictionary where each key is a cluster ID and each value is a set of node names.
    - output_path (str): The directory path where the plot will be saved.
    - iteration_name (str): A name used for the iteration to label the plot and file.
    - value (any): The value associated with the iteration, included in the plot title.
    - base_directory (str): Directory containing quality_report.tsv and graphmb_best_contig2bin.tsv.
    """
    # Map the node indices back to their original names
    mapped_node_names = [dataset.node_names[idx] for idx in data.node_names]

    # Perform 2D PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(embeddings)

    # Load quality report and filter high-quality clusters
    quality_report = pd.read_csv(os.path.join(Path(__file__).parent, dataset.name, 'quality_report.tsv'), sep='\t')
    hq_clusters = quality_report[(quality_report['Completeness'] > 90) & (quality_report['Contamination'] < 5)]
    hq_cluster_ids = set(hq_clusters['Name'])

    # Load graphmb_best_contig2bin file, skipping metadata rows and fixing headers
    contig2bin_file = os.path.join(Path(__file__).parent, dataset.name, 'graphmb_best_contig2bin.tsv')
    contig2bin = pd.read_csv(contig2bin_file, sep='\t', skiprows=3, names=['SEQUENCEID', 'BINID'])

    # Filter nodes in HQ clusters using graphmb_best_contig2bin.tsv
    hq_nodes = set()
    for cluster_id in hq_cluster_ids:
        cluster_nodes = contig2bin[contig2bin['BINID'] == cluster_id]['SEQUENCEID'].tolist()
        hq_nodes.update(cluster_nodes)

    # Ensure correct mapping from contig names to cluster IDs
    contig_to_cluster = {
        str(contig).strip().lower(): cluster_id
        for cluster_id, contigs in cluster_to_contig.items()
        for contig in contigs
    }

    # Get all node names
    node_names = [str(name).strip().lower() for name in mapped_node_names]

    # Get number of HQ clusters
    num_hq_clusters = len(hq_cluster_ids)

    # Create a larger figure
    plt.figure(figsize=(12, 10), dpi=300)

    # Use a qualitative colormap for HQ clusters
    if num_hq_clusters <= 20:
        cmap = plt.get_cmap("tab20", num_hq_clusters)
    elif num_hq_clusters <= 256:
        cmap = plt.get_cmap("turbo", num_hq_clusters)
    else:
        cmap = plt.get_cmap("nipy_spectral", num_hq_clusters)  # Works better for very large numbers of clusters

    # Define a list of marker shapes
    marker_shapes = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', 'x', '+', 'X', 'd', '|', '_']

    # Assign a unique combination of color and marker shape to each HQ cluster
    hq_cluster_styles = {}
    for i, cluster_id in enumerate(hq_cluster_ids):
        color = cmap(i)  # Assign a unique color from the colormap
        marker = marker_shapes[i % len(marker_shapes)]  # Assign a unique marker shape, cycling if necessary
        hq_cluster_styles[cluster_id] = {'color': color, 'marker': marker}

    # Separate HQ and non-HQ nodes
    hq_indices = []
    non_hq_indices = []
    for i, node_name in enumerate(node_names):
        if node_name in hq_nodes:
            cluster_id = contig_to_cluster.get(node_name, -1)
            if cluster_id in hq_cluster_styles:
                hq_indices.append(i)
        else:
            non_hq_indices.append(i)

    # Plot non-HQ clusters first (in the background)
    for i in non_hq_indices:
        plt.scatter(pca_result[i, 0], pca_result[i, 1], color='black', marker='o', s=8, alpha=0.6)

    # Plot HQ clusters last (in the foreground)
    for i in hq_indices:
        node_name = node_names[i]
        cluster_id = contig_to_cluster.get(node_name, -1)
        style = hq_cluster_styles[cluster_id]
        color = style['color']
        marker = style['marker']
        plt.scatter(pca_result[i, 0], pca_result[i, 1], color=color, marker=marker, s=8, alpha=0.6)

    # Create a proper colorbar for HQ clusters
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=num_hq_clusters - 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, label="HQ Cluster ID", ax=plt.gca())
    cbar.set_ticks(np.arange(num_hq_clusters))
    cbar.set_ticklabels(hq_cluster_ids)

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(f'2D PCA of {iteration_name} = {value}')
    plt.grid(True)

    # Save the plot
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    plt.savefig(os.path.join(output_path, f"{iteration_name}_scatterplot.png"), bbox_inches='tight')
    plt.close()
    print(f"✅ Plot saved to {os.path.join(output_path, f'{iteration_name}_scatterplot.png')}")
    
#from sklearn.manifold import TSNE


    
def plot_pca_embeddings_simple(data, embeddings, output_path, iteration_name, value):
    """
    Applies PCA to reduce embeddings to 2D and generates a scatter plot where all nodes fall within the visible area.

    Parameters:
    - data (torch_geometric.data.Data): The data object containing node names and other graph information.
    - embeddings (np.ndarray): The embeddings of the nodes.
    - output_path (str): The directory path where the plot will be saved.
    - iteration_name (str): A name used for the iteration to label the plot and file.
    - value (any): The value associated with the iteration, included in the plot title.
    """
    # Perform PCA on embeddings to reduce dimensions to 2
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)

    # Determine axis limits
    x_min, x_max = reduced_embeddings[:, 0].min() - 0.5, reduced_embeddings[:, 0].max() + 0.5
    y_min, y_max = reduced_embeddings[:, 1].min() - 0.5, reduced_embeddings[:, 1].max() + 0.5

    # Create the plot
    plt.figure(figsize=(12, 10))
    
    # Plot all points without any color distinction
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], s=10, alpha=0.7)
    
    # Set axis limits to ensure all points are visible
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    #ALPHA color values!!!s
    # Set plot title and labels
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title(f'2D PCA of {iteration_name} = {value}')
    plt.grid(True)
    
    # Save the plot to the specified output path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    plt.savefig(os.path.join(output_path, f"{iteration_name}_scatterplot.png"))
    plt.close()
    print(f"Plot saved to {os.path.join(output_path, f'{iteration_name}_scatterplot.png')}")
    
def completeness(reference_markers, genes):
    numerator = 0.0
    for marker_set in reference_markers:
        common = marker_set & genes
        if len(marker_set) > 0:
            numerator += len(common) / len(marker_set)
    return 100 * (numerator / len(reference_markers))


def contamination(reference_markers, genes):
    numerator = 0.0
    for i, marker_set in enumerate(reference_markers):
        inner_total = 0.0
        for gene in marker_set:
            if gene in genes and genes[gene] > 0:
                inner_total += genes[gene] - 1.0
        if len(marker_set) > 0:
            numerator += inner_total / len(marker_set)
    return 100.0 * (numerator / len(reference_markers))

    
def evaluate_contig_sets(marker_sets, contig_marker_counts, bin_to_contigs):
    """Calculate completeness and contamination for each bin given a set of
       marker gene sets and contig marker counts

    :param marker_sets: reference marker sets, from Bacteria
    :type marker_sets: list
    :param contig_marker_counts: Counts of each gene on each contig
    :type contig_marker_counts: dict
    :param bin_to_contigs: Mapping bins to contigs
    :type bin_to_contigs: dict
    """
    results = {}
    for bin in bin_to_contigs:
        bin_genes = {}
        for contig in bin_to_contigs[bin]:
            if contig not in contig_marker_counts:
                print("missing", contig)
                continue
            for gene in contig_marker_counts[contig]:
                if gene not in bin_genes:
                    bin_genes[gene] = 0
                bin_genes[gene] += contig_marker_counts[contig][gene]
        comp = completeness(marker_sets, set(bin_genes.keys()))
        cont = contamination(marker_sets, bin_genes)
        results[bin] = {"comp": comp, "cont": cont, "genes": bin_genes}
    return results
