import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sheaf_utils import build_graph
from motion_classifier_dataset import MotionClassifierDataset
from matplotlib.patches import Patch
import torch
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import os

COLOR_RESIDUES = {res.lower():plt.cm.tab20(i % 20) for i,res in enumerate(MotionClassifierDataset.AMINO_ACIDS)} # some residues will have the same colormaps
def plot_graph(positions:np.ndarray, colors, adjacency:np.ndarray, ax):
    """
    positions:np.ndarray, type= float, shape = (N,3)
    colors: list compatible with c arg in ax.scatter
    adjacency:np.ndarray, type=boolen, shape = (N,N), diag = 0
    fig, ax = plt.subplots(...)
    """
    N = positions.shape[0] 
    ax.scatter(positions[:,0], positions[:,1], positions[:,2], c=colors, marker="o")
    line_segments = np.triu(adjacency)[:,:,None,None] * np.stack(np.broadcast_arrays(positions[None,:,:], positions[:,None,:]), axis=2)
    line_segments = line_segments.reshape(N*N,2,3)
    edge_mask = np.triu(adjacency).flatten()
    
    segments = line_segments[edge_mask]
    edge_collection = Line3DCollection(segments, colors='gray', linewidths=1, alpha=0.6)
    ax.add_collection3d(edge_collection)
    



def build_and_plot_graphs(residues, conformation1:np.ndarray, conformation2:np.ndarray, save_path):
    """
    residues: List of residue_names
    conformation1: np.ndarray, dtype=float, shape=(N,3)
    conformation2: np.ndarray, dtype=float, shape=(N,3)
    save_path: relative or absolute path to save plot to
    """
    graph_tensor = build_graph(torch.from_numpy(conformation1)[None,:,:], torch.from_numpy(conformation2)[None,:,:], torch.tensor([conformation1.shape[0]]), 5, adjacency_matrix=True) 
    graph1, graph2 = graph_tensor.squeeze(0).numpy()
    fig, axes = plt.subplots(1,2, figsize = (20,10), subplot_kw={"projection":"3d"})
    
     
    plot_graph(conformation1, [COLOR_RESIDUES[res.lower()] for res in residues], graph1, axes[0])
    plot_graph(conformation2, [COLOR_RESIDUES[res.lower()] for res in residues], graph2, axes[1])
    
    fig.savefig(save_path)
    plt.close(fig)
    

if __name__ == "__main__":
    os.makedirs("graph_plots",exist_ok=True)
    motions_df = pd.read_csv("motions.csv")

    legend = [Patch(color=col, label=res) for res, col in COLOR_RESIDUES.items()]

    fig, _ = plt.subplots()

    leg = fig.legend(handles=legend, title="Phases", bbox_to_anchor=(1.4, 1.4))

    fig.canvas.draw()

    bbox = leg.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

    fig.savefig(os.path.join("graph_plots", "legend.svg"), format='svg', bbox_inches=bbox, pad_inches=0.05)

    plt.close(fig)

    for name, group in motions_df.groupby("motion_id"): 
        build_and_plot_graphs(group["res_name"].to_list(), group[["conf1_0","conf1_1","conf1_2"]].to_numpy(), group[["conf2_0","conf2_1","conf2_2"]].to_numpy(), os.path.abspath(os.path.join("graph_plots", f"{name}.png")))
