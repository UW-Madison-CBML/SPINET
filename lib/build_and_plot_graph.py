import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sheaf_utils import build_graph
from motion_classifier_dataset import MotionClassifierDataset

import os

COLOR_RESIDUES = {res:plt.cm.tab20(i % 20) for i,res in enumerate(MotionClassifierDataset.AMINO_ACIDS)} # some residues will have the same colormaps
def plot_graph(positions:np.ndarray, colors, adjacency:np.ndarray, ax):
    """
    positions:np.ndarray, type= float, shape = (T,3)
    colors: list compatible with c arg in ax.scatter
    adjacency:np.ndarray, type=boolen, shape = (T,T), diag = 0
    fig, ax = plt.subplots(...)
    """
    ax.scatter(positions[:,0], positions[:,1], positions[:,2], c=colors, marker="o")
    line_segments = np.triu(adjacency)[:,:,None] * np.stack(np.broadcast_arrays(positions[None,:,:], positions[:,None,:]), axis=2)
    line_segments = line_segments.reshape(T*T,2,3)
    mask = (line_segments[:,0] == 0) & (line_segments[:,1] == 0)
    line_segments = line_segments[mask]
    line_segments = np.pad(line_segments, ((0,0), (0,1), (0,0)), mode="constant", constant_value=np.nan)
    line_segments = line_segments.reshape(-1, 3)
    ax.plot(line_segments[:,0], line_segments[:,1], line_segments[:,2])
    



def build_and_plot_graphs(residues, conformation1:np.ndarray, conformation2:np.ndarray, save_path):
    """
    residues: List of residue_names
    conformation1: np.ndarray, dtype=float, shape=(T,3)
    conformation2: np.ndarray, dtype=float, shape=(T,3)
    save_path: relative or absolute path to save plot to
    """
    graph_tensor = build_graph(torch.from_numpy(conformation1)[None,:,:], torch.from_numpy(conformation2)[None,:,:], torch.tensor([conformations1.size(0)]), 5, adjacency_matrix=True) 
    graph1, graph2 = graph_tensor.squeeze(0)
    fig, axes = plt.subplots(2, subplots_kw={"projection":"3d"})
    
     
    plot_graph(conformation1, [COLOR_RESIDUES[res] for res in residues], graph1, axes[0])
    plot_graph(conformation2, [COLOR_RESIDUES[res] for res in residues], graph2, axes[1])
    
    fig.savefig(save_path)
    plt.close(fig)
    

if __name__ == "__main__":
    os.makedirs("graph_plots",exist_ok=True)
    motions_df = pd.read_csv("motions.csv")

    fig, _ = plt.subplots()
    fig.legend(handles=legend, title="Phases", bbox_to_anchor=(1.4, 1.4))
    fig.savefig(os.path.join("graph_plots", "legend.svg"))
    plt.close(fig)


    for name, group in motions_df.groupby("motion_id"): 
        build_and_plot_graphs(group["res_name"].to_list(), group[["conf1_0","conf1_1","conf1_2"]].to_numpy(), group[["conf2_0","conf2_1","conf2_2"]], os.path.abspath(os.path.join("graph_plots", f"{name}.jpg")))
