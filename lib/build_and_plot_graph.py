import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sheaf_utils import build_graph

def plot_graph(positions:np.ndarray, colors, adjacency:np.ndarray, fig, ax):
    """
    positions:np.ndarray, type= float, shape = (T,3)
    colors: list compatible with c arg in ax.scatter
    adjacency:np.ndarray, type=boolen, shape = (T,T), diag = 0
    fig, ax = plt.subplots(...)
    """
    ax.scatter(positions[:,0], positions[:,1], positions[:,2], c=colors, marker="o")
    line_segments = np.triu(adjacency)[:,:,None] * np.stack(np.broadcast_arrays(positions[None,:,:], positions[:,None,:]), axis=2) 
    line_segments = line_segments.reshape(T*T,2,3)
    
    

def build_and_plot_graphs(residues:np.ndarray, conformation1:np.ndarray, conformation2:np.ndarray, save_path):
    """
    residues: np.ndarray, dtype = long or int
    conformation1: np.ndarray, dtype=float, shape=(T,3) 
    conformation2: np.ndarray, dtype=float, shape=(T,3) 
    save_path: relative or absolute path to save plot to
    """
    


    
