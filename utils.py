import argparse

import numpy as np

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import animation



def parse_args():
    parser = argparse.ArgumentParser(description="Train GNN models")
    parser.add_argument('--model', type=str, choices=['gcn', 'gat'], default='gcn',
                        help="Choose which model to train: 'gcn' or 'gat'")
    return parser.parse_args()

def logger(model_name: str):
    print("\n" + "=" * 80)
    print(f"🚀🚀🚀 MODEL TRAINING COMPLETE: {model_name} 🚀🚀🚀".center(80))
    print("=" * 80 + "\n")

def video(g, embeddings, colors):
    N = 500
    snapshots = np.linspace(0, len(embeddings)-1, N).astype(int)

    fig, ax = plt.subplots(figsize=(10, 10))
    kwargs = {'cmap': 'gist_rainbow', 'edge_color': 'gray', }

    def update(idx):
        ax.clear()
        embed = embeddings[snapshots[idx]]
        pos = {i: embed[i,:] for i in range(embed.shape[0])}
        nx.draw(g, pos, node_color=colors, ax=ax, **kwargs)
    
    try:
        anim = animation.FuncAnimation(fig, update, frames=snapshots.shape[0], interval=10, repeat=False)
        anim.save('embed_anim.mp4', dpi=300)
    except Exception as err:
        print(err)