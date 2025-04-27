import numpy as np

from models.gcn.GCN import GCN
from models.gcn.utils import load_data
from models.gcn.trainer import train_gcn

from utils import parse_args, logger, video




if __name__ == "__main__":
    args = parse_args()
    g, n_classes, labels, adj_matrix, X, colors = load_data()

    if args.model == 'gcn':
        model = GCN(
                in_channels=g.number_of_nodes(),
                out_channels=n_classes,
                n_layers=2,
                hidden_sizes=[16, 2], 
                activation=np.tanh,
                seed=100,
            )
        embeddings, _, _, _ = train_gcn(model, adj_matrix, X, labels)
        logger(args.model.upper())

    #TODO: The task is in progress
    # elif args.model == 'gat':
    #     model = GAT()
    #     print("Using model: GAT")
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    
    ###
    # Make sure that you have completed the command > conda install ffmpeg
    ###
    video(g, embeddings, colors)




