import numpy as np
from .utils import glorot_init


class GradDescentOptim():
    def __init__(self, lr, wd):
        self.lr = lr
        self.wd = wd
        self._y_pred = None
        self._y_true = None
        self._out = None
        self.bs = None
        self.train_nodes = None
        
    def __call__(self, y_pred, y_true, train_nodes=None):
        self.y_pred = y_pred
        self.y_true = y_true
        
        if train_nodes is None:
            self.train_nodes = np.arange(y_pred.shape[0])
        else:
            self.train_nodes = train_nodes
            
        self.bs = self.train_nodes.shape[0]
        
    @property
    def out(self):
        return self._out
    
    @out.setter
    def out(self, y):
        self._out = y


class GCNLayer:
    def __init__(self, in_channels, out_channels, activation=None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = glorot_init(out_channels, in_channels)
        self.activation = activation

    def normalize_adj(self, adj_matrix):
        """
            Normialize adjacency_matrix -> D^{-1/2} A D^{-1/2}
        """
        
        adj_matrix = adj_matrix + np.eye(adj_matrix.shape[0])  # add self.attention
        D = np.sum(adj_matrix, axis=1)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(D))
        return D_inv_sqrt @ adj_matrix @ D_inv_sqrt


    def forward(self, adj_matrix, node_feats, W=None):
        """        
        Inputs:

            adj_matrix (adjacency matrix): Batch adjacency matrix of the graph 
                shape: [batch_size, num_nodes, num_nodes] 

            node_feats (node features): Matrix with node features of 
                shape[batch_size, num_nodes, in_channels]
        """

        self._adj_matrix = self.normalize_adj(adj_matrix)
        self._node_feats = (self.normalize_adj(adj_matrix) @ node_feats).T
        
        if W is None:
            W = self.W

        H = W @ self._node_feats
        if self.activation is not None:
            H = self.activation(H)
        self._H = H
        return self._H.T 

    def backward(self, optim, update=True):
        dtanh = 1 - np.asanyarray(self._H.T)**2
        d2 = np.multiply(optim.out, dtanh)

        self.grad = self._adj_matrix @ d2 @ self.W
        optim.out = self.grad
        
        dW = np.asarray(d2.T @ self._node_feats.T) / optim.bs
        dW_wd = self.W * optim.wd / optim.bs
        
        if update:
            self.W -= (dW + dW_wd) * optim.lr 
        
        return dW + dW_wd


class SoftmaxLayer:
    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = glorot_init(out_channels, in_channels)
        self.b = np.zeros((out_channels, 1))
        self._node_feats = None

    def shift(self, proj):
        shiftx = proj - np.max(proj, axis=0, keepdims=True)
        exps = np.exp(shiftx)
        return exps / np.sum(exps, axis=0, keepdims=True)

    def forward(self, node_feats, W=None, b=None):
        self._node_feats = node_feats.T
        if W is None:
            W = self.W
        if b is None:
            b = self.b
        
        proj = np.asarray(W @ self._node_feats) + b
        return self.shift(proj).T
    
    def backward(self, optim, update=True):
        train_mask = np.zeros(optim.y_pred.shape[0])
        train_mask[optim.train_nodes] = 1
        train_mask = train_mask.reshape((-1, 1))
        
        d1 = np.asarray((optim.y_pred - optim.y_true))
        d1 = np.multiply(d1, train_mask)
        
        self.grad = d1 @ self.W
        optim.out = self.grad
        
        dW = (d1.T @ self._node_feats.T) / optim.bs 
        db = d1.T.sum(axis=1, keepdims=True) / optim.bs
                
        dW_wd = self.W * optim.wd / optim.bs
        
        if update:   
            self.W -= (dW + dW_wd) * optim.lr
            self.b -= db.reshape(self.b.shape) * optim.lr
        
        return dW + dW_wd, db.reshape(self.b.shape)


class GCN:
    def __init__(self, in_channels, out_channels, n_layers, hidden_sizes, activation, seed=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        
        np.random.seed(seed)
        
        self.layers = list()
        gcn_in = GCNLayer(in_channels, hidden_sizes[0], activation)
        self.layers.append(gcn_in)
        
        for layer in range(n_layers):
            gcn = GCNLayer(self.layers[-1].W.shape[0], hidden_sizes[layer], activation)
            self.layers.append(gcn)
            
        sm_out = SoftmaxLayer(hidden_sizes[-1], out_channels)
        self.layers.append(sm_out)
    
    def embedding(self, adj_matrix, node_feats):
        H = node_feats
        for layer in self.layers[:-1]:
            H = layer.forward(adj_matrix, H)
        return np.asarray(H)
    
    def forward(self, adj_matrix, node_feats):
        H = self.embedding(adj_matrix, node_feats)
        p = self.layers[-1].forward(H)
        return np.asarray(p)