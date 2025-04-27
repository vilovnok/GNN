# train.py

import numpy as np
from .GCN import GradDescentOptim  
from .utils import negative_log_likelihood

def train_gcn(gcn_model, adj_matrix, X, labels, train_nodes=None, 
            lr=2e-2, wd=2.5e-2, n_epochs=15000, early_stop_patience=50):
    

    if train_nodes is None:
        train_nodes = np.array([0, 1, 8])

    test_nodes = np.setdiff1d(np.arange(labels.shape[0]), train_nodes)
    optimizer = GradDescentOptim(lr=lr, wd=wd)

    embeddings = []
    accuracies = []
    train_losses = []
    test_losses = []

    best_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(1, n_epochs + 1):
        y_pred = gcn_model.forward(adj_matrix, X)

        optimizer(y_pred, labels, train_nodes)
        for layer in reversed(gcn_model.layers):
            layer.backward(optimizer, update=True)

        embeddings.append(gcn_model.embedding(adj_matrix, X))

        preds = np.argmax(y_pred, axis=1)
        true_labels = np.argmax(labels, axis=1)

        acc = (preds[test_nodes] == true_labels[test_nodes]).mean()
        accuracies.append(acc)

        losses = negative_log_likelihood(y_pred, labels)
        train_loss = losses[train_nodes].mean()
        test_loss = losses[test_nodes].mean()

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if test_loss < best_loss:
            best_loss = test_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter > early_stop_patience:
            print(f"Early stopping at epoch {epoch}!")
            break

        if epoch % 100 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {acc:.4f}")

    return {
        "embeddings": embeddings,
        "accuracies": accuracies,
        "train_losses": train_losses,
        "test_losses": test_losses
    }


# pos = {i: embeds[-1][i,:] for i in range(embeds[-1].shape[0])}
# _ = draw_kkl(g, None, colors, pos=pos, cmap='gist_rainbow', edge_color='gray')