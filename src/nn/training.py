"""
Boucle d'entrainement reutilisable pour le VolatilityNet.
Appelee a la fois par train.py (script CLI) et par le notebook
(pour les experiences 6 et 7).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from src.nn.model import VolatilityNet


def get_device():
    """Retourne le meilleur device disponible : MPS (Mac), CUDA, sinon CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def masked_mse(pred, target):
    mask = ~torch.isnan(target)
    target_clean = torch.where(mask, target, torch.zeros_like(target))
    diff = (pred - target_clean)**2
    # Extra safety : on zero-out les positions qui seraient NaN dans pred aussi
    diff = torch.where(torch.isnan(diff), torch.zeros_like(diff), diff)
    return (diff * mask).sum() / mask.sum().clamp(min=1)

def train_model(X_norm, Y_norm, epochs=200, patience=25, batch=32, lr=1e-3,
                val_split=0.15, seed=42, device=None, verbose=True,
                log_every=10):
    """
    Entraine un VolatilityNet avec early stopping sur la loss de validation.

    Parameters
    ----------
    X_norm : ndarray, shape (n, input_dim)
        Parametres normalises.
    Y_norm : ndarray, shape (n, output_dim)
        Surfaces normalisees.
    epochs : int
    patience : int
        Nombre d'epochs sans amelioration avant l'arret.
    batch : int
    lr : float
    val_split : float
    seed : int
    device : torch.device, optional
    verbose : bool
    log_every : int

    Returns
    -------
    net : VolatilityNet
        Le reseau avec les meilleurs poids (pas les derniers).
    history : dict
        {'train': [...], 'val': [...]}
    """
    if device is None:
        device = get_device()

    X = torch.from_numpy(X_norm.astype(np.float32))
    Y = torch.from_numpy(Y_norm.astype(np.float32))

    n_val = int(val_split * len(X))
    n_train = len(X) - n_val

    dataset = TensorDataset(X, Y)
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch)

    net = VolatilityNet(input_size=X.shape[1], output_size=Y.shape[1]).to(device)
    opt = optim.Adam(net.parameters(), lr=lr)
    

    best_val = float('inf')
    best_state = None
    no_improve = 0
    history = {'train': [], 'val': []}

    for ep in range(epochs):
        # Train
        net.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = masked_mse(net(xb), yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= n_train

        # Val
        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += masked_mse(net(xb), yb).item() * xb.size(0)
        val_loss /= n_val

        history['train'].append(tr_loss)
        history['val'].append(val_loss)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if verbose and (ep + 1) % log_every == 0:
            print(f"Epoch {ep+1:3d}  train={tr_loss:.5f}  val={val_loss:.5f}")

        if no_improve >= patience:
            if verbose:
                print(f"Early stop at epoch {ep+1}")
            break

    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return net, history


def predict_denormalized(net, X_norm, y_mean, y_std, device=None, batch=512):
    """
    Predit et denormalise les sorties du reseau.

    Parameters
    ----------
    net : VolatilityNet
    X_norm : ndarray
    y_mean, y_std : ndarray ou scalaire
    device : torch.device, optional
    batch : int

    Returns
    -------
    Y_pred : ndarray, meme shape que les labels attendus
    """
    if device is None:
        device = next(net.parameters()).device

    X = torch.from_numpy(X_norm.astype(np.float32)).to(device)
    net.eval()

    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            preds.append(net(X[i:i+batch]).cpu().numpy())

    Y_pred_norm = np.concatenate(preds, axis=0)
    return Y_pred_norm * y_std + y_mean

