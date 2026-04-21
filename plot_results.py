"""
Fonctions de visualisation reutilisables par le notebook.
Chaque fonction accepte soit des donnees passees en argument,
soit charge depuis data/ si aucun argument n'est fourni.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm

from src.nn.model import VolatilityNet
from src.nn.training import get_device, predict_denormalized


def _load_from_disk(model=None, X_norm=None, Y_raw=None, y_mean=None, y_std=None):
    """Charge depuis data/ seulement ce qui manque."""
    if all(v is not None for v in (model, X_norm, Y_raw, y_mean, y_std)):
        return model, X_norm, Y_raw, y_mean, y_std

    device = get_device()
    if model is None:
        model = VolatilityNet().to(device)
        model.load_state_dict(torch.load("data/model_weights.pth", map_location=device))
        model.eval()
    if X_norm is None:
        X_norm = np.load("data/X_params_norm.npy").astype(np.float32)
    if Y_raw is None:
        Y_raw = np.load("data/Y_vols_raw.npy").astype(np.float32)
    if y_mean is None or y_std is None:
        stats = np.load("data/norm_stats.npz")
        y_mean = stats["y_mean"]
        y_std = stats["y_std"]
    return model, X_norm, Y_raw, y_mean, y_std


def _get_default_grid(maturities, strikes):
    """Renvoie la grille du papier si rien n'est fourni."""
    if maturities is not None and strikes is not None:
        return np.asarray(maturities), np.asarray(strikes)
    from src.model.rough_bergomi import RoughBergomi
    m = RoughBergomi()
    return (m.maturities if maturities is None else np.asarray(maturities),
            m.strikes if strikes is None else np.asarray(strikes))


# ---------------------------------------------------------------
# Plots
# ---------------------------------------------------------------

def plot_loss_history(history=None, save_path=None, ax=None):
    """Courbes train / val loss."""
    if history is None:
        h = np.load("data/training_history.npz")
        history = {"train": h["train"], "val": h["val"]}

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["train"], label="train")
    ax.plot(history["val"], label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE (normalized)")
    ax.set_yscale("log")
    ax.set_title("Training history")
    ax.legend()
    if standalone:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        plt.show()


def plot_error_heatmaps(model=None, X_norm=None, Y_raw=None,
                         y_mean=None, y_std=None,
                         maturities=None, strikes=None,
                         title_suffix="", save_path=None):
    """Trois heatmaps (avg, std, max) de l'erreur relative (Figure 6 du papier)."""
    model, X_norm, Y_raw, y_mean, y_std = _load_from_disk(
        model, X_norm, Y_raw, y_mean, y_std
    )
    maturities, strikes = _get_default_grid(maturities, strikes)
    n_T, n_K = len(maturities), len(strikes)

    Y_pred = predict_denormalized(model, X_norm, y_mean, y_std)
 


    mask = Y_raw > 1e-3
    rel_err = np.where(mask,
                    np.abs(Y_pred - Y_raw) / Y_raw,
                    np.nan)

    avg = np.nanmean(rel_err, axis=0).reshape(n_T, n_K)
    std = np.nanstd(rel_err, axis=0).reshape(n_T, n_K)
    maxe = np.nanmax(rel_err, axis=0).reshape(n_T, n_K)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    titles = ["Average rel err (%)", "Std rel err (%)", "Max rel err (%)"]
    for ax, M, t in zip(axes, [avg*100, std*100, maxe*100], titles):
        im = ax.imshow(M, aspect="auto", cmap="viridis", origin="lower")
        ax.set_xticks(range(n_K))
        ax.set_xticklabels([f"{k:.2f}" for k in strikes],
                           rotation=45 if n_K > 9 else 0)
        ax.set_yticks(range(n_T))
        ax.set_yticklabels([f"{t_:.2f}" for t_ in maturities])
        ax.set_xlabel("Strike")
        ax.set_ylabel("Maturity")
        ax.set_title(t)
        plt.colorbar(im, ax=ax)

    if title_suffix:
        fig.suptitle(f"NN vs MC, {title_suffix}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

    return {"rel_err": rel_err, "avg": avg, "std": std, "max": maxe}


def plot_smiles(model=None, X_norm=None, Y_raw=None,
                y_mean=None, y_std=None,
                maturities=None, strikes=None,
                n_samples=4, mat_idx=1, seed=42, save_path=None):
    """Compare des smiles predits vs MC sur quelques echantillons au hasard."""
    model, X_norm, Y_raw, y_mean, y_std = _load_from_disk(
        model, X_norm, Y_raw, y_mean, y_std
    )
    maturities, strikes = _get_default_grid(maturities, strikes)
    n_T, n_K = len(maturities), len(strikes)

    Y_pred = predict_denormalized(model, X_norm, y_mean, y_std)

    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(X_norm), n_samples, replace=False)

    cols = 2
    rows = (n_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(11, 3.3 * rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, idx in zip(axes, idxs):
        true_smile = Y_raw[idx].reshape(n_T, n_K)[mat_idx]
        pred_smile = Y_pred[idx].reshape(n_T, n_K)[mat_idx]
        ax.plot(strikes, true_smile, "ko", label="Monte Carlo", markersize=5)
        ax.plot(strikes, pred_smile, "r-", label="Neural network")
        ax.set_xlabel("Strike")
        ax.set_ylabel("Implied vol")
        ax.set_title(f"Sample {idx}, T = {maturities[mat_idx]:.2f}")
        ax.legend()

    for ax in axes[n_samples:]:
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_surface_3d(idx=0, model=None, X_norm=None, Y_raw=None,
                     y_mean=None, y_std=None,
                     maturities=None, strikes=None, save_path=None):
    """Surface 3D cote-a-cote : MC vs NN."""
    model, X_norm, Y_raw, y_mean, y_std = _load_from_disk(
        model, X_norm, Y_raw, y_mean, y_std
    )
    maturities, strikes = _get_default_grid(maturities, strikes)
    n_T, n_K = len(maturities), len(strikes)

    Y_pred = predict_denormalized(model, X_norm[idx:idx+1], y_mean, y_std)
    true_surf = Y_raw[idx].reshape(n_T, n_K)
    pred_surf = Y_pred[0].reshape(n_T, n_K)

    K, T = np.meshgrid(strikes, maturities)
    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot_surface(K, T, true_surf, cmap=cm.viridis,
                      alpha=0.85, edgecolor="k", lw=0.3)
    ax1.set_title("Monte Carlo")
    ax1.set_xlabel("K"); ax1.set_ylabel("T"); ax1.set_zlabel("sigma")

    ax2 = fig.add_subplot(122, projection="3d")
    ax2.plot_surface(K, T, pred_surf, cmap=cm.viridis,
                      alpha=0.85, edgecolor="k", lw=0.3)
    ax2.set_title("Neural network")
    ax2.set_xlabel("K"); ax2.set_ylabel("T"); ax2.set_zlabel("sigma")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
