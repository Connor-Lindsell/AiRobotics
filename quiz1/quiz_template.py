import argparse
from typing import Optional, Tuple

import numpy as np
import open3d as o3d
import polyscope as ps
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    f1_score, jaccard_score, roc_auc_score,
)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_ply_point_cloud(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load a point cloud from a .ply file.
    Returns points and binary ground truth labels derived from color:
        1 = ground / tarmac  (brown points)
        0 = non-ground features (buildings, aircraft, vehicles, etc.)
    """
    pcd = o3d.io.read_point_cloud(path)
    points = np.asarray(pcd.points)
    if points.size == 0:
        raise ValueError(f"No points found in {path}")

    labels = None
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        if colors.shape == points.shape:
            ground_color_ref = np.array([0.6, 0.4, 0.1])
            is_ground = np.all(np.isclose(colors, ground_color_ref, atol=1e-2), axis=1)
            labels = is_ground.astype(int)
            n_ground = labels.sum()
            print(f"  Ground truth: {n_ground} tarmac pts, "
                  f"{len(labels) - n_ground} feature pts")

    return points, labels


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
def perform_clustering(points: np.ndarray, k: int, z_scale: float = 5.0) -> np.ndarray:
    """
    Cluster the point cloud into k groups using K-Means on xyz coordinates,
    with z (height) scaled up before normalisation.

    Multiplying z by z_scale before StandardScaler amplifies the height signal
    so objects at different elevations (tarmac, walls, roofs, aircraft) end up
    in different clusters.

    Returns:
        cluster_labels: (N,) integer array in [0, k-1]
    """
    scaled_pts = points.copy()
    scaled_pts[:, 2] *= z_scale

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(scaled_pts)

    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    cluster_labels = km.fit_predict(features_scaled)

    for i in range(k):
        mask = cluster_labels == i
        mean_z = points[mask, 2].mean()
        print(f"  Cluster {i}: {mask.sum():>7} pts | mean elevation = {mean_z:.2f} m")

    return cluster_labels


# ---------------------------------------------------------------------------
# PCA per Cluster
# ---------------------------------------------------------------------------
def compute_cluster_pca(
    points: np.ndarray,
    cluster_labels: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    For each cluster fit a PCA on the raw xyz coordinates and collect:
      - cluster centre (mean position)
      - PC1/PC2/PC3 vectors scaled by std-dev (for Polyscope arrow visualisation)
      - per-point PCA projection coordinates (3 features) used as SVM input

    The PCA projection encodes where each point sits inside its cluster's
    principal-component coordinate frame. These are the "PCA features" fed
    to the SVM.

    Returns:
        centers      : (k, 3) cluster centroids in world coordinates
        pc1, pc2, pc3: (k, 3) principal axes scaled by sqrt(eigenvalue)
        pca_features : (N, 3) PCA projection for every point
    """
    centers = []
    pc1s, pc2s, pc3s = [], [], []
    pca_features = np.zeros((len(points), 3))

    for i in range(k):
        mask = cluster_labels == i
        pts  = points[mask]

        center = pts.mean(axis=0)
        centers.append(center)

        pca = PCA(n_components=3)
        pca.fit(pts)

        stds = np.sqrt(pca.explained_variance_)
        pc1s.append(pca.components_[0] * stds[0])
        pc2s.append(pca.components_[1] * stds[1])
        pc3s.append(pca.components_[2] * stds[2])

        pca_features[mask] = pca.transform(pts)

        print(f"  Cluster {i}: explained variance ratios = "
              f"{np.round(pca.explained_variance_ratio_, 3)}")

    return (
        np.array(centers),
        np.array(pc1s),
        np.array(pc2s),
        np.array(pc3s),
        pca_features,
    )


# ---------------------------------------------------------------------------
# Binary label derivation (fallback when PLY has no colours)
# ---------------------------------------------------------------------------
def derive_binary_labels(points: np.ndarray) -> np.ndarray:
    """
    If the PLY file carries no colour ground truth, derive binary labels
    from elevation: points at or below the median elevation are labelled
    tarmac (1); points above are labelled feature (0).
    """
    median_z = np.median(points[:, 2])
    labels = (points[:, 2] <= median_z).astype(int)
    n_ground = labels.sum()
    print(f"  Elevation fallback: {n_ground} tarmac pts (z ≤ {median_z:.2f} m), "
          f"{len(labels) - n_ground} feature pts")
    return labels


# ---------------------------------------------------------------------------
# SVM Classification
# ---------------------------------------------------------------------------
def train_svm(
    pca_features: np.ndarray,
    binary_labels: np.ndarray,
    train_sample_size: int = 50000,
) -> np.ndarray:
    """
    Train a binary RBF-SVM to classify each point as tarmac (1) or
    non-tarmac feature (0) using the 3 PCA projection coordinates as input.

    Why binary?
      The PCA projection features capture the shape of each point's cluster.
      Flat, sprawling tarmac clusters produce very different PCA projections
      (large PC1/PC2, near-zero PC3) compared to compact structured clusters
      (buildings, aircraft) which have more balanced PC1/PC2/PC3 spread.
      A binary SVM can exploit this to separate tarmac from features — this
      is the semantic "cluster label" the pipeline predicts.

    Evaluation (Tutorial 1 + Tutorial 3 methods):
      - 10-fold stratified cross-validation accuracy
      - Accuracy, F1, IoU / Jaccard scores
      - 2×2 confusion matrix (TP, FP, FN, TN)
      - Per-class classification report
      - ROC AUC score

    SVC(rbf) scales O(n²) so a stratified subsample is used for training
    and CV; predictions are then run on the full dataset for visualisation.

    Returns:
        svm_predictions: (N,) binary predictions on the full point cloud
    """
    scaler = StandardScaler()
    X_all = scaler.fit_transform(pca_features)   # normalise PCA projections
    y_all = binary_labels

    # Stratified subsample — keeps class proportions intact
    n_total     = len(y_all)
    sample_size = min(train_sample_size, n_total)
    print(f"Subsampling {sample_size} / {n_total} points for SVM training "
          f"(stratified: tarmac vs. feature)")

    _, X_train, _, y_train = train_test_split(
        X_all, y_all,
        test_size=sample_size / n_total,
        stratify=y_all,
        random_state=42,
    )

    svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True,
              random_state=42)

    # --- 10-fold stratified cross-validation (Tutorial 3 method) ---
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    print("Running 10-fold cross-validation...")
    cv_scores = cross_val_score(svm, X_train, y_train, cv=cv, scoring='accuracy')
    print(f"10-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"Per-fold scores:     {np.round(cv_scores, 4)}")

    # --- Train on full subsample and report Tutorial 1 metrics ---
    svm.fit(X_train, y_train)
    train_preds  = svm.predict(X_train)
    train_proba  = svm.predict_proba(X_train)[:, 1]

    acc = accuracy_score(y_train, train_preds)
    f1  = f1_score(y_train, train_preds)
    iou = jaccard_score(y_train, train_preds)
    auc = roc_auc_score(y_train, train_proba)

    print(f"\n--- Tutorial 1 Metrics (on training subsample) ---")
    print(f"  Accuracy  (TP+TN / total)             : {acc:.4f}")
    print(f"  F1 score  (harmonic mean prec/recall) : {f1:.4f}")
    print(f"  IoU/Jaccard (TP / TP+FP+FN)           : {iou:.4f}")
    print(f"  ROC AUC                                : {auc:.4f}")

    cm = confusion_matrix(y_train, train_preds)
    tn, fp, fn, tp = cm.ravel()
    print(f"\nConfusion matrix:")
    print(f"               Predicted tarmac  Predicted feature")
    print(f"  Actual tarmac      {tp:>6}              {fn:>6}")
    print(f"  Actual feature     {fp:>6}              {tn:>6}")
    print(f"\nClassification report:")
    print(classification_report(y_train, train_preds,
                                target_names=["Feature", "Tarmac"]))

    # --- Predict on the full point cloud for visualisation ---
    print("Predicting on full point cloud...")
    svm_predictions = svm.predict(X_all)

    full_acc = accuracy_score(y_all, svm_predictions)
    print(f"Full-dataset accuracy: {full_acc:.4f}")

    return svm_predictions


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def visualize(
    points: np.ndarray,
    gt_labels: Optional[np.ndarray],
    cluster_labels: np.ndarray,
    pca_centers: np.ndarray,
    pca_pc1: np.ndarray,
    pca_pc2: np.ndarray,
    pca_pc3: np.ndarray,
    svm_predictions: np.ndarray,
) -> None:
    """
    Polyscope visualisation (toggle quantities in the GUI):
      1. Elevation colourmap (enabled by default)
      2. Ground truth binary labels (tarmac=1, feature=0)
      3. K-Means cluster colours
      4. PCA principal component arrows at each cluster centre
         Red = PC1 (major spread), Green = PC2, Blue = PC3 (normal)
      5. SVM binary predictions — tarmac vs. non-tarmac features
    """
    print("Visualizing results in Polyscope...")
    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("none")

    cloud = ps.register_point_cloud("Processed Point Cloud", points, radius=0.0015)
    cloud.set_point_render_mode("quad")
    cloud.add_scalar_quantity("Elevation", points[:, 2], enabled=True)

    # Ground truth binary labels
    if gt_labels is not None:
        cloud.add_scalar_quantity("Ground Truth (1=tarmac, 0=feature)",
                                  gt_labels.astype(float), enabled=False)
        gt_colors = np.where(gt_labels[:, None], [0.4, 0.8, 0.4], [0.8, 0.3, 0.3])
        cloud.add_color_quantity("Ground Truth (color)", gt_colors, enabled=False)

    # K-Means cluster colours
    cmap = plt.get_cmap('tab10')
    cluster_colors = np.array([cmap(int(lbl) % 10)[:3] for lbl in cluster_labels])
    cloud.add_color_quantity("K-Means Clusters", cluster_colors, enabled=False)
    cloud.add_scalar_quantity("K-Means Cluster Labels",
                              cluster_labels.astype(float), enabled=False)

    # PCA arrows at cluster centres
    pca_cloud = ps.register_point_cloud("Cluster PCA Centers", pca_centers, radius=0.005)
    pca_cloud.add_vector_quantity("PC1 (Major)",  pca_pc1, enabled=True,
                                  color=(1, 0, 0), vectortype="ambient")
    pca_cloud.add_vector_quantity("PC2 (Minor)",  pca_pc2, enabled=True,
                                  color=(0, 1, 0), vectortype="ambient")
    pca_cloud.add_vector_quantity("PC3 (Normal)", pca_pc3, enabled=True,
                                  color=(0, 0, 1), vectortype="ambient")

    # SVM binary predictions: green = tarmac, red = feature
    svm_colors = np.where(svm_predictions[:, None], [0.4, 0.8, 0.4], [0.8, 0.3, 0.3])
    cloud.add_color_quantity("SVM: Tarmac (green) vs Feature (red)",
                             svm_colors, enabled=False)
    cloud.add_scalar_quantity("SVM Predictions (1=tarmac, 0=feature)",
                              svm_predictions.astype(float), enabled=False)

    ps.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Point Cloud Segmentation pipeline (K-Means → PCA → SVM)"
    )
    parser.add_argument("path", nargs="?", default="airport_downsample.ply")
    parser.add_argument("-k", "--clusters", type=int, default=100,
                        help="Number of K-Means clusters (default: 100)")
    parser.add_argument("--z-scale", type=float, default=5.0,
                        help="Height amplification factor before K-Means (default: 5.0)")
    args = parser.parse_args()

    # 1. Load — extracts ground truth binary labels from PLY colours
    print(f"Loading point cloud from {args.path} ...")
    points, gt_labels = load_ply_point_cloud(args.path)
    print(f"Loaded {len(points)} points")

    # 2. K-Means clustering on z-scaled xyz
    print(f"\n--- K-Means Clustering (k={args.clusters}, z_scale={args.z_scale}) ---")
    cluster_labels = perform_clustering(points, args.clusters, z_scale=args.z_scale)
    # Uncomment to cache and skip slow clustering on re-runs:
    # np.save("cluster_labels.npy", cluster_labels)
    # cluster_labels = np.load("cluster_labels.npy")

    # 3. PCA per cluster — produces per-point PCA projection features
    print("\n--- PCA per Cluster ---")
    pca_centers, pca_pc1, pca_pc2, pca_pc3, pca_features = compute_cluster_pca(
        points, cluster_labels, args.clusters
    )

    # 4. Binary labels for SVM: tarmac (1) vs. non-tarmac features (0)
    #    Use PLY colour ground truth if available, else derive from elevation
    print("\n--- Binary Labels for SVM ---")
    if gt_labels is not None:
        binary_labels = gt_labels
        print("  Using PLY colour ground truth (brown = tarmac)")
    else:
        binary_labels = derive_binary_labels(points)

    # 5. SVM: binary classification (tarmac vs. feature) from PCA features
    print("\n--- SVM Classification ---")
    svm_predictions = train_svm(pca_features, binary_labels)

    # 6. Visualise
    visualize(
        points, gt_labels,
        cluster_labels,
        pca_centers, pca_pc1, pca_pc2, pca_pc3,
        svm_predictions,
    )


if __name__ == "__main__":
    main()
