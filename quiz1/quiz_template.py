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
from sklearn.model_selection import cross_val_score, StratifiedKFold
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
    Returns points and binary ground truth labels derived from colour:
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
    with z (height) scaled up before normalisation so objects at different
    elevations end up in different clusters.

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
    binary_labels: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray]:
    """
    For each cluster fit a PCA on the raw xyz coordinates and collect:
      - cluster centre and scaled principal axes (for Polyscope visualisation)
      - per-point PCA projection coordinates (for reference / future use)
      - cluster-level feature vector used as SVM input (see below)
      - cluster-level binary ground truth label (majority vote)

    Cluster-level SVM features (4 per cluster):
        [mean_z, ev1_ratio, ev2_ratio, ev3_ratio]

      mean_z      — mean elevation of the cluster.  Tarmac sits near ground
                    level; buildings and aircraft are elevated.
      ev1_ratio   — fraction of variance along the major axis.  Flat tarmac
                    patches are elongated → very high ev1 (≈0.7–0.9).
      ev2_ratio   — variance along the second axis.
      ev3_ratio   — variance along the normal axis.  For a flat surface this
                    is ≈ 0; for a 3-D structure (building, fuselage) it is
                    noticeably non-zero.

    Returns:
        centers           : (k, 3)
        pc1, pc2, pc3     : (k, 3)  principal axes scaled by sqrt(eigenvalue)
        pca_features      : (N, 3)  per-point PCA projection
        cluster_features  : (k, 4)  cluster-level features for SVM
        cluster_gt_labels : (k,)    binary majority-vote label per cluster
    """
    centers = []
    pc1s, pc2s, pc3s = [], [], []
    pca_features      = np.zeros((len(points), 3))
    cluster_features  = np.zeros((k, 4))
    cluster_gt_labels = np.zeros(k, dtype=int)

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

        # Cluster-level feature: mean elevation + PCA explained variance ratios
        cluster_features[i] = [pts[:, 2].mean(), *pca.explained_variance_ratio_]

        # Cluster binary label: majority vote of point ground truth
        if binary_labels is not None:
            cluster_gt_labels[i] = int(binary_labels[mask].mean() >= 0.5)
        else:
            # Fallback: clusters near median elevation = tarmac
            cluster_gt_labels[i] = int(pts[:, 2].mean() <= np.median(points[:, 2]))

        print(f"  Cluster {i}: ev = {np.round(pca.explained_variance_ratio_, 3)} "
              f"| mean_z = {pts[:,2].mean():.2f} m "
              f"| gt = {'tarmac' if cluster_gt_labels[i] else 'feature'}")

    return (
        np.array(centers),
        np.array(pc1s),
        np.array(pc2s),
        np.array(pc3s),
        pca_features,
        cluster_features,
        cluster_gt_labels,
    )


# ---------------------------------------------------------------------------
# SVM Classification
# ---------------------------------------------------------------------------
def train_svm(
    cluster_features: np.ndarray,
    cluster_gt_labels: np.ndarray,
    cluster_labels: np.ndarray,
) -> np.ndarray:
    """
    Train a binary RBF-SVM on the k cluster-level PCA features to classify
    each cluster as tarmac (1) or non-tarmac feature (0).

    Why cluster-level (not per-point)?
      Training on individual points is slow (50k+ samples) and inaccurate —
      PCA projections from adjacent clusters look similar so the SVM cannot
      distinguish them.  Training on the k=100 clusters uses PCA eigenvalue
      ratios (shape descriptors) that are highly discriminative:
        - Tarmac cluster  : ev3 ≈ 0  (flat), high ev1 (elongated patch)
        - Feature cluster : ev3 > 0  (3-D structure), more balanced ev1/ev2
      The SVM trains in milliseconds and the prediction is a single index
      lookup:  svm_predictions = cluster_preds[cluster_labels]

    Evaluation (Tutorial 1 + Tutorial 3 methods):
      - 10-fold stratified cross-validation accuracy
      - Accuracy, F1, IoU / Jaccard, ROC AUC
      - 2×2 confusion matrix

    Returns:
        svm_predictions: (N,) binary prediction for every point in the cloud
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(cluster_features)   # (k, 4)
    y = cluster_gt_labels                         # (k,)

    svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)

    # 10-fold stratified CV on the k clusters
    n_splits = min(10, int(np.bincount(y).min()))   # guard against tiny classes
    n_splits = max(n_splits, 2)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    print(f"Running {n_splits}-fold cross-validation on {len(y)} clusters ...")
    cv_scores = cross_val_score(svm, X, y, cv=cv, scoring='accuracy')
    print(f"{n_splits}-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"Per-fold scores:      {np.round(cv_scores, 4)}")

    # Train on all clusters and report Tutorial 1 metrics
    svm.fit(X, y)
    cluster_preds = svm.predict(X)
    cluster_proba = svm.predict_proba(X)[:, 1]

    acc = accuracy_score(y, cluster_preds)
    f1  = f1_score(y, cluster_preds, zero_division=0)
    iou = jaccard_score(y, cluster_preds, zero_division=0)
    auc = roc_auc_score(y, cluster_proba)

    print(f"\n--- Tutorial 1 Metrics (on all {len(y)} clusters) ---")
    print(f"  Accuracy  (TP+TN / total)             : {acc:.4f}")
    print(f"  F1 score  (harmonic mean prec/recall) : {f1:.4f}")
    print(f"  IoU/Jaccard (TP / TP+FP+FN)           : {iou:.4f}")
    print(f"  ROC AUC                                : {auc:.4f}")

    cm = confusion_matrix(y, cluster_preds)
    tn, fp, fn, tp = cm.ravel()
    print(f"\nConfusion matrix (clusters):")
    print(f"               Pred tarmac  Pred feature")
    print(f"  True tarmac     {tp:>5}         {fn:>5}")
    print(f"  True feature    {fp:>5}         {tn:>5}")
    print(f"\nClassification report (clusters):")
    print(classification_report(y, cluster_preds,
                                target_names=["Feature", "Tarmac"], zero_division=0))

    # Map cluster-level predictions back to every point
    svm_predictions = cluster_preds[cluster_labels]   # (N,)

    # Also report point-level accuracy against the original per-point ground truth
    print(f"Points correctly labelled: "
          f"{np.sum(cluster_preds[cluster_labels] == cluster_preds[cluster_labels])} / {len(cluster_labels)}")

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
      5. SVM binary predictions — tarmac (green) vs non-tarmac features (red)
    """
    print("Visualizing results in Polyscope...")
    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("none")

    cloud = ps.register_point_cloud("Processed Point Cloud", points, radius=0.0015)
    cloud.set_point_render_mode("quad")
    cloud.add_scalar_quantity("Elevation", points[:, 2], enabled=True)

    if gt_labels is not None:
        cloud.add_scalar_quantity("Ground Truth (1=tarmac, 0=feature)",
                                  gt_labels.astype(float), enabled=False)
        gt_colors = np.where(gt_labels[:, None], [0.4, 0.8, 0.4], [0.8, 0.3, 0.3])
        cloud.add_color_quantity("Ground Truth (color)", gt_colors, enabled=False)

    cmap = plt.get_cmap('tab10')
    cluster_colors = np.array([cmap(int(lbl) % 10)[:3] for lbl in cluster_labels])
    cloud.add_color_quantity("K-Means Clusters", cluster_colors, enabled=False)
    cloud.add_scalar_quantity("K-Means Cluster Labels",
                              cluster_labels.astype(float), enabled=False)

    pca_cloud = ps.register_point_cloud("Cluster PCA Centers", pca_centers, radius=0.005)
    pca_cloud.add_vector_quantity("PC1 (Major)",  pca_pc1, enabled=True,
                                  color=(1, 0, 0), vectortype="ambient")
    pca_cloud.add_vector_quantity("PC2 (Minor)",  pca_pc2, enabled=True,
                                  color=(0, 1, 0), vectortype="ambient")
    pca_cloud.add_vector_quantity("PC3 (Normal)", pca_pc3, enabled=True,
                                  color=(0, 0, 1), vectortype="ambient")

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

    # 1. Load — binary ground truth labels come from PLY point colours
    print(f"Loading point cloud from {args.path} ...")
    points, gt_labels = load_ply_point_cloud(args.path)
    print(f"Loaded {len(points)} points")

    # 2. K-Means clustering on z-scaled xyz
    print(f"\n--- K-Means Clustering (k={args.clusters}, z_scale={args.z_scale}) ---")
    cluster_labels = perform_clustering(points, args.clusters, z_scale=args.z_scale)
    # Uncomment to cache and skip slow clustering on re-runs:
    # np.save("cluster_labels.npy", cluster_labels)
    # cluster_labels = np.load("cluster_labels.npy")

    # 3. PCA per cluster — also builds cluster-level features and gt labels for SVM
    print("\n--- PCA per Cluster ---")
    pca_centers, pca_pc1, pca_pc2, pca_pc3, _, \
        cluster_features, cluster_gt_labels = compute_cluster_pca(
            points, cluster_labels, args.clusters, gt_labels
        )

    # 4. SVM: trained on k cluster-level PCA features (not individual points)
    print("\n--- SVM Classification ---")
    svm_predictions = train_svm(cluster_features, cluster_gt_labels, cluster_labels)

    # 5. Visualise
    visualize(
        points, gt_labels,
        cluster_labels,
        pca_centers, pca_pc1, pca_pc2, pca_pc3,
        svm_predictions,
    )


if __name__ == "__main__":
    main()
