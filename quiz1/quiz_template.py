import argparse

import numpy as np
import open3d as o3d
import polyscope as ps
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def load_ply_point_cloud(path: str):
    pcd = o3d.io.read_point_cloud(path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    ground_color_ref = np.array([0.6, 0.4, 0.1])
    is_ground = np.all(np.isclose(colors, ground_color_ref, atol=1e-2), axis=1)
    labels = is_ground.astype(int)

    n_ground = labels.sum()
    print(f"  Ground truth: {n_ground} tarmac pts, {len(labels) - n_ground} feature pts")

    return points, labels


def perform_clustering(points, k, z_scale=5.0):
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


def compute_cluster_pca(points, cluster_labels, k, binary_labels):
    centers = []
    pc1s, pc2s, pc3s = [], [], []
    pca_features = np.zeros((len(points), 3))
    cluster_features = np.zeros((k, 4))
    cluster_gt_labels = np.zeros(k, dtype=int)

    for i in range(k):
        mask = cluster_labels == i
        pts = points[mask]

        center = pts.mean(axis=0)
        centers.append(center)

        pca = PCA(n_components=3)
        pca.fit(pts)

        stds = np.sqrt(pca.explained_variance_)
        pc1s.append(pca.components_[0] * stds[0])
        pc2s.append(pca.components_[1] * stds[1])
        pc3s.append(pca.components_[2] * stds[2])

        pca_features[mask] = pca.transform(pts)
        cluster_features[i] = [pts[:, 2].mean(), *pca.explained_variance_ratio_]
        cluster_gt_labels[i] = int(binary_labels[mask].mean() >= 0.5)

        print(f"  Cluster {i}: ev = {np.round(pca.explained_variance_ratio_, 3)} "
              f"| mean_z = {pts[:, 2].mean():.2f} m "
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


def train_svm(cluster_features, cluster_gt_labels, cluster_labels, point_gt_labels):
    y = cluster_gt_labels

    scaler = StandardScaler()
    X = scaler.fit_transform(cluster_features)

    svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)

    n_splits = min(10, int(np.bincount(y).min()))
    n_splits = max(n_splits, 2)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = cross_val_score(svm, X, y, cv=cv, scoring='accuracy')

    print(f"Running {n_splits}-fold cross-validation on {len(y)} clusters ...")
    print(f"{n_splits}-Fold CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"Per-fold scores: {np.round(cv_scores, 3)}")

    svm.fit(X, y)
    cluster_preds = svm.predict(X)

    cv_acc = accuracy_score(y, cluster_preds)
    cv_f1 = f1_score(y, cluster_preds, zero_division=0)

    print(f"\n--- Cluster-Level Metrics ({len(y)} clusters) ---")
    print(f"  Accuracy  : {cv_acc:.3f}")
    print(f"  F1 score  : {cv_f1:.3f}")

    cv_cm = confusion_matrix(y, cluster_preds, labels=[1, 0])
    tp, fn, fp, tn = cv_cm.ravel()
    print(f"\nConfusion matrix (clusters):")
    print(f"               Pred tarmac  Pred feature")
    print(f"  True tarmac     {tp:>5}         {fn:>5}")
    print(f"  True feature    {fp:>5}         {tn:>5}")

    svm_predictions = cluster_preds[cluster_labels]

    point_acc = accuracy_score(point_gt_labels, svm_predictions)
    point_f1 = f1_score(point_gt_labels, svm_predictions, zero_division=0)
    n_wrong = int(np.sum(svm_predictions != point_gt_labels))

    print(f"\n--- Point-Level Metrics ---")
    print(f"  Accuracy  : {point_acc:.3f}")
    print(f"  F1 score  : {point_f1:.3f}")

    point_cm = confusion_matrix(point_gt_labels, svm_predictions, labels=[1, 0])
    tp, fn, fp, tn = point_cm.ravel()
    print(f"\nConfusion matrix (points):")
    print(f"               Pred tarmac  Pred feature")
    print(f"  True tarmac     {tp:>7}       {fn:>7}")
    print(f"  True feature    {fp:>7}       {tn:>7}")
    print(f"Misclassified: {n_wrong} / {len(point_gt_labels)}")

    return svm_predictions


def visualize(points, gt_labels, cluster_labels, pca_centers,
              pca_pc1, pca_pc2, pca_pc3, svm_predictions):
    print("Visualizing results in Polyscope...")
    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("none")

    cloud = ps.register_point_cloud("Processed Point Cloud", points, radius=0.0015)
    cloud.set_point_render_mode("quad")
    cloud.add_scalar_quantity("Elevation", points[:, 2], enabled=True)

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
    pca_cloud.add_vector_quantity("PC1 (Major)", pca_pc1, enabled=True,
                                  color=(1, 0, 0), vectortype="ambient")
    pca_cloud.add_vector_quantity("PC2 (Minor)", pca_pc2, enabled=True,
                                  color=(0, 1, 0), vectortype="ambient")
    pca_cloud.add_vector_quantity("PC3 (Normal)", pca_pc3, enabled=True,
                                  color=(0, 0, 1), vectortype="ambient")

    svm_colors = np.where(svm_predictions[:, None], [0.4, 0.8, 0.4], [0.8, 0.3, 0.3])
    cloud.add_color_quantity("SVM: Tarmac (green) vs Feature (red)",
                             svm_colors, enabled=False)
    cloud.add_scalar_quantity("SVM Predictions (1=tarmac, 0=feature)",
                              svm_predictions.astype(float), enabled=False)

    ps.show()


def main():
    parser = argparse.ArgumentParser(
        description="Point Cloud Segmentation pipeline (K-Means -> PCA -> SVM)"
    )
    parser.add_argument("path", nargs="?", default="airport_downsample.ply")
    parser.add_argument("-k", "--clusters", type=int, default=100)
    parser.add_argument("--z-scale", type=float, default=5.0)
    args = parser.parse_args()

    print(f"Loading {args.path} ...")
    points, gt_labels = load_ply_point_cloud(args.path)
    print(f"Loaded {len(points)} points")

    print(f"\n--- K-Means Clustering (k={args.clusters}, z_scale={args.z_scale}) ---")
    cluster_labels = perform_clustering(points, args.clusters, z_scale=args.z_scale)

    print("\n--- PCA per Cluster ---")
    pca_centers, pca_pc1, pca_pc2, pca_pc3, _, cluster_features, cluster_gt_labels = \
        compute_cluster_pca(points, cluster_labels, args.clusters, gt_labels)

    print("\n--- SVM Classification ---")
    svm_predictions = train_svm(
        cluster_features, cluster_gt_labels, cluster_labels, gt_labels
    )

    visualize(points, gt_labels, cluster_labels,
              pca_centers, pca_pc1, pca_pc2, pca_pc3, svm_predictions)


if __name__ == "__main__":
    main()
