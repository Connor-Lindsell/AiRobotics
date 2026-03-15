import argparse
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import open3d as o3d
import polyscope as ps

# You can use the following packages for clustering, PCA, and SVM
#from sklearn.cluster import KMeans
#from sklearn.decomposition import PCA
#from sklearn.svm import SVC

# Additionally, you may want to use StandardScaler for height normalisation before clustering
#from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_ply_point_cloud(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load a point cloud from a .ply file.
    Returns points and binary ground truth labels derived from color (1=ground, 0=other).
    """
    pcd = o3d.io.read_point_cloud(path)
    points = np.asarray(pcd.points)
    if points.size == 0:
        raise ValueError(f"No points found in {path}")

    labels = None
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        if colors.shape == points.shape:
             # Ground color reference (Brown)
             ground_color_ref = np.array([0.6, 0.4, 0.1])
             # Check for exact equality
             is_ground = np.all(np.isclose(colors, ground_color_ref, atol=1e-2), axis=1)
             labels = is_ground.astype(int)

    return points, labels


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def visualize(points: np.ndarray, labels: np.ndarray, k: int) -> None:
    """
    Set up Polyscope serialization.
    """
    print("Visualizing results in Polyscope...")
    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("none")

    # 1. Register main point cloud
    cloud = ps.register_point_cloud("Processed Point Cloud", points, radius=0.0015)
    cloud.set_point_render_mode("quad")
    cloud.add_scalar_quantity("Elevation", points[:, 2], enabled=True)

    # 2. Add ground truth labels
    cloud.add_scalar_quantity("Ground truth data", labels, enabled=False)
    
    # 3. Add colors for clusters

    # 4. Register PCA visualization (Cluster Centers and Principal Components)
    # you can register pointclouds: https://polyscope.run/py/structures/point_cloud/basics/
    # you can then add vector quantities: https://polyscope.run/py/structures/point_cloud/vector_quantities/
    #pca_cloud = ps.register_point_cloud("Cluster PCA Centers", pca_centers, radius=0.0005)
    #pca_cloud.add_vector_quantity("PC1 (Major)", pca_pc1, enabled=True, color=(1, 0, 0), vectortype="ambient")
    #pca_cloud.add_vector_quantity("PC2 (Minor)", pca_pc1, enabled=True, color=(0, 1, 0), vectortype="ambient")
    #pca_cloud.add_vector_quantity("PC3 (Normal)", pca_pc1, enabled=True, color=(0, 0, 1), vectortype="ambient")

    # 5. Add SVM Predictions


    ps.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Point Cloud Segmentation pipeline (K-Means -> PCA -> SVM)")
    parser.add_argument("path", nargs="?", default="airport_downsample.ply")
    parser.add_argument("-k", "--clusters", type=int, default=6, help="Number of k-means clusters (default: 6)")
    args = parser.parse_args()

    # 1. Load Data
    ## returns points and labels
    points, point_gt_labels = load_ply_point_cloud(args.path)
    print(f"Loaded {len(points)} points from {args.path}")

    # You might want to create functions for each of the following points to keep the code clean

    # 2. Clustering
    # once the clustering is solved, you might consider saving the cluster in a .npy file
    # np.save("cluster_labels.npy", cluster_labels)
    # cluster_labels = np.load("cluster_labels.npy")

    # 3. Feature Extraction (PCA)

    # 4. Ground Truth Generation (for training SVM)

    # 5. SVM Classification

    # 6. Visualization
    visualize(points, point_gt_labels, args.clusters)


if __name__ == "__main__":
    main()