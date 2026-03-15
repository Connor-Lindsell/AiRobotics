# Instructions

## Venvs

It is recomended to create a virtual environment for this tutorial to avoid dependency conflicts with other projects. Follow the instruction there: https://www.w3schools.com/python/python_virtualenv.asp

## Dependcies

You will need to install the following Python packages to run the code in this tutorial:

```bash
pip install numpy open3d polyscope  
```

Additionally, if you want to run the k-means clustering, PCA, and SVM parts from `scikit-learn`, you will also need:

```bash
pip install scikit-learn
```

If you want to use other libraries, please contact the teaching team to make sure they are compatible with the code and the grading process.

## Running the code

To run the code, navigate to the `quiz_1` directory and execute the `quiz_template.py` script:

```bash
python quiz_template.py airport_downsample.ply
```

## Tasks

Points:

- Perform clustering of the points using k-means. (20 points)
    - Visualize the result of the clustering using Polyscope. (10 points)

- Compute the PCA of the points for each clusters. (20 points)
    - Visualize the result of the PCA using Polyscope. (10 points)

- Train an SVM classifier to predict the cluster labels based on the PCA features. (20 points)
    - Make sure to evaluate the performance of the classifier (e.g., 10-fold cross-validation). (10 points)
    - Visualize the classification results using Polyscope. (10 points)
