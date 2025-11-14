# Requirements:
# pip install numpy pandas scikit-learn joblib


# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def extract_features(cluster_points):
    """
    cluster_points : (M,3) numpy array (x,y,z)
    Retourne un vecteur de features (1D numpy)
    """
    if len(cluster_points) == 0:
        return np.zeros(12)

    x = cluster_points[:,0]
    y = cluster_points[:,1]
    z = cluster_points[:,2]

    # Basic stats
    n_pts = len(cluster_points)
    mean_z = z.mean()
    std_z = z.std()
    min_z = z.min()
    max_z = z.max()
    span_z = max_z - min_z

    # Bounding box extents
    ext_x = x.max() - x.min()
    ext_y = y.max() - y.min()
    ext_z = span_z

    # Density: points per volume of bounding box (add epsilon)
    volume = max(ext_x * ext_y * ext_z, 1e-6)
    density = n_pts / volume

    # PCA eigenvalues (shape measure)
    cov = np.cov(cluster_points.T)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(eigvals)[::-1]  # descending
    # if degenerate, pad
    if eigvals.shape[0] < 3:
        eigvals = np.pad(eigvals, (0,3-eigvals.shape[0]), 'constant')

    # verticality: ratio of largest to sum
    verticality = eigvals[0] / (eigvals.sum() + 1e-9)

    features = np.array([
        n_pts, mean_z, std_z, min_z, max_z,
        ext_x, ext_y, ext_z, density,
        eigvals[0], eigvals[1], eigvals[2]
    ], dtype=float)
    return features

# Exemple d'utilisation :
# clusters = [np.array([...]), np.array([...]), ...]  # liste d'arrays (x,y,z)
# labels_list = [0,1,0, ...]  # 0 = non obstacle, 1 = obstacle (par ex)

def train_rf(clusters, labels_list):
    X = np.vstack([extract_features(c) for c in clusters])
    y = np.array(labels_list)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    joblib.dump(clf, "rf_cluster_model.joblib")
    return clf
