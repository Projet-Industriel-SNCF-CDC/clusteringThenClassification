import os
import time
import joblib
import pickle
import numpy as np
import hdbscan
from scipy.spatial import ConvexHull, distance_matrix
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# ------------------------------
#      CHARGEMENT MODELE RF
# ------------------------------
#model_path = "model_export/random_forest_obstacle_detector.joblib"
model_path = "model_export/histGradBoosting_detector.joblib"
#print("[INFO] Chargement du modèle Random Forest...")
print("[INFO] Chargement du modèle Hist grad boosting...")

try:
    rf_model = joblib.load(model_path)
    print("[OK] Modèle chargé.")
except Exception as e:
    print("[ERREUR] Impossible de charger le modèle :", e)
    exit()

# ------------------------------
#      PARAMETRES HDBSCAN
# ------------------------------
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=20, #10 #20
    min_samples=40, #30 #40
    metric='euclidean'
)


def extract_geometric_features(point_cloud):
    """
    Extrait des features géométriques d'un nuage de points 3D
    """
    n_points = len(point_cloud)
    features = {}
    
    # Centroïde
    centroid = np.mean(point_cloud, axis=0)
    
    # 1. Densité (points par unité de volume)
    try:
        hull = ConvexHull(point_cloud)
        volume = hull.volume
        features['density'] = n_points / volume if volume > 0 else 0
        features['hull_area'] = hull.area
    except:
        features['density'] = 0
        features['hull_area'] = 0
    
    # 2. Distances inter-points (échantillonnage pour optimisation)
    sample_size = min(50, n_points)
    sample_indices = np.random.choice(n_points, sample_size, replace=False)
    sample_points = point_cloud[sample_indices]
    dist_matrix = distance_matrix(sample_points, sample_points)
    
    upper_triangle = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
    features['mean_interdist'] = np.mean(upper_triangle)
    features['std_interdist'] = np.std(upper_triangle)
    features['min_interdist'] = np.min(upper_triangle) if len(upper_triangle) > 0 else 0
    features['max_interdist'] = np.max(upper_triangle) if len(upper_triangle) > 0 else 0
    
    # 3. Distance au centroïde
    distances_to_centroid = np.linalg.norm(point_cloud - centroid, axis=1)
    features['mean_dist_centroid'] = np.mean(distances_to_centroid)
    features['std_dist_centroid'] = np.std(distances_to_centroid)
    
    # 4. Variance sur chaque axe
    features['var_x'] = np.var(point_cloud[:, 0])
    features['var_y'] = np.var(point_cloud[:, 1])
    features['var_z'] = np.var(point_cloud[:, 2])
    
    # 5. Bounding box et ratios
    min_coords = np.min(point_cloud, axis=0)
    max_coords = np.max(point_cloud, axis=0)
    bbox_dims = max_coords - min_coords
    bbox_dims = np.where(bbox_dims == 0, 1e-6, bbox_dims)  # Éviter division par zéro
    
    features['bbox_volume'] = np.prod(bbox_dims)
    features['bbox_ratio_xy'] = bbox_dims[0] / bbox_dims[1]
    features['bbox_ratio_xz'] = bbox_dims[0] / bbox_dims[2]
    features['bbox_ratio_yz'] = bbox_dims[1] / bbox_dims[2]
    
    # 6. Nombre de voisins dans un rayon (densité locale)
    radius = np.percentile(distances_to_centroid, 25) if len(distances_to_centroid) > 0 else 1
    sample_size_neighbors = min(20, n_points)
    sample_idx = np.random.choice(n_points, sample_size_neighbors, replace=False)
    neighbors_counts = []
    for idx in sample_idx:
        distances = np.linalg.norm(point_cloud - point_cloud[idx], axis=1)
        neighbors_counts.append(np.sum(distances < radius))
    features['mean_neighbors'] = np.mean(neighbors_counts)
    features['std_neighbors'] = np.std(neighbors_counts)
    
    # 7. Dispersion globale (spread)
    features['spread'] = np.max(distances_to_centroid) if len(distances_to_centroid) > 0 else 0
    
    # 8. Nombre total de points
    features['n_points'] = n_points
    
    return features


def filter_points(points, labels=None, z_range=(-0.78, 3), y_range=(-2, 2)):
    """
    Keep only points within specified Z and Y ranges.
    Also filters corresponding labels if provided.
    """
    z_mask = (points[:, 2] >= z_range[0]) & (points[:, 2] <= z_range[1])
    y_mask = (points[:, 0] >= y_range[0]) & (points[:, 0] <= y_range[1])
    x_mask = (points[:, 1] <= 80)
    
    final_mask = z_mask & y_mask & x_mask
    
    filtered_points = points[final_mask]
    filtered_labels = labels[final_mask] if labels is not None else None
    
    return filtered_points, filtered_labels


# ------------------------------
#      TRAITEMENT DES FICHIERS
# ------------------------------
#base_path = "/store/rob4sncf/data_witdth_cut_art"
base_path = "/store/rob4sncf/real_data"

# MESURE TEMPS GLOBAL
start_time = time.time()

# Listes pour stocker les prédictions et vérités terrain
all_predictions = []
all_ground_truths = []
#(405, 406, 407, 408, 409, 410)
#range(1, 18)
#range(10, 20)

for i in (405, 406, 407, 408, 409, 410):
    file_path = os.path.join(base_path, f"points_export_{i}.dat")

    if not os.path.exists(file_path):
        print(f"[WARN] Fichier manquant : {file_path}")
        continue

    print(f"\n=== Fichier {i} ===")

    # ------------------------------
    #      LECTURE DU FICHIER
    # ------------------------------
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        # Format attendu : (n, 4) → [x, y, z, label]
        points = data[:, :3].astype(np.float32)
        obstacle_labels = data[:, 3].astype(np.int64)

    except Exception as e:
        print(f"[ERREUR] Lecture du fichier : {e}")
        continue


    points, obstacle_labels = filter_points(points, obstacle_labels)


    print(f" - Points chargés : {points.shape[0]}")

    # ------------------------------
    #      HDBSCAN CLUSTERING
    # ------------------------------
    try:
        cluster_labels = clusterer.fit_predict(points)
        cluster_ids = set(cluster_labels)
        n_clusters = len(cluster_ids) - (1 if -1 in cluster_ids else 0)
        print(f" - Clusters détectés : {n_clusters}")
    except Exception as e:
        print(f"[ERREUR] HDBSCAN : {e}")
        continue

    # ------------------------------
    #      CLASSIFICATION RF SUR CHAQUE CLUSTER
    # ------------------------------
    for cluster_id in cluster_ids:
        if n_clusters > 1 and cluster_id == -1:
            continue  # Ignorer le bruit

        mask = cluster_labels == cluster_id
        cluster_points = points[mask]
        cluster_obstacle_labels = obstacle_labels[mask]


        # --- Extraction des features géométriques ---
        try:
            features_dict = extract_geometric_features(cluster_points)
            # Convertir en vecteur numpy dans l'ordre attendu par le modèle
            feature_names = [
                'density', 'hull_area', 'mean_interdist', 'std_interdist', 'min_interdist', 'max_interdist',
                'mean_dist_centroid', 'std_dist_centroid', 'var_x', 'var_y', 'var_z',
                'bbox_volume', 'bbox_ratio_xy', 'bbox_ratio_xz', 'bbox_ratio_yz',
                'mean_neighbors', 'std_neighbors', 'spread', 'n_points'
            ]
            feature_vector = np.array([features_dict[f] for f in feature_names]).reshape(1, -1)
        except Exception as e:
            print(f"   - Cluster {cluster_id} : Erreur extraction features : {e}")
            continue

        

        # --- Prédiction RF ---
        try:
            predictions = rf_model.predict(feature_vector)
            rf_label = int(predictions[0])
            rf_name = "obstacle" if rf_label == 1 else "non_obstacle"
        except Exception as e:
            print(f"   - Cluster {cluster_id} : Erreur Random Forest : {e}")
            continue

        # --- Comparaison avec la vérité ---
        mean_label = np.mean(cluster_obstacle_labels)
        final_label = int(np.round(mean_label))  # 0 ou 1
        label_name = "obstacle" if final_label == 1 else "non_obstacle"

        # Stocker pour les métriques globales
        all_predictions.append(rf_label)
        all_ground_truths.append(final_label)

        # --- Affichage ---
        print(f"   - Cluster {cluster_id} : {cluster_points.shape[0]} points")
        print(f"       RF prédictions : {rf_label} : {rf_name}")
        print(f"       Vérité (moyenne du cluster) : {final_label} : {label_name}")
        if rf_label == final_label : 
            print("################## VRAI")
        else : 
            print("################## FAUX")


#TEMPS D'EXECUTION
end_time = time.time()
execution_time = end_time - start_time
print(f"\n{'='*60}")
print(f"Temps d'exécution total : {execution_time:.2f} secondes")
print("="*60)

# ------------------------------
#      CALCUL DES MÉTRIQUES
# ------------------------------
print("\n" + "="*60)
print("RÉSULTATS GLOBAUX")
print("="*60)

if len(all_predictions) > 0:
    # Conversion en arrays numpy
    y_true = np.array(all_ground_truths)
    y_pred = np.array(all_predictions)
    
    # Calcul des métriques
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"\nNombre total de clusters évalués : {len(all_predictions)}")
    print(f"\nAccuracy  : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    
    # Rapport de classification détaillé
    print("\n" + "-"*60)
    print("RAPPORT DE CLASSIFICATION")
    print("-"*60)
    print(classification_report(y_true, y_pred, 
                                target_names=['Non-Obstacle', 'Obstacle'],
                                zero_division=0))
    
    # Matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    print("\n" + "-"*60)
    print("MATRICE DE CONFUSION")
    print("-"*60)
    print(f"                 Prédit: Non-Obstacle  Prédit: Obstacle")
    print(f"Réel: Non-Obstacle        {cm[0,0]:5d}              {cm[0,1]:5d}")
    print(f"Réel: Obstacle            {cm[1,0]:5d}              {cm[1,1]:5d}")
    
    # Visualisation de la matrice de confusion
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Obstacle', 'Obstacle'],
                yticklabels=['Non-Obstacle', 'Obstacle'])
    plt.title('Matrice de Confusion')
    plt.ylabel('Vérité Terrain')
    plt.xlabel('Prédiction')
    plt.tight_layout()
    
    # Sauvegarder la figure
    output_path = 'confusion_matrix.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nMatrice de confusion sauvegardée : {output_path}")
    plt.close()
    
else:
    print("\n[WARN] Aucune prédiction n'a été effectuée.")

