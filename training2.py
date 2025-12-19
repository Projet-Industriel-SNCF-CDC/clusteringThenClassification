import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.spatial import ConvexHull, distance_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import glob
import joblib
import json

# Configuration
np.random.seed(42)

print("="*80)
print("CLASSIFICATION DE NUAGES DE POINTS - DÉTECTION D'OBSTACLES")
print("="*80)

# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def load_point_cloud(filepath):

    #df_input = pd.read_csv('rails/lidar_points_with_labels.csv')  # Adapter le nom du fichier
    df_input = pd.read_csv(filepath) 
    points = df_input[['x', 'y', 'z']].values
    #obstacle_labels = df_input['label'].values  # Colonne contenant les labels (0 ou 1)
    return points



def load_clouds_from_folder(folder_path):
    """
    Charge tous les nuages de points d'un dossier.
    Retourne une liste de nuages de points (numpy arrays).
    """
    clouds = []
    supported_extensions = ['*.csv']
    
    files = []
    for ext in supported_extensions:
        files.extend(glob.glob(os.path.join(folder_path, ext)))
    
    print(f"   Dossier: {folder_path}")
    print(f"   Fichiers trouvés: {len(files)}")
    
    for filepath in sorted(files):
        try:
            cloud = load_point_cloud(filepath)
            clouds.append(cloud)               
        except Exception as e:
            print(f" Erreur chargement {os.path.basename(filepath)}: {e}")
    
    print(f" Nuages chargés: {len(clouds)}")
    return clouds


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


# ============================================================
# ÉTAPE 1 : RÉCUPÉRATION DES DONNÉES
# ============================================================
print("\n1. Chargement des données depuis les dossiers...")

# Chemins des dossiers
OBSTACLE_FOLDER = "export/obstacle"
NON_OBSTACLE_FOLDER = "export/non_obstacle"

# Vérification de l'existence des dossiers
if not os.path.exists(OBSTACLE_FOLDER):
    os.makedirs(OBSTACLE_FOLDER, exist_ok=True)
    print(f"Dossier créé (vide): {OBSTACLE_FOLDER}")

if not os.path.exists(NON_OBSTACLE_FOLDER):
    os.makedirs(NON_OBSTACLE_FOLDER, exist_ok=True)
    print(f"Dossier créé (vide): {NON_OBSTACLE_FOLDER}")

# Chargement des nuages de points
clouds_obstacle = load_clouds_from_folder(OBSTACLE_FOLDER)
clouds_non_obstacle = load_clouds_from_folder(NON_OBSTACLE_FOLDER)

if len(clouds_obstacle) == 0 or len(clouds_non_obstacle) == 0:
    print("\n" + "="*80)
    print("ERREUR: Données insuffisantes!")
    print("Veuillez placer vos fichiers de nuages de points dans:")
    print(f"  - {OBSTACLE_FOLDER}/ pour les obstacles")
    print(f"  - {NON_OBSTACLE_FOLDER}/ pour les non-obstacles")
    print("Formats supportés: .csv")
    print("="*80)
    exit(1)

# ============================================================
# ÉTAPE 2 : SÉPARATION TRAIN/TEST
# ============================================================
print("\n2. Séparation des données en train/test...")

TEST_RATIO = 0.2

# Mélanger et séparer les données obstacle
np.random.shuffle(clouds_obstacle)
split_idx_obs = int(len(clouds_obstacle) * (1 - TEST_RATIO))
train_clouds_obstacle = clouds_obstacle[:split_idx_obs]
test_clouds_obstacle = clouds_obstacle[split_idx_obs:]

# Mélanger et séparer les données non-obstacle
np.random.shuffle(clouds_non_obstacle)
split_idx_non = int(len(clouds_non_obstacle) * (1 - TEST_RATIO))
train_clouds_non_obstacle = clouds_non_obstacle[:split_idx_non]
test_clouds_non_obstacle = clouds_non_obstacle[split_idx_non:]

print(f"   ✓ Train - Obstacles: {len(train_clouds_obstacle)}, Non-obstacles: {len(train_clouds_non_obstacle)}")
print(f"   ✓ Test  - Obstacles: {len(test_clouds_obstacle)}, Non-obstacles: {len(test_clouds_non_obstacle)}")

# ============================================================
# ÉTAPE 3 : EXTRACTION DES FEATURES POUR L'ENTRAÎNEMENT
# ============================================================
print("\n3. Extraction des features géométriques (entraînement)...")

X_train = []
y_train = []

# Features pour obstacles (classe 1)
for cloud in train_clouds_obstacle:
    features = extract_geometric_features(cloud)
    X_train.append(list(features.values()))
    y_train.append(1)  # 1 = OBSTACLE

# Features pour non-obstacles (classe 0)
for cloud in train_clouds_non_obstacle:
    features = extract_geometric_features(cloud)
    X_train.append(list(features.values()))
    y_train.append(0)  # 0 = NON_OBSTACLE

X_train = np.array(X_train)
y_train = np.array(y_train)

feature_names = list(features.keys())
print(f"   ✓ Features extraites : {len(feature_names)}")
print(f"   ✓ Total échantillons d'entraînement : {len(X_train)}")

# ============================================================
# ÉTAPE 4 : EXTRACTION DES FEATURES POUR LE TEST
# ============================================================
print("\n4. Extraction des features géométriques (test)...")

X_test = []
y_test = []
test_clouds_all = []

for cloud in test_clouds_obstacle:
    features = extract_geometric_features(cloud)
    X_test.append(list(features.values()))
    y_test.append(1)
    test_clouds_all.append(cloud)

for cloud in test_clouds_non_obstacle:
    features = extract_geometric_features(cloud)
    X_test.append(list(features.values()))
    y_test.append(0)
    test_clouds_all.append(cloud)

X_test = np.array(X_test)
y_test = np.array(y_test)

print(f"   ✓ Total échantillons de test : {len(X_test)}")

# ============================================================
# ÉTAPE 5 : ENTRAÎNEMENT DU RANDOM FOREST
# ============================================================
print("\n5. Entraînement du modèle Random Forest...")

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=3, #
    min_samples_leaf=2, #
    class_weight="balanced", #
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
print(f"   ✓ Modèle entraîné avec {rf_model.n_estimators} arbres")

# ============================================================
# ÉTAPE 6 : ÉVALUATION
# ============================================================
print("\n6. Évaluation du modèle...")

y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n   PRÉCISION (Accuracy) : {accuracy * 100:.2f}%")
print("\n   RAPPORT DE CLASSIFICATION :")
print(classification_report(y_test, y_pred, 
                          target_names=['NON_OBSTACLE', 'OBSTACLE']))

cm = confusion_matrix(y_test, y_pred)
print("\n   MATRICE DE CONFUSION :")
print(cm)

# Importance des features
print("\n   TOP 10 FEATURES LES PLUS IMPORTANTES :")
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.head(10).iterrows():
    print(f"   {row['feature']:25s} : {row['importance']:.4f}")

# ============================================================
# ÉTAPE 7 : EXPORT DES PARAMÈTRES DU MODÈLE
# ============================================================
print("\n7. Export des paramètres du modèle...")

EXPORT_DIR = "model_export"
os.makedirs(EXPORT_DIR, exist_ok=True)

# 7.1 Export du modèle complet (joblib)
model_path = os.path.join(EXPORT_DIR, "random_forest_obstacle_detector.joblib")
joblib.dump(rf_model, model_path)
print(f"   ✓ Modèle sauvegardé : {model_path}")

# 7.2 Export des métadonnées et paramètres
model_metadata = {
    "model_type": "RandomForestClassifier",
    "classes": ["NON_OBSTACLE", "OBSTACLE"],
    "class_mapping": {0: "NON_OBSTACLE", 1: "OBSTACLE"},
    "feature_names": feature_names,
    "n_features": len(feature_names),
    "hyperparameters": {
        "n_estimators": rf_model.n_estimators,
        "max_depth": rf_model.max_depth,
        "min_samples_split": rf_model.min_samples_split,
        "min_samples_leaf": rf_model.min_samples_leaf,
        "random_state": 42
    },
    "training_info": {
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
        "train_obstacle_count": int(np.sum(y_train == 1)),
        "train_non_obstacle_count": int(np.sum(y_train == 0)),
        "test_accuracy": float(accuracy)
    },
    "feature_importance": {
        name: float(imp) for name, imp in zip(feature_names, rf_model.feature_importances_)
    }
}

metadata_path = os.path.join(EXPORT_DIR, "model_metadata.json")
with open(metadata_path, 'w') as f:
    json.dump(model_metadata, f, indent=4)
print(f"   ✓ Métadonnées sauvegardées : {metadata_path}")

# 7.3 Export des statistiques des features (pour normalisation future)
feature_stats = {
    "mean": X_train.mean(axis=0).tolist(),
    "std": X_train.std(axis=0).tolist(),
    "min": X_train.min(axis=0).tolist(),
    "max": X_train.max(axis=0).tolist(),
    "feature_names": feature_names
}

stats_path = os.path.join(EXPORT_DIR, "feature_statistics.json")
with open(stats_path, 'w') as f:
    json.dump(feature_stats, f, indent=4)
print(f"   ✓ Statistiques features : {stats_path}")

# 7.4 Export de l'importance des features
importance_path = os.path.join(EXPORT_DIR, "feature_importance.csv")
feature_importance.to_csv(importance_path, index=False)
print(f"   ✓ Importance features : {importance_path}")

print(f"\n   ✓ Tous les fichiers exportés dans : {EXPORT_DIR}/")

# ============================================================
# ÉTAPE 8 : VISUALISATION
# ============================================================
print("\n8. Génération des visualisations...")

fig = plt.figure(figsize=(16, 12))

# Visualisation 1-2 : Exemples de nuages de points
if len(train_clouds_obstacle) > 0:
    ax1 = fig.add_subplot(3, 3, 1, projection='3d')
    cloud = train_clouds_obstacle[0]
    ax1.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], c='red', alpha=0.6, s=10)
    ax1.set_title('OBSTACLE - Exemple', fontweight='bold')
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')

if len(train_clouds_non_obstacle) > 0:
    ax2 = fig.add_subplot(3, 3, 2, projection='3d')
    cloud = train_clouds_non_obstacle[0]
    ax2.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], c='blue', alpha=0.6, s=10)
    ax2.set_title('NON_OBSTACLE - Exemple', fontweight='bold')
    ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')

# Visualisation 3 : Distribution des classes
ax3 = fig.add_subplot(3, 3, 3)
class_counts = [np.sum(y_train == 0), np.sum(y_train == 1)]
ax3.bar(['NON_OBSTACLE', 'OBSTACLE'], class_counts, color=['blue', 'red'])
ax3.set_title('Distribution des classes (Train)', fontweight='bold')
ax3.set_ylabel('Nombre d\'échantillons')

# Visualisation 4 : Importance des features
ax4 = fig.add_subplot(3, 3, 4)
top_features = feature_importance.head(10)
ax4.barh(range(len(top_features)), top_features['importance'], color='green')
ax4.set_yticks(range(len(top_features)))
ax4.set_yticklabels(top_features['feature'])
ax4.set_xlabel('Importance')
ax4.set_title('Top 10 Features', fontweight='bold')
ax4.invert_yaxis()

# Visualisation 5 : Matrice de confusion
ax5 = fig.add_subplot(3, 3, 5)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax5,
           xticklabels=['NON_OBSTACLE', 'OBSTACLE'],
           yticklabels=['NON_OBSTACLE', 'OBSTACLE'])
ax5.set_title('Matrice de Confusion', fontweight='bold')
ax5.set_xlabel('Prédiction')
ax5.set_ylabel('Vérité')

# Visualisation 6-7 : Feature Space
if len(X_train) > 0:
    ax6 = fig.add_subplot(3, 3, 6)
    feat1_idx = feature_names.index('density')
    feat2_idx = feature_names.index('mean_interdist')
    ax6.scatter(X_train[y_train==0, feat1_idx], X_train[y_train==0, feat2_idx],
                c='blue', alpha=0.5, label='NON_OBSTACLE', s=30)
    ax6.scatter(X_train[y_train==1, feat1_idx], X_train[y_train==1, feat2_idx],
                c='red', alpha=0.5, label='OBSTACLE', s=30)
    ax6.set_xlabel('Density')
    ax6.set_ylabel('Mean Interdist')
    ax6.set_title('Feature Space 1', fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    ax7 = fig.add_subplot(3, 3, 7)
    feat3_idx = feature_names.index('mean_neighbors')
    feat4_idx = feature_names.index('std_dist_centroid')
    ax7.scatter(X_train[y_train==0, feat3_idx], X_train[y_train==0, feat4_idx],
                c='blue', alpha=0.5, label='NON_OBSTACLE', s=30)
    ax7.scatter(X_train[y_train==1, feat3_idx], X_train[y_train==1, feat4_idx],
                c='red', alpha=0.5, label='OBSTACLE', s=30)
    ax7.set_xlabel('Mean Neighbors')
    ax7.set_ylabel('Std Dist Centroid')
    ax7.set_title('Feature Space 2', fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

# Visualisation 8 : Accuracy par classe
ax8 = fig.add_subplot(3, 3, 8)
if len(cm) == 2:
    precision_non_obs = cm[0,0] / (cm[0,0] + cm[1,0]) if (cm[0,0] + cm[1,0]) > 0 else 0
    precision_obs = cm[1,1] / (cm[0,1] + cm[1,1]) if (cm[0,1] + cm[1,1]) > 0 else 0
    recall_non_obs = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
    recall_obs = cm[1,1] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0
    
    x = np.arange(2)
    width = 0.35
    ax8.bar(x - width/2, [precision_non_obs, precision_obs], width, label='Precision', color='steelblue')
    ax8.bar(x + width/2, [recall_non_obs, recall_obs], width, label='Recall', color='coral')
    ax8.set_xticks(x)
    ax8.set_xticklabels(['NON_OBSTACLE', 'OBSTACLE'])
    ax8.set_ylabel('Score')
    ax8.set_title('Precision & Recall par classe', fontweight='bold')
    ax8.legend()
    ax8.set_ylim(0, 1.1)

plt.tight_layout()
plt.savefig(os.path.join(EXPORT_DIR, 'obstacle_classification_results.png'), dpi=150, bbox_inches='tight')
print(f"   ✓ Visualisations sauvegardées : {EXPORT_DIR}/obstacle_classification_results.png")
plt.show()

print("\n" + "="*80)
print("ENTRAÎNEMENT TERMINÉ")
print("="*80)
print(f"Précision finale : {accuracy * 100:.2f}%")
print(f"Fichiers exportés dans : {EXPORT_DIR}/")
print("="*80)