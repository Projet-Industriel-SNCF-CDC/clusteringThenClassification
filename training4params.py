import numpy as np
import pandas as pd
import os
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from scipy.spatial import ConvexHull, distance_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import joblib
import json
from tqdm import tqdm

# Configuration
np.random.seed(42)

print("="*80)
print("OPTIMISATION DES PARAMÈTRES - HISTGRADIENTBOOSTINGCLASSIFIER")
print("="*80)

# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def load_point_cloud(filepath):
    df_input = pd.read_csv(filepath) 
    points = df_input[['x', 'y', 'z']].values
    return points

def load_clouds_from_folder(folder_path):
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
    n_points = len(point_cloud)
    features = {}
    
    centroid = np.mean(point_cloud, axis=0)
    
    try:
        hull = ConvexHull(point_cloud)
        volume = hull.volume
        features['density'] = n_points / volume if volume > 0 else 0
        features['hull_area'] = hull.area
    except:
        features['density'] = 0
        features['hull_area'] = 0
    
    sample_size = min(50, n_points)
    sample_indices = np.random.choice(n_points, sample_size, replace=False)
    sample_points = point_cloud[sample_indices]
    dist_matrix = distance_matrix(sample_points, sample_points)
    
    upper_triangle = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
    features['mean_interdist'] = np.mean(upper_triangle)
    features['std_interdist'] = np.std(upper_triangle)
    features['min_interdist'] = np.min(upper_triangle) if len(upper_triangle) > 0 else 0
    features['max_interdist'] = np.max(upper_triangle) if len(upper_triangle) > 0 else 0
    
    distances_to_centroid = np.linalg.norm(point_cloud - centroid, axis=1)
    features['mean_dist_centroid'] = np.mean(distances_to_centroid)
    features['std_dist_centroid'] = np.std(distances_to_centroid)
    
    features['var_x'] = np.var(point_cloud[:, 0])
    features['var_y'] = np.var(point_cloud[:, 1])
    features['var_z'] = np.var(point_cloud[:, 2])
    
    min_coords = np.min(point_cloud, axis=0)
    max_coords = np.max(point_cloud, axis=0)
    bbox_dims = max_coords - min_coords
    bbox_dims = np.where(bbox_dims == 0, 1e-6, bbox_dims)
    
    features['bbox_volume'] = np.prod(bbox_dims)
    features['bbox_ratio_xy'] = bbox_dims[0] / bbox_dims[1]
    features['bbox_ratio_xz'] = bbox_dims[0] / bbox_dims[2]
    features['bbox_ratio_yz'] = bbox_dims[1] / bbox_dims[2]
    
    radius = np.percentile(distances_to_centroid, 25) if len(distances_to_centroid) > 0 else 1
    sample_size_neighbors = min(20, n_points)
    sample_idx = np.random.choice(n_points, sample_size_neighbors, replace=False)
    neighbors_counts = []
    for idx in sample_idx:
        distances = np.linalg.norm(point_cloud - point_cloud[idx], axis=1)
        neighbors_counts.append(np.sum(distances < radius))
    features['mean_neighbors'] = np.mean(neighbors_counts)
    features['std_neighbors'] = np.std(neighbors_counts)
    
    features['spread'] = np.max(distances_to_centroid) if len(distances_to_centroid) > 0 else 0
    features['n_points'] = n_points
    
    return features

def train_and_evaluate(X_train, y_train, X_test, y_test, **model_params):
    """Entraîne et évalue le modèle avec les paramètres donnés"""
    model = HistGradientBoostingClassifier(
        random_state=42,
        class_weight='balanced',
        **model_params
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    
    return metrics

# ============================================================
# CHARGEMENT DES DONNÉES
# ============================================================
print("\n1. Chargement des données depuis les dossiers...")

OBSTACLE_FOLDER = "export/obstacle"
NON_OBSTACLE_FOLDER = "export/non_obstacle"

if not os.path.exists(OBSTACLE_FOLDER):
    os.makedirs(OBSTACLE_FOLDER, exist_ok=True)
    print(f"Dossier créé (vide): {OBSTACLE_FOLDER}")

if not os.path.exists(NON_OBSTACLE_FOLDER):
    os.makedirs(NON_OBSTACLE_FOLDER, exist_ok=True)
    print(f"Dossier créé (vide): {NON_OBSTACLE_FOLDER}")

clouds_obstacle = load_clouds_from_folder(OBSTACLE_FOLDER)
clouds_non_obstacle = load_clouds_from_folder(NON_OBSTACLE_FOLDER)

if len(clouds_obstacle) == 0 or len(clouds_non_obstacle) == 0:
    print("\n" + "="*80)
    print("ERREUR: Données insuffisantes!")
    print("="*80)
    exit(1)

# ============================================================
# SÉPARATION TRAIN/TEST
# ============================================================
print("\n2. Séparation des données en train/test...")

TEST_RATIO = 0.2

np.random.shuffle(clouds_obstacle)
split_idx_obs = int(len(clouds_obstacle) * (1 - TEST_RATIO))
train_clouds_obstacle = clouds_obstacle[:split_idx_obs]
test_clouds_obstacle = clouds_obstacle[split_idx_obs:]

np.random.shuffle(clouds_non_obstacle)
split_idx_non = int(len(clouds_non_obstacle) * (1 - TEST_RATIO))
train_clouds_non_obstacle = clouds_non_obstacle[:split_idx_non]
test_clouds_non_obstacle = clouds_non_obstacle[split_idx_non:]

print(f"   ✓ Train - Obstacles: {len(train_clouds_obstacle)}, Non-obstacles: {len(train_clouds_non_obstacle)}")
print(f"   ✓ Test  - Obstacles: {len(test_clouds_obstacle)}, Non-obstacles: {len(test_clouds_non_obstacle)}")

# ============================================================
# EXTRACTION DES FEATURES
# ============================================================
print("\n3. Extraction des features géométriques...")

X_train = []
y_train = []

for cloud in train_clouds_obstacle:
    features = extract_geometric_features(cloud)
    X_train.append(list(features.values()))
    y_train.append(1)

for cloud in train_clouds_non_obstacle:
    features = extract_geometric_features(cloud)
    X_train.append(list(features.values()))
    y_train.append(0)

X_train = np.array(X_train)
y_train = np.array(y_train)

X_test = []
y_test = []

for cloud in test_clouds_obstacle:
    features = extract_geometric_features(cloud)
    X_test.append(list(features.values()))
    y_test.append(1)

for cloud in test_clouds_non_obstacle:
    features = extract_geometric_features(cloud)
    X_test.append(list(features.values()))
    y_test.append(0)

X_test = np.array(X_test)
y_test = np.array(y_test)

print(f"   ✓ Total échantillons d'entraînement : {len(X_train)}")
print(f"   ✓ Total échantillons de test : {len(X_test)}")

# ============================================================
# OPTIMISATION DES PARAMÈTRES
# ============================================================
print("\n4. Optimisation des paramètres...")

# Paramètres de base (valeurs par défaut optimales)
BASE_PARAMS = {
    'learning_rate': 0.1,
    'max_depth': 20,
    'max_iter': 300,
    'min_samples_leaf': 20
}

# Définition des plages de variation pour chaque paramètre
PARAM_RANGES = {
    'learning_rate': np.linspace(0.01, 0.5, 15),
    'max_depth': list(range(3, 31, 2)),
    'max_iter': list(range(50, 501, 25)),
    'min_samples_leaf': list(range(1, 51, 3))
}

# Stockage des résultats
results = {}

# Test de chaque paramètre indépendamment
for param_name, param_values in PARAM_RANGES.items():
    print(f"\n   Testing parameter: {param_name}")
    print(f"   Values tested: {len(param_values)}")
    
    results[param_name] = {
        'values': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': []
    }
    
    for param_value in tqdm(param_values, desc=f"   {param_name}"):
        # Créer une copie des paramètres de base
        current_params = BASE_PARAMS.copy()
        # Modifier le paramètre testé
        current_params[param_name] = param_value
        
        # Entraîner et évaluer
        metrics = train_and_evaluate(X_train, y_train, X_test, y_test, **current_params)
        
        # Stocker les résultats
        results[param_name]['values'].append(param_value)
        results[param_name]['accuracy'].append(metrics['accuracy'])
        results[param_name]['precision'].append(metrics['precision'])
        results[param_name]['recall'].append(metrics['recall'])
        results[param_name]['f1_score'].append(metrics['f1_score'])

# ============================================================
# VISUALISATION DES RÉSULTATS
# ============================================================
print("\n5. Génération des graphiques d'optimisation...")

EXPORT_DIR = "model_optimization_HGB"
os.makedirs(EXPORT_DIR, exist_ok=True)

# Créer une figure avec 4 sous-graphiques (un par paramètre)
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
axes = axes.flatten()

for idx, (param_name, data) in enumerate(results.items()):
    ax = axes[idx]
    
    # Tracer les 4 métriques
    ax.plot(data['values'], data['accuracy'], 'o-', label='Accuracy', linewidth=2, markersize=5)
    ax.plot(data['values'], data['precision'], 's-', label='Precision', linewidth=2, markersize=5)
    ax.plot(data['values'], data['recall'], '^-', label='Recall', linewidth=2, markersize=5)
    ax.plot(data['values'], data['f1_score'], 'd-', label='F1-Score', linewidth=2, markersize=5)
    
    # Trouver la meilleure valeur (basée sur F1-score)
    best_idx = np.argmax(data['f1_score'])
    best_value = data['values'][best_idx]
    best_f1 = data['f1_score'][best_idx]
    
    # Marquer le meilleur point
    ax.axvline(x=best_value, color='red', linestyle='--', alpha=0.5, label=f'Best: {best_value}')
    
    # Mise en forme
    ax.set_xlabel(param_name, fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Impact de {param_name} sur les performances\n(Meilleure valeur: {best_value}, F1={best_f1:.4f})', 
                 fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.7, 1.05])

plt.tight_layout()
plt.savefig(os.path.join(EXPORT_DIR, 'parameter_optimization.png'), dpi=150, bbox_inches='tight')
print(f"   ✓ Graphiques sauvegardés : {EXPORT_DIR}/parameter_optimization.png")

# ============================================================
# EXPORT DES RÉSULTATS
# ============================================================
print("\n6. Export des résultats détaillés...")

# Sauvegarder les résultats en CSV pour chaque paramètre
for param_name, data in results.items():
    df = pd.DataFrame({
        param_name: data['values'],
        'accuracy': data['accuracy'],
        'precision': data['precision'],
        'recall': data['recall'],
        'f1_score': data['f1_score']
    })
    
    csv_path = os.path.join(EXPORT_DIR, f'{param_name}_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"   ✓ {param_name} : {csv_path}")

# Créer un résumé des meilleures valeurs
summary = {}
for param_name, data in results.items():
    best_idx = np.argmax(data['f1_score'])
    summary[param_name] = {
        'best_value': data['values'][best_idx],
        'accuracy': data['accuracy'][best_idx],
        'precision': data['precision'][best_idx],
        'recall': data['recall'][best_idx],
        'f1_score': data['f1_score'][best_idx]
    }

summary_path = os.path.join(EXPORT_DIR, 'best_parameters_summary.json')
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=4)
print(f"   ✓ Résumé : {summary_path}")

# Afficher le résumé
print("\n" + "="*80)
print("RÉSUMÉ DES MEILLEURES VALEURS PAR PARAMÈTRE")
print("="*80)
for param_name, metrics in summary.items():
    print(f"\n{param_name:20s} = {metrics['best_value']}")
    print(f"  → F1-Score: {metrics['f1_score']:.4f} | Accuracy: {metrics['accuracy']:.4f}")
    print(f"  → Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}")

print("\n" + "="*80)
print("OPTIMISATION TERMINÉE")
print("="*80)
print(f"Résultats exportés dans : {EXPORT_DIR}/")
print("="*80)

plt.show()