import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.spatial import ConvexHull, distance_matrix
#from scipy.stats import entropy
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Configuration
np.random.seed(42)

print("="*80)
print("CLASSIFICATION DE NUAGES DE POINTS PAR EXTRACTION DE FEATURES GÉOMÉTRIQUES")
print("="*80)

# ============================================================
# FONCTIONS D'EXTRACTION DE FEATURES GÉOMÉTRIQUES
# ============================================================

def extract_geometric_features(point_cloud):
    """
    Extrait des features géométriques d'un nuage de points 3D
    
    Features extraites :
    1. Densité (points / volume convex hull)
    2. Écart-type des distances inter-points
    3. Distance moyenne au centroïde
    4. Variance sur chaque axe (X, Y, Z)
    5. Ratio de la bounding box
    6. Nombre moyen de voisins dans un rayon donné
    7. Dispersion globale
    8. Entropie de la distribution
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
    
    # Prendre seulement le triangle supérieur (sans diagonale)
    upper_triangle = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
    features['mean_interdist'] = np.mean(upper_triangle)
    features['std_interdist'] = np.std(upper_triangle)
    features['min_interdist'] = np.min(upper_triangle)
    features['max_interdist'] = np.max(upper_triangle)
    
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
    features['bbox_volume'] = np.prod(bbox_dims)
    features['bbox_ratio_xy'] = bbox_dims[0] / bbox_dims[1] if bbox_dims[1] > 0 else 0
    features['bbox_ratio_xz'] = bbox_dims[0] / bbox_dims[2] if bbox_dims[2] > 0 else 0
    features['bbox_ratio_yz'] = bbox_dims[1] / bbox_dims[2] if bbox_dims[2] > 0 else 0
    
    # 6. Nombre de voisins dans un rayon (densité locale)
    radius = np.percentile(distances_to_centroid, 25)  # Rayon adaptatif
    sample_size_neighbors = min(20, n_points)
    sample_idx = np.random.choice(n_points, sample_size_neighbors, replace=False)
    neighbors_counts = []
    for idx in sample_idx:
        distances = np.linalg.norm(point_cloud - point_cloud[idx], axis=1)
        neighbors_counts.append(np.sum(distances < radius))
    features['mean_neighbors'] = np.mean(neighbors_counts)
    features['std_neighbors'] = np.std(neighbors_counts)
    
    # 7. Dispersion globale (spread)
    features['spread'] = np.max(distances_to_centroid)
    
    # 8. Nombre total de points
    features['n_points'] = n_points
    
    return features

# ============================================================
# GÉNÉRATION DE NUAGES DE POINTS
# ============================================================

def generate_dense_cloud(n_points=200, center=[0, 0, 0], radius=5):
    """Génère un nuage de points DENSE (compacte)"""
    # Distribution uniforme dans une sphère
    phi = np.random.uniform(0, 2*np.pi, n_points)
    theta = np.random.uniform(0, np.pi, n_points)
    r = np.random.uniform(0, radius, n_points) ** (1/3)  # Pour distribution uniforme en volume
    
    x = r * np.sin(theta) * np.cos(phi) + center[0]
    y = r * np.sin(theta) * np.sin(phi) + center[1]
    z = r * np.cos(theta) + center[2]
    
    return np.column_stack([x, y, z])

def generate_sparse_cloud(n_points=100, center=[0, 0, 0], radius=5):
    """Génère un nuage de points SPARSE (dispersé)"""
    # Distribution plus étalée
    points = np.random.randn(n_points, 3) * radius * 0.7 + center
    return points

# ============================================================
# ÉTAPE 1 : CRÉATION DES DONNÉES D'ENTRAÎNEMENT
# ============================================================
print("\n1. Génération des nuages de points d'entraînement...")

n_train_per_class = 50

# Catégorie 0 : Nuages DENSES
train_clouds_dense = []
for i in range(n_train_per_class):
    n_points = np.random.randint(150, 250)
    radius = np.random.uniform(3, 7)
    center = np.random.randn(3) * 5
    cloud = generate_dense_cloud(n_points, center, radius)
    train_clouds_dense.append(cloud)

# Catégorie 1 : Nuages SPARSE
train_clouds_sparse = []
for i in range(n_train_per_class):
    n_points = np.random.randint(50, 150)
    radius = np.random.uniform(3, 7)
    center = np.random.randn(3) * 5
    cloud = generate_sparse_cloud(n_points, center, radius)
    train_clouds_sparse.append(cloud)

print(f"   - Catégorie 0 (DENSE) : {len(train_clouds_dense)} nuages de points")
print(f"   - Catégorie 1 (SPARSE) : {len(train_clouds_sparse)} nuages de points")

# ============================================================
# ÉTAPE 2 : EXTRACTION DES FEATURES POUR L'ENTRAÎNEMENT
# ============================================================
print("\n2. Extraction des features géométriques (entraînement)...")

X_train = []
y_train = []

# Features pour nuages denses
for cloud in train_clouds_dense:
    features = extract_geometric_features(cloud)
    X_train.append(list(features.values()))
    y_train.append(0)

# Features pour nuages sparse
for cloud in train_clouds_sparse:
    features = extract_geometric_features(cloud)
    X_train.append(list(features.values()))
    y_train.append(1)

X_train = np.array(X_train)
y_train = np.array(y_train)

feature_names = list(features.keys())
print(f"   ✓ Features extraites : {len(feature_names)}")
print(f"   ✓ Total échantillons d'entraînement : {len(X_train)}")

# ============================================================
# ÉTAPE 3 : CRÉATION DES DONNÉES DE TEST
# ============================================================
print("\n3. Génération des nuages de points de test...")

n_test_per_class = 20

# Test - Catégorie 0 : Nuages DENSES
test_clouds_dense = []
for i in range(n_test_per_class):
    n_points = np.random.randint(150, 250)
    radius = np.random.uniform(3, 7)
    center = np.random.randn(3) * 5
    cloud = generate_dense_cloud(n_points, center, radius)
    test_clouds_dense.append(cloud)

# Test - Catégorie 1 : Nuages SPARSE
test_clouds_sparse = []
for i in range(n_test_per_class):
    n_points = np.random.randint(50, 150)
    radius = np.random.uniform(3, 7)
    center = np.random.randn(3) * 5
    cloud = generate_sparse_cloud(n_points, center, radius)
    test_clouds_sparse.append(cloud)

print(f"   - Catégorie 0 (DENSE) : {len(test_clouds_dense)} nuages de points")
print(f"   - Catégorie 1 (SPARSE) : {len(test_clouds_sparse)} nuages de points")

# ============================================================
# ÉTAPE 4 : EXTRACTION DES FEATURES POUR LE TEST
# ============================================================
print("\n4. Extraction des features géométriques (test)...")

X_test = []
y_test = []
test_clouds_all = []

for cloud in test_clouds_dense:
    features = extract_geometric_features(cloud)
    X_test.append(list(features.values()))
    y_test.append(0)
    test_clouds_all.append(cloud)

for cloud in test_clouds_sparse:
    features = extract_geometric_features(cloud)
    X_test.append(list(features.values()))
    y_test.append(1)
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
                          target_names=['DENSE', 'SPARSE']))

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
# ÉTAPE 7 : VISUALISATION
# ============================================================
print("\n7. Génération des visualisations...")

fig = plt.figure(figsize=(20, 12))

# Visualisation 1-4 : Exemples de nuages de points
for i in range(4):
    ax = fig.add_subplot(3, 4, i+1, projection='3d')
    if i < 2:
        cloud = train_clouds_dense[i]
        title = f'DENSE - Exemple {i+1}'
        color = 'blue'
    else:
        cloud = train_clouds_sparse[i-2]
        title = f'SPARSE - Exemple {i-1}'
        color = 'red'
    
    ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], 
              c=color, alpha=0.6, s=10)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

# Visualisation 5-8 : Prédictions sur nuages de test
for i in range(4):
    ax = fig.add_subplot(3, 4, i+5, projection='3d')
    cloud_idx = i * 10
    cloud = test_clouds_all[cloud_idx]
    true_label = y_test[cloud_idx]
    pred_label = y_pred[cloud_idx]
    
    color = 'green' if true_label == pred_label else 'orange'
    true_name = 'DENSE' if true_label == 0 else 'SPARSE'
    pred_name = 'DENSE' if pred_label == 0 else 'SPARSE'
    
    ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], 
              c=color, alpha=0.6, s=10)
    ax.set_title(f'Test: {true_name} → Pred: {pred_name}', fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

# Visualisation 9 : Importance des features
ax9 = fig.add_subplot(3, 4, 9)
top_features = feature_importance.head(10)
ax9.barh(range(len(top_features)), top_features['importance'])
ax9.set_yticks(range(len(top_features)))
ax9.set_yticklabels(top_features['feature'])
ax9.set_xlabel('Importance')
ax9.set_title('Top 10 Features', fontweight='bold')
ax9.invert_yaxis()

# Visualisation 10 : Matrice de confusion
ax10 = fig.add_subplot(3, 4, 10)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax10,
           xticklabels=['DENSE', 'SPARSE'],
           yticklabels=['DENSE', 'SPARSE'])
ax10.set_title('Matrice de Confusion', fontweight='bold')
ax10.set_xlabel('Prédiction')
ax10.set_ylabel('Vérité')

# Visualisation 11 : Distribution de 2 features importantes
ax11 = fig.add_subplot(3, 4, 11)
feat1_idx = feature_names.index('density')
feat2_idx = feature_names.index('mean_interdist')
ax11.scatter(X_train[y_train==0, feat1_idx], X_train[y_train==0, feat2_idx],
            c='blue', alpha=0.5, label='DENSE', s=30)
ax11.scatter(X_train[y_train==1, feat1_idx], X_train[y_train==1, feat2_idx],
            c='red', alpha=0.5, label='SPARSE', s=30)
ax11.set_xlabel('Density')
ax11.set_ylabel('Mean Interdist')
ax11.set_title('Feature Space', fontweight='bold')
ax11.legend()
ax11.grid(True, alpha=0.3)

# Visualisation 12 : Distribution de 2 autres features
ax12 = fig.add_subplot(3, 4, 12)
feat3_idx = feature_names.index('mean_neighbors')
feat4_idx = feature_names.index('std_dist_centroid')
ax12.scatter(X_train[y_train==0, feat3_idx], X_train[y_train==0, feat4_idx],
            c='blue', alpha=0.5, label='DENSE', s=30)
ax12.scatter(X_train[y_train==1, feat3_idx], X_train[y_train==1, feat4_idx],
            c='red', alpha=0.5, label='SPARSE', s=30)
ax12.set_xlabel('Mean Neighbors')
ax12.set_ylabel('Std Dist Centroid')
ax12.set_title('Feature Space 2', fontweight='bold')
ax12.legend()
ax12.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('point_cloud_classification.png', dpi=150, bbox_inches='tight')
print("   ✓ Visualisations sauvegardées : point_cloud_classification.png")
plt.show()

# ============================================================
# ÉTAPE 8 : TEST FINAL SUR DE NOUVEAUX NUAGES
# ============================================================
print("\n8. Test final sur de nouveaux nuages de points...")

# Créer 4 nouveaux nuages pour test final
new_test_clouds = [
    generate_dense_cloud(180, [0, 0, 0], 5),
    generate_sparse_cloud(150, [5, 5, 5], 7),
    generate_dense_cloud(220, [-3, 2, 1], 6),
    generate_sparse_cloud(110, [2, -3, 4], 4)
]

expected_labels = ['DENSE', 'SPARSE', 'DENSE', 'SPARSE']

print("\n   PRÉDICTIONS SUR NOUVEAUX NUAGES :")
for i, (cloud, expected) in enumerate(zip(new_test_clouds, expected_labels)):
    features = extract_geometric_features(cloud)
    features_array = np.array([list(features.values())])
    
    prediction = rf_model.predict(features_array)[0]
    proba = rf_model.predict_proba(features_array)[0]
    
    pred_label = 'DENSE' if prediction == 0 else 'SPARSE'
    confidence = proba[int(prediction)] * 100
    
    status = "✓" if pred_label == expected else "✗"
    print(f"   {status} Nuage {i+1} ({len(cloud)} points) : {pred_label} "
          f"(Confiance: {confidence:.1f}%) - Attendu: {expected}")

print("\n" + "="*80)
print("CLASSIFICATION PAR FEATURES GÉOMÉTRIQUES TERMINÉE !")
print("="*80)