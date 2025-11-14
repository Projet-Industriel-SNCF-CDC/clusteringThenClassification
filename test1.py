
# pip install hdbscan numpy matplotlib scikit-learn

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import hdbscan

# 1. Créer des données simulées (nuage de points avec obstacles)
np.random.seed(42)

# Obstacle 1 : Cube dense (véhicule)
obstacle1 = np.random.randn(100, 3) * 0.3 + [2, 2, 1]

# Obstacle 2 : Sphère (piéton)
obstacle2 = np.random.randn(80, 3) * 0.2 + [5, 3, 1]

# Obstacle 3 : Petit cluster (poteau)
obstacle3 = np.random.randn(30, 3) * 0.15 + [7, 5, 1.5]

# Bruit : points épars (sol, végétation)
bruit = np.random.uniform(0, 10, (200, 3))

# Combiner tous les points
#points = np.vstack([obstacle1, obstacle2, obstacle3, bruit])

points = np.loadtxt('rails/lidar_points_0.0.csv', delimiter=',', skiprows=1)

print(f"Nombre total de points : {len(points)}")
print(f"Dimensions du nuage : {points.shape}")
print(f"Étendue X : [{points[:,0].min():.2f}, {points[:,0].max():.2f}]")
print(f"Étendue Y : [{points[:,1].min():.2f}, {points[:,1].max():.2f}]")
print(f"Étendue Z : [{points[:,2].min():.2f}, {points[:,2].max():.2f}]")

# 2. Appliquer HDBSCAN
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=1000,    # Taille minimum pour un obstacle
    min_samples=20,          # Robustesse au bruit
    metric='euclidean'
)
1, 
labels = clusterer.fit_predict(points)

# 3. Analyser les résultats
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f"\n=== Résultats ===")
print(f"Nombre de clusters détectés : {n_clusters}")
print(f"Points de bruit : {n_noise}")
print(f"Labels uniques : {set(labels)}")

# Afficher les statistiques par cluster
for cluster_id in set(labels):
    if cluster_id == -1:
        continue
    n_points = list(labels).count(cluster_id)
    print(f"Cluster {cluster_id} : {n_points} points")

# 4. Visualisation 3D
# 4. Visualisation 3D améliorée
fig = plt.figure(figsize=(18, 8))

# Vue 1 : Points originaux avec gradients de hauteur
ax1 = fig.add_subplot(131, projection='3d')
scatter1 = ax1.scatter(points[:, 0], points[:, 1], points[:, 2], 
                       c=points[:, 2], cmap='viridis', 
                       alpha=0.6, s=8, edgecolors='none')
ax1.set_title('Nuage de points original\n(couleur = hauteur Z)', fontsize=12, weight='bold')
ax1.set_xlabel('X (m)', fontsize=10)
ax1.set_ylabel('Y (m)', fontsize=10)
ax1.set_zlabel('Z (m)', fontsize=10)
plt.colorbar(scatter1, ax=ax1, shrink=0.5, label='Hauteur (m)')
ax1.view_init(elev=20, azim=45)

# Vue 2 : Vue du dessus (carte 2D)
ax2 = fig.add_subplot(132)
for cluster_id in set(labels):
    mask = labels == cluster_id
    if cluster_id == -1:
        ax2.scatter(points[mask, 0], points[mask, 1],
                   c='lightgray', alpha=0.2, s=3, label='Bruit')
    else:
        ax2.scatter(points[mask, 0], points[mask, 1],
                   alpha=0.8, s=20, label=f'Cluster {cluster_id}')
ax2.set_title(f'Vue du dessus\n({n_clusters} clusters détectés)', fontsize=12, weight='bold')
ax2.set_xlabel('X (m)', fontsize=10)
ax2.set_ylabel('Y (m)', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.axis('equal')
ax2.legend(loc='upper right', fontsize=8)

# Vue 3 : Clusters en 3D avec couleurs distinctes
ax3 = fig.add_subplot(133, projection='3d')
colors = plt.cm.tab20(np.linspace(0, 1, max(n_clusters, 1)))

for cluster_id in set(labels):
    mask = labels == cluster_id
    if cluster_id == -1:
        ax3.scatter(points[mask, 0], points[mask, 1], points[mask, 2],
                   c='lightgray', alpha=0.15, s=3, label='Bruit')
    else:
        ax3.scatter(points[mask, 0], points[mask, 1], points[mask, 2],
                   c=[colors[cluster_id % len(colors)]], alpha=0.9, s=25, 
                   label=f'Obstacle {cluster_id}', edgecolors='black', linewidth=0.5)

ax3.set_title(f'Obstacles détectés en 3D\n({n_clusters} obstacles)', fontsize=12, weight='bold')
ax3.set_xlabel('X (m)', fontsize=10)
ax3.set_ylabel('Y (m)', fontsize=10)
ax3.set_zlabel('Z (m)', fontsize=10)
ax3.legend(loc='upper left', fontsize=8, ncol=2)
ax3.view_init(elev=20, azim=45)

plt.tight_layout()
plt.show()



# 5. Extraire les clusters individuels (pour traitement ultérieur)
# print("\n=== Extraction des obstacles ===")
# obstacles_detectes = {}
# for cluster_id in set(labels):
#     if cluster_id != -1:
#         mask = labels == cluster_id
#         obstacle_points = points[mask]
#         obstacles_detectes[cluster_id] = obstacle_points
        
#         # Calculer le centre et les dimensions
#         centre = np.mean(obstacle_points, axis=0)
#         dimensions = np.ptp(obstacle_points, axis=0)  # étendue
        
#         print(f"\nObstacle {cluster_id}:")
#         print(f"  Centre : ({centre[0]:.2f}, {centre[1]:.2f}, {centre[2]:.2f})")
#         print(f"  Dimensions (L x l x h) : {dimensions[0]:.2f} x {dimensions[1]:.2f} x {dimensions[2]:.2f}")


