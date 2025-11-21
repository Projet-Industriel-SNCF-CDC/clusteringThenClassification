import numpy as np
import hdbscan
import pandas as pd
import os
from visu2 import visualize_pointcloud_csv
from visu import visualize_pointcloud_csv1
import pickle

# Créer les dossiers d'export s'ils n'existent pas
os.makedirs('export/obstacle', exist_ok=True)
os.makedirs('export/non_obstacle', exist_ok=True)

def filter_points(points, labels=None, z_range=(-0.8, 3), y_range=(-2, 2)):
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

# ==================== MAIN ====================


def main(filepath) :

    # 1. Charger le fichier CSV avec les labels existants
    #df_input = pd.read_csv('rails/lidar_points_with_labels.csv')  # Adapter le nom du fichier
    #points = df_input[['x', 'y', 'z']].values
    #obstacle_labels = df_input['label'].values  # Colonne contenant les labels (0 ou 1)

    #file_path = "rails/points_export_421.dat"
    file_path = filepath
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    # data est supposé avoir la forme (n, 4) : [x, y, z, label]
    points = data[:, :3].astype(np.float32)   # coordonnées XYZ
    obstacle_labels = data[:, 3].astype(np.int64)      # labels


    print(f"Données chargées: {len(points)} points")
    print(f"Points obstacles (label=1): {int(obstacle_labels.sum())}")
    print(f"Points non-obstacles (label=0): {int(len(obstacle_labels) - obstacle_labels.sum())}")

    # 2. Filtrer les points (optionnel)
    points, obstacle_labels = filter_points(points, obstacle_labels)
    print(f"\nAprès filtrage: {len(points)} points")

    # 3. Appliquer HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=10,
        min_samples=30,
        metric='euclidean',
    )
    cluster_labels = clusterer.fit_predict(points)

    # 4. Résultats du clustering
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)

    print(f"\n=== Résultats Clustering ===")
    print(f"Nombre de clusters détectés : {n_clusters}")
    print(f"Points de bruit : {n_noise}")

    # 5. Analyser chaque cluster et exporter
    cluster_counter = {'obstacle': 0, 'non_obstacle': 0}

    for cluster_id in set(cluster_labels):
        if cluster_id == -1:
            continue  # Ignorer le bruit
        
        # Récupérer les points et labels du cluster
        mask = cluster_labels == cluster_id
        cluster_points = points[mask]
        cluster_obstacle_labels = obstacle_labels[mask]
        n_points = cluster_points.shape[0]
        
        # Calculer le label moyen et arrondir
        mean_label = np.mean(cluster_obstacle_labels)
        final_label = int(np.round(mean_label))  # 0 ou 1
        
        label_name = "obstacle" if final_label == 1 else "non_obstacle"
        
        print(f"\nCluster {cluster_id}:")
        print(f"  - Points: {n_points}")
        print(f"  - Label moyen: {mean_label:.3f}")
        print(f"  - Label final: {final_label} ({label_name})")
        
        # Exporter le cluster
        if n_points > 10:
            # == EXPORT EN CSV == 
            df_cluster = pd.DataFrame(cluster_points, columns=['x', 'y', 'z'])
            # Exporter dans le bon dossier
            folder = f"export/{label_name}"
            filename = f"{folder}/cluster_{cluster_counter[label_name]}.csv"
            df_cluster.to_csv(filename, index=False)
            cluster_counter[label_name] += 1
            print(f"  - Exported: {filename}")

            # == EXPORT EN .DAT == 
            # Convertir en numpy array
            #cluster_array = np.array(cluster_points, dtype=np.float32)
            # Créer le dossier si nécessaire
            #folder = f"export/{label_name}"
            #filename = f"{folder}/cluster_{cluster_counter[label_name]}.dat"
            # Export en .dat avec pickle
            #with open(filename, 'wb') as f:
            #   pickle.dump(cluster_array, f)
            #cluster_counter[label_name] += 1
            #print(f"  - Exported: {filename}")

    # 6. Résumé final
    print(f"\n=== Export terminé ===")
    print(f"Clusters obstacles exportés: {cluster_counter['obstacle']}")
    print(f"Clusters non-obstacles exportés: {cluster_counter['non_obstacle']}")



if __name__ == "__main__" : 

    #main("rails/points_export_421.dat")

    for i in (100, 200) : 
        file_path = f"rails/points_export_{i}.dat"
        main(file_path)

    # 7. Sauvegarder pour visualisation
    #df_visu = pd.DataFrame(points, columns=['x', 'y', 'z'])
    #df_visu['cluster_label'] = cluster_labels
    #df_visu['obstacle_label'] = obstacle_labels
    #df_visu.to_csv('rails/lidar_points_with_clusters.csv', index=False)

    # 8. Visualiser
    #visualize_pointcloud_csv('rails/lidar_points_with_clusters.csv')