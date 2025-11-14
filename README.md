
### Clustering then Classification

## Still in process...

# Clustering : 

file : test2.py 

dependancies : pip install hdbscan numpy matplotlib scikit-learn pandas

- Load un fichier CSV
- Ajoute des données artificielles (type blob)
- Utilise l'algo HDBSCAN de la bibliotheque Scipy-kit
    - Les paramètres peuvent être modifiés
- Export individuel des clusters dans des fichiers CSV si ils respectent un gabarit de taille
- Export du nuage de point global 'clusterisé' avec des labels pour chacun des points
- Visualisation du nuage de point avant et après clustering


# Classification : 

file : ai3.py

dependancies : pip install seaborn

- Génération des données d'entrainements (50 pour chaque type : nuage dense et nuage léger) 
- Génération des données de tests (20 chacun)
- Extraction des features géométriques
- Entrainement de l'algo random forest de la bibliotheque scipy
- Evaluation
- Visualisation


