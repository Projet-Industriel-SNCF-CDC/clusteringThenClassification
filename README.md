
# Clustering then Classification

## Still in process...

## Dependancies : 

pip install hdbscan numpy matplotlib scikit-learn pandas seaborn tqdm

### Test avec données générées artificiellement (de type blob)

#### Clustering : 

file : test2.py 

- Load un fichier CSV
- Ajoute des données artificielles (type blob)
- Utilise l'algo HDBSCAN de la bibliotheque Scipy-kit
    - Les paramètres peuvent être modifiés
- Export individuel des clusters dans des fichiers CSV si ils respectent un gabarit de taille
- Export du nuage de point global 'clusterisé' avec des labels pour chacun des points
- Visualisation du nuage de point avant et après clustering

file : test3.py : 

- Load fichier .dat
- Clustering
- Visualisation du fichier avec et sans clustering

#### Classification : 

file : ai3.py

- Génération des données d'entrainements (50 pour chaque type : nuage dense et nuage léger) 
- Génération des données de tests (20 chacun)
- Extraction des features géométriques
- Entrainement de l'algo random forest de la bibliotheque scipy
- Evaluation
- Visualisation



### Génération dataset de clusters 

file : geneData.py 

- Réalise un clustering des données synthétiques labellisées obtenus par ray-tracing. En faisant une moyenne des points des clusters, on obtient leur label et on peut les rangées dans le dossier correspondant : obstacle ou non-obstacle

### Entraînement modèle de classification

file : training2.py

- Sur le dataset généré, entraîne un algorithme de classification **Random Forest** avec une extraction préalable des caractéristiques du nuage de point.
- Evalue le modèle avec le dataset de test et génère les graphiques d'évaluation
- Exporte le modèle pour utilisation ultérieur


file : training3.py 

- Sur le dataset généré, entraîne un algorithme de classification **Histogramme Gradien Boosting** avec une extraction préalable des caractéristiques du nuage de point.
- Evalue le modèle avec le dataset de test et génère les graphiques d'évaluation
- Exporte le modèle pour utilisation ultérieur


#### Optimisation des paramètres 

file : training4params.py

- Optimise les paramètres suivants du classificateur **Histogramme Gradien Boosting** : learning_rate, max_depth, max_iter, min_samples_leaf.
- Génère les graphiques de l'évolution des 4 paramètres et indique le plus optimale. 


### Utilisation de la pipeline (clustering + classification) sur nouvelles données : 

file : testModel1.py

- Importe un model de classification
- Filtre les points dans une zone prédéfinie (celle du passage du véhicule)
- Clustering avec hdbscan
- Extraction des caractéristiques géométriques des clusters de points
- Classification avec le model importé
- Affiche le résultat (prédiction et vérité)


