import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration pour la reproductibilité
np.random.seed(42)

print("="*60)
print("EXEMPLE DE CLASSIFICATION AVEC RANDOM FOREST")
print("="*60)

# ============================================================
# ÉTAPE 1 : CRÉATION DES DONNÉES D'ENTRAÎNEMENT (2 catégories)
# ============================================================
print("\n1. Création des données d'entraînement...")

# Catégorie 0 : 3 nuages de points distincts
print("   Catégorie 0 : 3 nuages de points")
nuage1_cat0 = np.random.randn(35, 2) * 0.8 + [2, 2]
nuage2_cat0 = np.random.randn(35, 2) * 0.8 + [2, 8]
nuage3_cat0 = np.random.randn(30, 2) * 0.8 + [5, 5]
X_train_cat0 = np.vstack([nuage1_cat0, nuage2_cat0, nuage3_cat0])
y_train_cat0 = np.zeros(len(X_train_cat0))

# Catégorie 1 : 3 nuages de points distincts
print("   Catégorie 1 : 3 nuages de points")
nuage1_cat1 = np.random.randn(35, 2) * 0.8 + [8, 2]
nuage2_cat1 = np.random.randn(35, 2) * 0.8 + [8, 8]
nuage3_cat1 = np.random.randn(30, 2) * 0.8 + [5, 10]
X_train_cat1 = np.vstack([nuage1_cat1, nuage2_cat1, nuage3_cat1])
y_train_cat1 = np.ones(len(X_train_cat1))

# Combinaison des données d'entraînement
X_train = np.vstack([X_train_cat0, X_train_cat1])
y_train = np.concatenate([y_train_cat0, y_train_cat1])

print(f"   - Catégorie 0 (entraînement) : {len(X_train_cat0)} échantillons (3 nuages)")
print(f"   - Catégorie 1 (entraînement) : {len(X_train_cat1)} échantillons (3 nuages)")
print(f"   - Total entraînement : {len(X_train)} échantillons")
print(f"   - Caractéristiques (features) : {X_train.shape[1]}")

# ============================================================
# ÉTAPE 2 : CRÉATION DES DONNÉES DE TEST (2 catégories)
# ============================================================
print("\n2. Création des données de test...")

# Catégorie 0 : 3 nuages de points pour le test
print("   Catégorie 0 : 3 nuages de points (test)")
nuage1_test_cat0 = np.random.randn(10, 2) * 0.8 + [2, 2]
nuage2_test_cat0 = np.random.randn(10, 2) * 0.8 + [2, 8]
nuage3_test_cat0 = np.random.randn(10, 2) * 0.8 + [5, 5]
X_test_cat0 = np.vstack([nuage1_test_cat0, nuage2_test_cat0, nuage3_test_cat0])
y_test_cat0 = np.zeros(len(X_test_cat0))

# Catégorie 1 : 3 nuages de points pour le test
print("   Catégorie 1 : 3 nuages de points (test)")
nuage1_test_cat1 = np.random.randn(10, 2) * 0.8 + [8, 2]
nuage2_test_cat1 = np.random.randn(10, 2) * 0.8 + [8, 8]
nuage3_test_cat1 = np.random.randn(10, 2) * 0.8 + [5, 10]
X_test_cat1 = np.vstack([nuage1_test_cat1, nuage2_test_cat1, nuage3_test_cat1])
y_test_cat1 = np.ones(len(X_test_cat1))

# Combinaison des données de test
X_test = np.vstack([X_test_cat0, X_test_cat1])
y_test = np.concatenate([y_test_cat0, y_test_cat1])

print(f"   - Catégorie 0 (test) : {len(X_test_cat0)} échantillons (3 nuages)")
print(f"   - Catégorie 1 (test) : {len(X_test_cat1)} échantillons (3 nuages)")
print(f"   - Total test : {len(X_test)} échantillons")

# ============================================================
# ÉTAPE 3 : ENTRAÎNEMENT DU MODÈLE RANDOM FOREST
# ============================================================
print("\n3. Entraînement du modèle Random Forest...")

# Création et entraînement du modèle
rf_model = RandomForestClassifier(
    n_estimators=100,      # Nombre d'arbres dans la forêt
    max_depth=10,          # Profondeur maximale des arbres
    random_state=42,       # Pour reproductibilité
    n_jobs=-1              # Utiliser tous les processeurs
)

rf_model.fit(X_train, y_train)
print(f"   ✓ Modèle entraîné avec {rf_model.n_estimators} arbres")

# ============================================================
# ÉTAPE 4 : PRÉDICTIONS ET ÉVALUATION
# ============================================================
print("\n4. Évaluation du modèle...")

# Prédictions sur les données de test
y_pred = rf_model.predict(X_test)

# Calcul de la précision
accuracy = accuracy_score(y_test, y_pred)
print(f"\n   PRÉCISION (Accuracy) : {accuracy * 100:.2f}%")

# Rapport de classification détaillé
print("\n   RAPPORT DE CLASSIFICATION :")
print(classification_report(y_test, y_pred, 
                          target_names=['Catégorie 0', 'Catégorie 1']))

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
print("\n   MATRICE DE CONFUSION :")
print(cm)

# Importance des features
print("\n   IMPORTANCE DES CARACTÉRISTIQUES :")
for i, importance in enumerate(rf_model.feature_importances_):
    print(f"   - Feature {i+1} : {importance:.4f}")

# ============================================================
# ÉTAPE 5 : VISUALISATION DES RÉSULTATS
# ============================================================
print("\n5. Génération des visualisations...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Graphique 1 : Données d'entraînement
axes[0].scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], 
               c='blue', alpha=0.6, s=50, label='Catégorie 0', edgecolors='k')
axes[0].scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], 
               c='red', alpha=0.6, s=50, label='Catégorie 1', edgecolors='k')
axes[0].set_title('Données d\'Entraînement', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Graphique 2 : Données de test avec prédictions
correct = y_test == y_pred
axes[1].scatter(X_test[correct & (y_test==0), 0], X_test[correct & (y_test==0), 1],
               c='blue', alpha=0.8, s=100, marker='o', label='Cat 0 (Correct)', edgecolors='k')
axes[1].scatter(X_test[correct & (y_test==1), 0], X_test[correct & (y_test==1), 1],
               c='red', alpha=0.8, s=100, marker='o', label='Cat 1 (Correct)', edgecolors='k')
axes[1].scatter(X_test[~correct, 0], X_test[~correct, 1],
               c='yellow', alpha=0.8, s=150, marker='X', label='Erreur', edgecolors='k', linewidths=2)
axes[1].set_title('Prédictions sur Données de Test', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Feature 1')
axes[1].set_ylabel('Feature 2')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Graphique 3 : Matrice de confusion
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2],
           xticklabels=['Cat 0', 'Cat 1'],
           yticklabels=['Cat 0', 'Cat 1'])
axes[2].set_title('Matrice de Confusion', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Prédiction')
axes[2].set_ylabel('Vérité')

plt.tight_layout()
plt.savefig('random_forest_results.png', dpi=150, bbox_inches='tight')
print("   ✓ Graphique sauvegardé : random_forest_results.png")
plt.show()

# ============================================================
# ÉTAPE 6 : TEST SUR DE NOUVELLES DONNÉES
# ============================================================
print("\n6. Test sur de nouvelles données...")

# Créer quelques nouveaux points pour tester
nouveaux_points = np.array([
    [1.5, 2.0],   # Devrait être Catégorie 0
    [8.5, 7.8],   # Devrait être Catégorie 1
    [2.2, 1.8],   # Devrait être Catégorie 0
    [7.5, 8.2]    # Devrait être Catégorie 1
])

predictions = rf_model.predict(nouveaux_points)
probas = rf_model.predict_proba(nouveaux_points)

print("\n   PRÉDICTIONS SUR NOUVEAUX POINTS :")
for i, (point, pred, proba) in enumerate(zip(nouveaux_points, predictions, probas)):
    print(f"   Point {i+1} {point} -> Catégorie {int(pred)} "
          f"(Confiance: {proba[int(pred)]*100:.1f}%)")

print("\n" + "="*60)
print("ENTRAÎNEMENT ET TEST TERMINÉS AVEC SUCCÈS !")
print("="*60)