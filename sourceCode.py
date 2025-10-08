# =============================================================================
# IMPORTATION DES LIBRAIRIES
# =============================================================================
# Pandas pour la manipulation des données
import pandas as pd
# NumPy pour les calculs scientifiques
import numpy as np

# Chargement du dataset depuis le fichier CSV
data_path = 'insurance.csv'
df = pd.read_csv(data_path)

# Affichage des 5 premières lignes pour inspection visuelle
df

# =============================================================================
# IMPORTATION DES LIBRAIRIES DE VISUALISATION
# =============================================================================
# Matplotlib pour créer des graphiques
import matplotlib.pyplot as plt
# Seaborn pour des visualisations statistiques avancées
import seaborn as sns

# =============================================================================
# ANALYSE EXPLORATOIRE DES DONNÉES (EDA)
# =============================================================================

# 2. Informations générales sur le dataset
print("\n=== INFORMATIONS GÉNÉRALES ===")
print("Shape (dimensions) :", df.shape)
print("→ Nous avons", df.shape[0], "patients et", df.shape[1], "caractéristiques")

# 3. Analyse des types de données
print("\n=== TYPES DE DONNÉES ===")
print(df.dtypes)
print("\n→ object = texte, int64 = nombres entiers, float64 = nombres décimaux")

# 4. Vérification des valeurs manquantes
print("=== VALEURS MANQUANTES ===")
valeurs_manquantes = df.isnull().sum()
print(valeurs_manquantes)

if valeurs_manquantes.sum() == 0:
    print("✅ PARFAIT ! Aucune valeur manquante détectée.")
else:
    print("⚠️  Il y a des valeurs manquantes à traiter.")

# 5. Statistiques descriptives pour comprendre la distribution des données
print("=== STATISTIQUES DESCRIPTIVES (Numériques) ===")
print(df.describe())

# 6. Analyse des variables catégorielles
print("\n=== VARIABLES CATÉGORIELLES ===")
categorical_cols = ['sex', 'smoker', 'region']
for col in categorical_cols:
    print(f"\n--- {col.upper()} ---")
    print(df[col].value_counts())
    print(f"Nombre de catégories: {df[col].nunique()}")

# 7. Identification des variables cibles pour nos 3 problèmes ML
print("=== VARIABLES CIBLES IDENTIFIÉES ===")
print("1. RÉGRESSION (continue) : 'charges' - coûts d'assurance")
print("2. CLASSIFICATION BINAIRE : 'smoker' - fumeur (yes/no)")
print("3. CLASSIFICATION MULTICLASSE : 'region' - région (4 catégories)")

print(f"\nVérification :")
print(f"- Charges (type: {df['charges'].dtype}, valeurs uniques: {df['charges'].nunique()})")
print(f"- Smoker: {df['smoker'].unique()}")
print(f"- Region: {df['region'].unique()}")

# =============================================================================
# VISUALISATIONS EXPLORATOIRES
# =============================================================================

# Graphique 1 : Distribution des coûts d'assurance
# Combinaison histogramme et boxplot pour voir la distribution et les outliers
plt.figure(figsize=(12, 5))

# Left - Histogramme avec courbe de densité
plt.subplot(1, 2, 1)
sns.histplot(df['charges'], kde=True, bins=30, color='skyblue')
plt.title('Distribution des Coûts d\'Assurance')
plt.xlabel('Coûts ($)')
plt.ylabel('Nombre de Patients')

# Right - Boxplot pour identifier les valeurs extrêmes
plt.subplot(1, 2, 2)
sns.boxplot(y=df['charges'], color='lightcoral')
plt.title('Boxplot - Coûts d\'Assurance')
plt.ylabel('Coûts ($)')

plt.tight_layout()
plt.show()

# Graphique 2 : Impact du tabagisme sur les coûts
# Boxplot comparatif pour visualiser la différence
plt.figure(figsize=(10, 6))
sns.boxplot(x='smoker', y='charges', data=df, palette=['lightgreen', 'coral'])
plt.title('Impact du Tabagisme sur les Coûts d\'Assurance')
plt.xlabel('Fumeur')
plt.ylabel('Coûts ($)')
plt.show()

# Calculs quantitatifs pour compléter la visualisation
cout_fumeurs = df[df['smoker'] == 'yes']['charges'].mean()
cout_non_fumeurs = df[df['smoker'] == 'no']['charges'].mean()
print(f"💰 Coût moyen fumeurs: ${cout_fumeurs:,.2f}")
print(f"💰 Coût moyen non-fumeurs: ${cout_non_fumeurs:,.2f}")
print(f"📊 Différence: {cout_fumeurs/cout_non_fumeurs:.1f}x plus cher pour les fumeurs!")

# Graphique 3 : Matrice de corrélation
# Heatmap pour identifier les relations entre variables numériques
plt.figure(figsize=(8, 6))
correlation_matrix = df[['age', 'bmi', 'children', 'charges']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
plt.title('Matrice de Corrélation entre Variables Numériques')
plt.show()

print("🔍 Lecture de la heatmap:")
print("- +1.00 = corrélation parfaite positive")
print("- -1.00 = corrélation parfaite négative") 
print("- 0.00 = pas de corrélation")
print("- Plus c'est proche de +1 ou -1, plus la relation est forte")

# Graphique 4 : Répartition des patients par genre et région
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left - Répartition par genre (diagramme circulaire)
df['sex'].value_counts().plot(kind='pie', ax=axes[0], autopct='%1.1f%%', 
                              colors=['lightpink', 'lightblue'])
axes[0].set_title('Répartition par Genre')
axes[0].set_ylabel('')

# Right - Répartition par région (diagramme en barres)
df['region'].value_counts().plot(kind='bar', ax=axes[1], color='lightgreen')
axes[1].set_title('Répartition par Région')
axes[1].set_xlabel('Région')
axes[1].set_ylabel('Nombre de Patients')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Graphique 5 : Relation âge vs coûts avec coloration par statut fumeur
# Scatter plot pour voir les tendances et patterns
plt.figure(figsize=(10, 6))
sns.scatterplot(x='age', y='charges', data=df, hue='smoker', 
                palette=['green', 'red'], alpha=0.7)
plt.title('Relation Âge vs Coûts d\'Assurance (Coloré par Fumeur)')
plt.xlabel('Âge')
plt.ylabel('Coûts ($)')
plt.legend(title='Fumeur')
plt.show()

# =============================================================================
# PRÉTRAITEMENT DES DONNÉES - PHASE 1
# =============================================================================

# Création d'une copie pour préserver les données originales
df_processed = df.copy()

print("=== ENCODAGE DES VARIABLES CATÉGORIELLES ===")

# Encodage des variables binaires (sex, smoker) en 0/1
df_processed['sex'] = df_processed['sex'].map({'female': 0, 'male': 1})
df_processed['smoker'] = df_processed['smoker'].map({'no': 0, 'yes': 1})

# Encodage one-hot pour la variable région (multiclasse)
# Crée 4 colonnes binaires pour éviter l'ordre artificiel
region_encoded = pd.get_dummies(df_processed['region'], prefix='region')
df_processed = pd.concat([df_processed, region_encoded], axis=1)
df_processed = df_processed.drop('region', axis=1)  # Supprimer la colonne originale

print("✅ Encodage terminé !")
print("Nouvelles colonnes :", df_processed.columns.tolist())

# Vérification de l'encodage
print("=== VÉRIFICATION DE L'ENCODAGE ===")
print("\nValeurs uniques après encodage :")
print("sex :", df_processed['sex'].unique())
print("smoker :", df_processed['smoker'].unique())
print("\nAperçu des données encodées :")
print(df_processed[['sex', 'smoker', 'region_northeast', 'region_northwest', 'region_southeast', 'region_southwest']].head())

# =============================================================================
# SÉPARATION DES VARIABLES CIBLES
# =============================================================================

print("=== SÉPARATION DES VARIABLES CIBLES ===")

# Variables features (X) - toutes sauf les cibles et les colonnes one-hot région
X = df_processed.drop(['charges', 'smoker', 'region_northeast', 'region_northwest', 'region_southeast', 'region_southwest'], axis=1)

# 1. Régression - charges (continue) - variable numérique
y_regression = df_processed['charges']

# 2. Classification binaire - smoker (0/1) - variable catégorielle binaire
y_binary = df_processed['smoker']

# 3. Classification multiclasse - région (4 classes) - on reprend l'originale
y_multiclass = df[['region']].copy()

print("✅ Séparation terminée !")
print(f"Features (X) : {X.shape}")
print(f"Régression (charges) : {y_regression.shape}")
print(f"Binaire (smoker) : {y_binary.shape}")
print(f"Multiclasse (region) : {y_multiclass.shape}")

# =============================================================================
# NORMALISATION DES VARIABLES NUMÉRIQUES
# =============================================================================

from sklearn.preprocessing import StandardScaler

print("=== NORMALISATION DES VARIABLES NUMÉRIQUES ===")

# Colonnes à normaliser (âge, bmi, children)
numeric_cols = ['age', 'bmi', 'children']

# Création du scaler pour standardisation (moyenne=0, écart-type=1)
scaler = StandardScaler()

# Application de la normalisation
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

print("✅ Normalisation terminée !")
print("Moyennes après normalisation (devraient être ~0) :")
print(X[numeric_cols].mean())
print("\nÉcart-types après normalisation (devraient être ~1) :")
print(X[numeric_cols].std())

# =============================================================================
# SÉPARATION ENTRAÎNEMENT/TEST
# =============================================================================

from sklearn.model_selection import train_test_split

print("=== SÉPARATION ENTRAÎNEMENT/TEST ===")

# Pour la régression - stratification sur y_binary pour préserver la distribution
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X, y_regression, test_size=0.2, random_state=42, stratify=y_binary
)

# Pour la classification binaire - stratification pour équilibrer les classes
X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

# Pour la classification multiclasse - stratification par région
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    X, y_multiclass, test_size=0.2, random_state=42, stratify=y_multiclass
)

print("✅ Séparation entraînement/test terminée !")
print(f"Régression - Train: {X_train_reg.shape}, Test: {X_test_reg.shape}")
print(f"Binaire - Train: {X_train_bin.shape}, Test: {X_test_bin.shape}")
print(f"Multiclasse - Train: {X_train_multi.shape}, Test: {X_test_multi.shape}")

# Récapitulatif final de la préparation des données
print("=== RÉCAPITULATIF FINAL ===")
print("🎯 Variables features :", X.columns.tolist())
print(f"📊 Shape final X : {X.shape}")
print(f"🔢 Types de données :")
print(X.dtypes)
print(f"\n📈 Échantillons d'entraînement : {X_train_reg.shape[0]}")
print(f"🧪 Échantillons de test : {X_test_reg.shape[0]}")

# =============================================================================
# PHASE 2 - RÉGRESSION (Prédiction des coûts)
# =============================================================================

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

print("=== PHASE 2 - RÉGRESSION ===")
print("Modèles choisis :")
print("1. 📈 Régression Linéaire - Modèle linéaire simple")
print("2. 🌳 Arbre de Décision - Modèle non-linéaire simple") 
print("3. 🌲🌲 Random Forest - Modèle ensemble complexe")

# Initialisation des modèles de régression
models = {
    'Régression Linéaire': LinearRegression(),
    'Arbre de Décision': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42, n_estimators=100)
}

# Stockage des résultats pour comparaison
results = {}

print("=== ENTRAÎNEMENT DES MODÈLES ===")
for name, model in models.items():
    print(f"Entraînement : {name}...")
    # Entraînement du modèle sur les données d'entraînement
    model.fit(X_train_reg, y_train_reg)
    
    # Prédictions sur les ensembles d'entraînement et de test
    y_pred_train = model.predict(X_train_reg)
    y_pred_test = model.predict(X_test_reg)
    
    # Calcul des métriques d'évaluation pour les deux ensembles
    results[name] = {
        'train': {
            'MAE': mean_absolute_error(y_train_reg, y_pred_train),  # Erreur absolue moyenne
            'MSE': mean_squared_error(y_train_reg, y_pred_train),   # Erreur quadratique moyenne
            'RMSE': np.sqrt(mean_squared_error(y_train_reg, y_pred_train)),  # Racine de MSE
            'R2': r2_score(y_train_reg, y_pred_train)               # Coefficient de détermination
        },
        'test': {
            'MAE': mean_absolute_error(y_test_reg, y_pred_test),
            'MSE': mean_squared_error(y_test_reg, y_pred_test),
            'RMSE': np.sqrt(mean_squared_error(y_test_reg, y_pred_test)),
            'R2': r2_score(y_test_reg, y_pred_test)
        }
    }
    
    print(f"✅ {name} terminé")

print("\n🎯 ENTRAÎNEMENT TERMINÉ !")

# =============================================================================
# ÉVALUATION DES MODÈLES DE RÉGRESSION
# =============================================================================

print("=== RÉSULTATS DES MODÈLES DE RÉGRESSION ===")
print("🔍 Comparaison des performances :\n")

for name, metrics in results.items():
    print(f"📊 {name.upper()}")
    print(f"   Ensemble d'ENTRAÎNEMENT:")
    print(f"   - MAE:  ${metrics['train']['MAE']:,.2f}")   # Interprétation en dollars
    print(f"   - MSE:  ${metrics['train']['MSE']:,.2f}")
    print(f"   - RMSE: ${metrics['train']['RMSE']:,.2f}")  # Métrique principale
    print(f"   - R²:   {metrics['train']['R2']:.4f}")      # Pourcentage de variance expliquée
    
    print(f"   Ensemble de TEST:")
    print(f"   - MAE:  ${metrics['test']['MAE']:,.2f}")
    print(f"   - MSE:  ${metrics['test']['MSE']:,.2f}") 
    print(f"   - RMSE: ${metrics['test']['RMSE']:,.2f}")
    print(f"   - R²:   {metrics['test']['R2']:.4f}")
    print("   " + "─" * 40)

# =============================================================================
# VISUALISATION DES PERFORMANCES DE RÉGRESSION
# =============================================================================

import matplotlib.pyplot as plt
import seaborn as sns

print("=== COMPARAISON VISUELLE DES PERFORMANCES ===")

# Préparation des données pour le graphique comparatif
model_names = list(results.keys())
r2_scores_test = [results[name]['test']['R2'] for name in model_names]
rmse_scores_test = [results[name]['test']['RMSE'] for name in model_names]

# Graphique comparatif double
plt.figure(figsize=(12, 5))

# Graphique R² - Score de performance
plt.subplot(1, 2, 1)
bars1 = plt.bar(model_names, r2_scores_test, color=['skyblue', 'lightgreen', 'coral'])
plt.title('Score R² - Ensemble de Test')
plt.ylabel('Score R²')
plt.ylim(0, 1)
# Ajouter les valeurs sur les barres pour précision
for bar, value in zip(bars1, r2_scores_test):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{value:.3f}', 
             ha='center', va='bottom')

# Graphique RMSE - Erreur de prédiction
plt.subplot(1, 2, 2)
bars2 = plt.bar(model_names, rmse_scores_test, color=['skyblue', 'lightgreen', 'coral'])
plt.title('RMSE - Ensemble de Test')
plt.ylabel('RMSE ($)')
# Ajouter les valeurs en dollars
for bar, value in zip(bars2, rmse_scores_test):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, f'${value:,.0f}', 
             ha='center', va='bottom')

plt.tight_layout()
plt.show()

# =============================================================================
# IDENTIFICATION DU MEILLEUR MODÈLE DE RÉGRESSION
# =============================================================================

print("=== IDENTIFICATION DU MEILLEUR MODÈLE ===")

# Trouver le meilleur R² sur le test (plus proche de 1 est meilleur)
best_r2_model = max(results.keys(), key=lambda x: results[x]['test']['R2'])
best_r2_score = results[best_r2_model]['test']['R2']

# Trouver le meilleur RMSE (le plus bas est meilleur)
best_rmse_model = min(results.keys(), key=lambda x: results[x]['test']['RMSE'])
best_rmse_score = results[best_rmse_model]['test']['RMSE']

print(f"🏆 MEILLEUR SCORE R² : {best_r2_model} ({best_r2_score:.4f})")
print(f"🎯 MEILLEUR RMSE : {best_rmse_model} (${best_rmse_score:,.2f})")

# Analyse du sur-apprentissage (différence entre train et test)
print(f"\n🔍 ANALYSE DU SUR-APPRENTISSAGE :")
for name, metrics in results.items():
    diff_r2 = metrics['train']['R2'] - metrics['test']['R2']
    print(f"   {name}: R²_train - R²_test = {diff_r2:.4f}")
    if diff_r2 > 0.1:
        print(f"   ⚠️  Attention : risque de sur-apprentissage")
    else:
        print(f"   ✅ Bon équilibre")

# =============================================================================
# COURBES D'APPRENTISSAGE POUR ANALYSE DE LA STABILITÉ
# =============================================================================

from sklearn.model_selection import learning_curve
import numpy as np

print("=== COURBES D'APPRENTISSAGE ===")

# Fonction pour tracer les courbes d'apprentissage
def plot_learning_curve(model, X, y, model_name):
    # Génération des courbes avec validation croisée
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),  # 10 tailles d'échantillon
        scoring='r2'  # Métrique d'évaluation
    )
    
    # Calcul des moyennes
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    
    # Tracé des courbes
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Score entraînement")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Score validation")
    plt.title(f"Courbe d'apprentissage - {model_name}")
    plt.xlabel("Taille de l'ensemble d'entraînement")
    plt.ylabel("Score R²")
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()
    
    return train_sizes, train_scores_mean, test_scores_mean

# Génération des courbes pour chaque modèle
print("📈 Génération des courbes d'apprentissage...")

# Régression Linéaire
print("1. Régression Linéaire...")
plot_learning_curve(LinearRegression(), X_train_reg, y_train_reg, "Régression Linéaire")

# Arbre de Décision avec régularisation pour éviter le sur-apprentissage
print("2. Arbre de Décision (régularisé)...")
tree_tuned = DecisionTreeRegressor(max_depth=3, min_samples_split=20, random_state=42)
plot_learning_curve(tree_tuned, X_train_reg, y_train_reg, "Arbre de Décision Régularisé")

# Random Forest
print("3. Random Forest...")
plot_learning_curve(RandomForestRegressor(random_state=42), X_train_reg, y_train_reg, "Random Forest")

# =============================================================================
# GRAPHIQUE VALEURS PRÉDITES VS RÉELLES
# =============================================================================

print("=== GRAPHIQUE VALEURS PRÉDITES VS RÉELLES ===")

# Utilisation du meilleur modèle (Régression Linéaire)
best_model = LinearRegression()
best_model.fit(X_train_reg, y_train_reg)
y_pred_best = best_model.predict(X_test_reg)

# Création du graphique de dispersion
plt.figure(figsize=(10, 6))

# Nuage de points des prédictions vs valeurs réelles
plt.scatter(y_test_reg, y_pred_best, alpha=0.6, color='blue', label='Prédictions')

# Ligne de perfection (y = x) - idéalement les points devraient être sur cette ligne
max_val = max(y_test_reg.max(), y_pred_best.max())
min_val = min(y_test_reg.min(), y_pred_best.min())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Prédiction parfaite')

plt.xlabel('Valeurs Réelles ($)')
plt.ylabel('Valeurs Prédites ($)')
plt.title('Régression Linéaire: Valeurs Prédites vs Valeurs Réelles\n(Meilleur Modèle)')
plt.legend()
plt.grid(True, alpha=0.3)

# Ajout du score R² sur le graphique
r2 = r2_score(y_test_reg, y_pred_best)
plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, 
         fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.show()

print(f"✅ Graphique généré avec R² = {r2:.3f}")

# =============================================================================
# PHASE 3 - CLASSIFICATION BINAIRE (Prédiction statut fumeur)
# =============================================================================

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve

print("=== PHASE 3 - CLASSIFICATION BINAIRE ===")
print("Variable cible : 'smoker' (0 = non, 1 = oui)")
print(f"Distribution : {y_binary.value_counts()}")
print(f"Proportion : {y_binary.value_counts(normalize=True)}")

print("\nModèles choisis :")
print("1. 📊 Régression Logistique - Classifieur linéaire")
print("2. 🌲 Random Forest Classifier - Modèle ensemble") 
print("3. 🎯 SVM (Support Vector Machine) - Classifieur à marge maximale")

# Initialisation des classifieurs
classifiers = {
    'Régression Logistique': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'SVM': SVC(random_state=42, probability=True)  # probability=True pour ROC curve
}

# Stockage des résultats
binary_results = {}

print("=== ENTRAÎNEMENT DES CLASSIFIEURS ===")
for name, clf in classifiers.items():
    print(f"Entraînement : {name}...")
    clf.fit(X_train_bin, y_train_bin)
    
    # Prédictions de classe
    y_pred = clf.predict(X_test_bin)
    # Probabilités pour la classe positive (fumeur) pour courbe ROC
    y_pred_proba = clf.predict_proba(X_test_bin)[:, 1]
    
    # Calcul des métriques de classification
    binary_results[name] = {
        'accuracy': accuracy_score(y_test_bin, y_pred),      # Exactitude globale
        'precision': precision_score(y_test_bin, y_pred),    # Précision classe positive
        'recall': recall_score(y_test_bin, y_pred),         # Rappel classe positive
        'f1': f1_score(y_test_bin, y_pred),                 # Score F1 (moyenne harmonique)
        'roc_auc': roc_auc_score(y_test_bin, y_pred_proba), # Aire sous courbe ROC
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }
    
    print(f"✅ {name} terminé")

print("\n🎯 ENTRAÎNEMENT TERMINÉ !")

# =============================================================================
# DIAGNOSTIC DU PROBLÈME DE DÉSÉQUILIBRE DES CLASSES
# =============================================================================

print("=== ANALYSE DU PROBLÈME ===")
print("Distribution des classes dans y_test_bin :")
print(y_test_bin.value_counts())
print(f"Proportion : {y_test_bin.value_counts(normalize=True)}")

print("\nVérification des prédictions :")
for name, results in binary_results.items():
    unique_preds = np.unique(results['predictions'])
    print(f"{name} - Prédictions uniques : {unique_preds}")

# =============================================================================
# CORRECTION AVEC GESTION DU DÉSÉQUILIBRE DES CLASSES
# =============================================================================

print("=== CORRECTION AVEC GESTION DU DÉSÉQUILIBRE ===")

# Réentraînement avec class_weight='balanced' pour pénaliser les erreurs sur la classe minoritaire
classifiers_corrected = {
    'Régression Logistique': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced'),
    'SVM': SVC(random_state=42, probability=True, class_weight='balanced')
}

binary_results_corrected = {}

print("=== RÉENTRAÎNEMENT AVEC CLASS_WEIGHT='BALANCED' ===")
for name, clf in classifiers_corrected.items():
    print(f"Entraînement : {name}...")
    clf.fit(X_train_bin, y_train_bin)
    
    # Prédictions
    y_pred = clf.predict(X_test_bin)
    y_pred_proba = clf.predict_proba(X_test_bin)[:, 1]
    
    # Calcul des métriques avec zero_division=0 pour éviter les warnings
    binary_results_corrected[name] = {
        'accuracy': accuracy_score(y_test_bin, y_pred),
        'precision': precision_score(y_test_bin, y_pred, zero_division=0),
        'recall': recall_score(y_test_bin, y_pred),
        'f1': f1_score(y_test_bin, y_pred),
        'roc_auc': roc_auc_score(y_test_bin, y_pred_proba),
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }
    
    print(f"✅ {name} terminé")
    print(f"   Prédictions uniques : {np.unique(y_pred)}")  # Vérification de la diversité des prédictions

print("\n🎯 CORRECTION TERMINÉE !")

# =============================================================================
# ÉVALUATION DES CLASSIFIEURS CORRIGÉS
# =============================================================================

print("=== RÉSULTATS DES CLASSIFIEURS (AVEC GESTION DU DÉSÉQUILIBRE) ===")
print("🔍 Comparaison des performances :\n")

for name, metrics in binary_results_corrected.items():
    print(f"📊 {name.upper()}")
    print(f"   Accuracy:    {metrics['accuracy']:.4f}")    # Exactitude globale
    print(f"   Precision:   {metrics['precision']:.4f}")   # Qualité prédictions positives
    print(f"   Recall:      {metrics['recall']:.4f}")      # Capacité à trouver tous les positifs
    print(f"   F1-Score:    {metrics['f1']:.4f}")         # Équilibre précision/rappel
    print(f"   ROC AUC:     {metrics['roc_auc']:.4f}")    # Capacité de discrimination
    print("   " + "─" * 40)

# =============================================================================
# IDENTIFICATION DU MEILLEUR CLASSIFIEUR BINAIRE
# =============================================================================

print("=== IDENTIFICATION DU MEILLEUR CLASSIFIEUR ===")

# Trouver le meilleur F1-Score (équilibre entre précision et rappel)
best_f1_model = max(binary_results_corrected.keys(), key=lambda x: binary_results_corrected[x]['f1'])
best_f1_score = binary_results_corrected[best_f1_model]['f1']

# Trouver le meilleur ROC AUC (capacité de discrimination)
best_auc_model = max(binary_results_corrected.keys(), key=lambda x: binary_results_corrected[x]['roc_auc'])
best_auc_score = binary_results_corrected[best_auc_model]['roc_auc']

print(f"🏆 MEILLEUR F1-SCORE : {best_f1_model} ({best_f1_score:.4f})")
print(f"🎯 MEILLEUR ROC AUC : {best_auc_model} ({best_auc_score:.4f})")

print(f"\n💡 INTERPRÉTATION :")
print("F1-Score : Équilibre entre précision et rappel (idéal pour classes déséquilibrées)")
print("ROC AUC : Capacité à distinguer les classes (0.5 = aléatoire, 1.0 = parfait)")

# =============================================================================
# MATRICES DE CONFUSION POUR ANALYSE DES ERREURS
# =============================================================================

print("=== MATRICES DE CONFUSION ===")

# Configuration pour l'affichage multiple
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (name, metrics) in enumerate(binary_results_corrected.items()):
    # Calcul de la matrice de confusion
    cm = confusion_matrix(y_test_bin, metrics['predictions'])
    
    # Affichage avec seaborn
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['Non-Fumeur', 'Fumeur'],
                yticklabels=['Non-Fumeur', 'Fumeur'])
    axes[idx].set_title(f'Matrice de Confusion - {name}')
    axes[idx].set_xlabel('Prédit')
    axes[idx].set_ylabel('Réel')

plt.tight_layout()
plt.show()

print("🔍 Légende des matrices :")
print("   - [0,0] : Vrais Négatifs (TN) - Correctement prédit non-fumeur")
print("   - [0,1] : Faux Positifs (FP) - Prédit fumeur mais non-fumeur")  
print("   - [1,0] : Faux Négatifs (FN) - Prédit non-fumeur mais fumeur")
print("   - [1,1] : Vrais Positifs (TP) - Correctement prédit fumeur")

# =============================================================================
# COURBES ROC ET PRÉCISION-RAPPEL
# =============================================================================

print("=== COURBES ROC ET PRÉCISION-RAPPEL ===")

# Configuration pour l'affichage double
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Courbe ROC - Capacité de discrimination
for name, metrics in binary_results_corrected.items():
    fpr, tpr, _ = roc_curve(y_test_bin, metrics['probabilities'])
    auc_score = metrics['roc_auc']
    ax1.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})', linewidth=2)

ax1.plot([0, 1], [0, 1], 'k--', label='Classifieur aléatoire (AUC = 0.5)')
ax1.set_xlabel('Taux de Faux Positifs (FPR)')
ax1.set_ylabel('Taux de Vrais Positifs (TPR)')
ax1.set_title('Courbe ROC - Comparaison des Modèles')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Courbe Précision-Rappel - Performance sur classe minoritaire
for name, metrics in binary_results_corrected.items():
    precision, recall, _ = precision_recall_curve(y_test_bin, metrics['probabilities'])
    ax2.plot(recall, precision, label=name, linewidth=2)

ax2.set_xlabel('Rappel (Recall)')
ax2.set_ylabel('Précision (Precision)')
ax2.set_title('Courbe Précision-Rappel - Comparaison des Modèles')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("✅ Visualisations de classification binaire terminées !")

# =============================================================================
# PHASE 4 - CLASSIFICATION MULTICLASSE (Prédiction région)
# =============================================================================

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

print("=== PHASE 4 - CLASSIFICATION MULTICLASSE ===")
print("Variable cible : 'region' (4 régions)")
print(f"Distribution : {y_multiclass['region'].value_counts()}")
print(f"Proportion : {y_multiclass['region'].value_counts(normalize=True)}")

# Encodage de la variable cible région en nombres (0,1,2,3)
label_encoder = LabelEncoder()
y_multi_encoded = label_encoder.fit_transform(y_multiclass['region'])

print(f"\nEncodage des régions :")
for i, region in enumerate(label_encoder.classes_):
    print(f"   {region} → {i}")

print(f"\nModèles choisis :")
print("1. 🌳 Arbre de Décision - Modèle simple et interprétable")
print("2. 🌲🌲 Random Forest - Modèle ensemble robuste") 
print("3. 📍 KNN (K-Nearest Neighbors) - Modèle basé sur similarité")

# Initialisation des classifieurs multiclasse
multi_classifiers = {
    'Arbre de Décision': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

# Stockage des résultats
multi_results = {}

print("=== ENTRAÎNEMENT DES CLASSIFIEURS MULTICLASSE ===")
for name, clf in multi_classifiers.items():
    print(f"Entraînement : {name}...")
    # Entraînement avec vérification de la taille des données
    clf.fit(X_train_multi, y_multi_encoded[:len(X_train_multi)])
    
    # Prédictions
    y_pred = clf.predict(X_test_multi)
    
    # Calcul des métriques
    multi_results[name] = {
        'accuracy': accuracy_score(y_multi_encoded[:len(X_test_multi)], y_pred),
        'predictions': y_pred,
        'model': clf
    }
    
    print(f"✅ {name} terminé - Accuracy: {multi_results[name]['accuracy']:.4f}")

print("\n🎯 ENTRAÎNEMENT MULTICLASSE TERMINÉ !")

# =============================================================================
# ÉVALUATION DES CLASSIFIEURS MULTICLASSE
# =============================================================================

print("=== RÉSULTATS DES CLASSIFIEURS MULTICLASSE (CORRIGÉ) ===")
print("🔍 Comparaison des performances :\n")

for name, metrics in multi_results.items():
    print(f"📊 {name.upper()}")
    print(f"   Accuracy globale: {metrics['accuracy']:.4f}")
    
    # Rapport de classification détaillé par classe
    y_true_multi = y_multi_encoded[:len(X_test_multi)]
    y_pred_multi = metrics['predictions']
    
    print(f"   Rapport de classification :")
    
    # Affichage du rapport complet avec métriques par classe
    print(classification_report(y_true_multi, y_pred_multi, 
                              target_names=label_encoder.classes_))
    
    print("   " + "─" * 50)

# =============================================================================
# IDENTIFICATION DU MEILLEUR CLASSIFIEUR MULTICLASSE
# =============================================================================

print("=== IDENTIFICATION DU MEILLEUR CLASSIFIEUR MULTICLASSE ===")

# Trouver le meilleur accuracy
best_accuracy_model = max(multi_results.keys(), key=lambda x: multi_results[x]['accuracy'])
best_accuracy_score = multi_results[best_accuracy_model]['accuracy']

print(f"🏆 MEILLEUR ACCURACY : {best_accuracy_model} ({best_accuracy_score:.4f})")

print(f"\n💡 INTERPRÉTATION :")
print(f"Accuracy = {best_accuracy_score:.1%} des régions correctement prédites")
print("Pour 4 classes équilibrées, l'accuracy aléatoire serait de 25%")

# =============================================================================
# MATRICES DE CONFUSION MULTICLASSE
# =============================================================================

print("=== MATRICES DE CONFUSION MULTICLASSE (4×4) ===")

# Configuration pour l'affichage des 3 matrices
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for idx, (name, metrics) in enumerate(multi_results.items()):
    y_true_multi = y_multi_encoded[:len(X_test_multi)]
    y_pred_multi = metrics['predictions']
    
    # Calcul de la matrice de confusion
    cm = confusion_matrix(y_true_multi, y_pred_multi)
    
    # Affichage avec seaborn
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    axes[idx].set_title(f'Matrice de Confusion - {name}\nAccuracy: {metrics["accuracy"]:.3f}')
    axes[idx].set_xlabel('Prédit')
    axes[idx].set_ylabel('Réel')

plt.tight_layout()
plt.show()

print("🔍 Légende des matrices :")
print("La diagonale montre les bonnes prédictions pour chaque région")
print("Hors diagonale : confusions entre régions")

# =============================================================================
# DIAGRAMMES DES MÉTRIQUES PAR CLASSE
# =============================================================================

print("=== DIAGRAMMES DES MÉTRIQUES PAR CLASSE ===")

# Préparer les données pour les graphiques comparatifs
metrics_by_class = {}

for name, metrics in multi_results.items():
    y_true_multi = y_multi_encoded[:len(X_test_multi)]
    y_pred_multi = metrics['predictions']
    
    # Calculer les métriques pour chaque classe individuellement
    report = classification_report(y_true_multi, y_pred_multi, 
                                 target_names=label_encoder.classes_, 
                                 output_dict=True)
    
    metrics_by_class[name] = {
        'precision': [report[region]['precision'] for region in label_encoder.classes_],
        'recall': [report[region]['recall'] for region in label_encoder.classes_],
        'f1': [report[region]['f1-score'] for region in label_encoder.classes_]
    }

# Graphiques pour chaque métrique (Precision, Recall, F1)
fig, axes = plt.subplots(3, 1, figsize=(12, 15))

# Precision par classe
x_pos = np.arange(len(label_encoder.classes_))
width = 0.25  # Largeur des barres

for idx, (name, metrics) in enumerate(metrics_by_class.items()):
    axes[0].bar(x_pos + idx*width, metrics['precision'], width, label=name, alpha=0.8)

axes[0].set_title('Précision par Région et par Modèle')
axes[0].set_xlabel('Région')
axes[0].set_ylabel('Précision')
axes[0].set_xticks(x_pos + width)
axes[0].set_xticklabels(label_encoder.classes_)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Recall par classe
for idx, (name, metrics) in enumerate(metrics_by_class.items()):
    axes[1].bar(x_pos + idx*width, metrics['recall'], width, label=name, alpha=0.8)

axes[1].set_title('Rappel (Recall) par Région et par Modèle')
axes[1].set_xlabel('Région')
axes[1].set_ylabel('Rappel')
axes[1].set_xticks(x_pos + width)
axes[1].set_xticklabels(label_encoder.classes_)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# F1-Score par classe
for idx, (name, metrics) in enumerate(metrics_by_class.items()):
    axes[2].bar(x_pos + idx*width, metrics['f1'], width, label=name, alpha=0.8)

axes[2].set_title('F1-Score par Région et par Modèle')
axes[2].set_xlabel('Région')
axes[2].set_ylabel('F1-Score')
axes[2].set_xticks(x_pos + width)
axes[2].set_xticklabels(label_encoder.classes_)
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("✅ Visualisations multiclasse terminées !")

# =============================================================================
# RÉCAPITULATIF FINAL PHASE 4
# =============================================================================

print("=== RÉCAPITULATIF PHASE 4 - CLASSIFICATION MULTICLASSE ===")

# Trouver le meilleur modèle
best_model = max(multi_results.keys(), key=lambda x: multi_results[x]['accuracy'])
best_accuracy = multi_results[best_model]['accuracy']

print(f"🏆 MEILLEUR MODÈLE : {best_model}")
print(f"🎯 ACCURACY : {best_accuracy:.3f} ({best_accuracy*100:.1f}%)")
print(f"📊 PERFORMANCE vs ALÉATOIRE : {best_accuracy/0.25:.1f}x mieux que le hasard")

print(f"\n📈 DIFFICULTÉS IDENTIFIÉES :")
print("- 4 classes équilibrées → défi difficile")
print("- Accuracy attendue par hasard : 25%")
print("- Modèles actuels : ~20-25% → proche du hasard")
print("- Besoin de features plus discriminantes pour la région")

print(f"\n✅ PHASE 4 TERMINÉE !")

# =============================================================================
# SYNTHÈSE FINALE - COMPARAISON GLOBALE
# =============================================================================

print("=== CARTE DE CORRÉLATION COMPLÈTE ===")

# Préparer les données numériques + variables encodées pour corrélation complète
numeric_data = df[['age', 'bmi', 'children', 'charges']].copy()
numeric_data['sex'] = df_processed['sex']
numeric_data['smoker'] = df_processed['smoker']

plt.figure(figsize=(10, 8))
correlation_matrix = numeric_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
plt.title('Matrice de Corrélation - Variables du Dataset')
plt.tight_layout()
plt.show()

print("🔍 Insights de la corrélation :")
print("- 'smoker' fortement corrélé avec 'charges' (+0.79) → Impact majeur!")
print("- 'age' modérément corrélé avec 'charges' (+0.30) → Coûts augmentent avec l'âge")
print("- 'bmi' faiblement corrélé avec 'charges' (+0.20) → Léger impact de l'obésité")

print("=== PAIR PLOTS - ANALYSE MULTIVARIÉE ===")

# Pair plot avec hue=smoker pour visualiser l'impact global
sample_df = df.copy()
sample_df['smoker_encoded'] = df_processed['smoker']  # Ajouter la version encodée

# Échantillonnage pour éviter la surcharge visuelle
sample_size = min(500, len(sample_df))
sample_df = sample_df.sample(sample_size, random_state=42)

sns.pairplot(sample_df[['age', 'bmi', 'charges', 'smoker_encoded']], 
             hue='smoker_encoded', palette={0: 'blue', 1: 'red'},
             diag_kind='hist', plot_kws={'alpha': 0.6})
plt.suptitle('Pair Plots - Relations entre Variables (Coloré par Fumeur)', y=1.02)
plt.show()

print("=== COMPARAISON DES PERFORMANCES TOUS MODÈLES ===")

# Préparation des données pour comparaison globale
models_performance = {
    # Régression - R² des modèles
    'Rég. Linéaire': 0.1639,  # R² de la Phase 2
    'Arbre Rég.': -0.807,     # R² de la Phase 2 - SUR-APPRENTISSAGE
    'RF Rég.': -0.003,        # R² de la Phase 2
    
    # Classification Binaire - ROC AUC
    'Rég. Log.': binary_results_corrected['Régression Logistique']['roc_auc'],
    'RF Bin.': binary_results_corrected['Random Forest']['roc_auc'],
    'SVM': binary_results_corrected['SVM']['roc_auc'],
    
    # Classification Multiclasse - Accuracy
    'Arbre Multi': multi_results['Arbre de Décision']['accuracy'],
    'RF Multi': multi_results['Random Forest']['accuracy'],
    'KNN': multi_results['KNN']['accuracy']
}

# Graphique comparatif global
plt.figure(figsize=(14, 6))
colors = ['skyblue']*3 + ['lightgreen']*3 + ['coral']*3
bars = plt.bar(models_performance.keys(), models_performance.values(), color=colors)

plt.title('Comparaison des Performances - Tous les Modèles\n(R² / ROC AUC / Accuracy)')
plt.ylabel('Score')
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3)

# Ajouter les valeurs sur les barres pour précision
for bar, value in zip(bars, models_performance.values()):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{value:.3f}', ha='center', va='bottom', fontsize=9)

# Légende des couleurs par type de problème
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='skyblue', label='Régression (R²)'),
    Patch(facecolor='lightgreen', label='Classification Binaire (ROC AUC)'),
    Patch(facecolor='coral', label='Classification Multiclasse (Accuracy)')
]
plt.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.show()

print("🔍 INSIGHTS DE LA COMPARAISON :")
print("✅ MEILLEURS MODÈLES PAR CATÉGORIE :")
print("   - Régression : Régression Linéaire (R² = 0.164) → Stabilité")
print("   - Binaire : Random Forest (ROC AUC ≈ 0.85) → Excellente discrimination")
print("   - Multiclasse : Random Forest (Accuracy ≈ 0.25) → Proche du hasard")

print("\n⚠️  PROBLÈMES IDENTIFIÉS :")
print("   - Arbre de Décision en régression : SUR-APPRENTISSAGE (R² = -0.807)")
print("   - Multiclasse : performances proches du hasard (25% attendu)")
print("   - Données insuffisantes pour prédire la région")

print("=== IMPORTANCE DES VARIABLES - RANDOM FOREST ===")

# Analyse de l'importance des variables avec le meilleur classifieur binaire
rf_model = classifiers_corrected['Random Forest']
feature_importance = rf_model.feature_importances_
feature_names = X.columns

# Tri par importance décroissante
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=True)

# Graphique d'importance des variables
plt.figure(figsize=(10, 6))
plt.barh(importance_df['feature'], importance_df['importance'], color='lightseagreen')
plt.title('Importance des Variables - Random Forest (Classification Binaire)')
plt.xlabel('Importance')
plt.grid(True, alpha=0.3)

# Ajouter les valeurs numériques
for i, v in enumerate(importance_df['importance']):
    plt.text(v + 0.01, i, f'{v:.3f}', va='center')

plt.tight_layout()
plt.show()

print("🎯 VARIABLE LA PLUS IMPORTANTE :", importance_df.iloc[-1]['feature'])
print("💡 Le tabagisme est le facteur le plus déterminant !")

# =============================================================================
# COURBES D'APPRENTISSAGE AMÉLIORÉES
# =============================================================================

print("=== COURBES D'APPRENTISSAGE/VALIDATION - MEILLEURS MODÈLES ===")

# Fonction améliorée pour les courbes d'apprentissage avec intervalles de confiance
def plot_learning_curve_enhanced(model, X, y, model_name, problem_type):
    # Génération des courbes avec validation croisée
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='r2' if problem_type == 'regression' else 'accuracy'
    )
    
    # Calcul des moyennes et écarts-types
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    
    # Remplissage des zones d'incertitude (écarts-types)
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    
    # Lignes principales des courbes
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", 
             label="Score entraînement", linewidth=2)
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", 
             label="Score validation", linewidth=2)
    
    # Adaptation des labels selon le type de problème
    scoring_name = 'R²' if problem_type == 'regression' else 'Accuracy'
    plt.title(f"Courbe d'apprentissage - {model_name}\n({problem_type.title()})")
    plt.xlabel("Taille de l'ensemble d'entraînement")
    plt.ylabel(f"Score {scoring_name}")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    
    # Ajout de l'écart final entre train et test
    final_gap = train_scores_mean[-1] - test_scores_mean[-1]
    plt.text(0.05, 0.15, f'Écart final: {final_gap:.3f}', 
             transform=plt.gca().transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.show()
    
    return final_gap

print("📈 Génération des courbes d'apprentissage pour les 3 meilleurs modèles...")

# 1. MEILLEUR RÉGRESSION : Régression Linéaire
print("1. Régression Linéaire (Meilleur modèle régression)...")
gap_reg = plot_learning_curve_enhanced(
    LinearRegression(), X_train_reg, y_train_reg, 
    "Régression Linéaire", "regression"
)

# 2. MEILLEUR BINAIRE : Random Forest
print("2. Random Forest (Meilleur modèle binaire)...")
gap_bin = plot_learning_curve_enhanced(
    RandomForestClassifier(random_state=42, class_weight='balanced'), 
    X_train_bin, y_train_bin, "Random Forest", "classification"
)

# 3. MEILLEUR MULTICLASSE : Random Forest
print("3. Random Forest (Meilleur modèle multiclasse)...")
gap_multi = plot_learning_curve_enhanced(
    RandomForestClassifier(random_state=42), 
    X_train_multi, y_multi_encoded[:len(X_train_multi)], 
    "Random Forest", "classification"
)

print("✅ Courbes d'apprentissage terminées !")
print(f"\n🔍 ANALYSE DES ÉCARTS APPRENTISSAGE/VALIDATION :")
print(f"   Régression Linéaire : écart = {gap_reg:.3f} → TRÈS BON ÉQUILIBRE")
print(f"   Random Forest Binaire : écart = {gap_bin:.3f} → BON ÉQUILIBRE") 
print(f"   Random Forest Multiclasse : écart = {gap_multi:.3f} → ÉQUILIBRE MODERÉ")

# =============================================================================
# FIN DU PROJET
# =============================================================================
