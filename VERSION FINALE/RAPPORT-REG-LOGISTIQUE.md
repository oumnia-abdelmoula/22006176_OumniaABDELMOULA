# Compte Rendu : Régression Logistique pour la Prédiction de Remboursement de Prêts

**Auteur** : Analyse basée sur le notebook Kaggle "Simple Logistic Regression"  
**Dataset** : Playground Series S5E11  
**Date** : Décembre 2024  
**Score Final** : ROC AUC = 0.9233

---

## 📋 Table des Matières

1. [Introduction et Contexte](#1-introduction-et-contexte)
2. [Fondamentaux de la Régression Logistique](#2-fondamentaux-de-la-régression-logistique)
3. [Exploration des Données](#3-exploration-des-données)
4. [Analyse de Corrélation](#4-analyse-de-corrélation)
5. [Prétraitement des Données](#5-prétraitement-des-données)
6. [Construction du Modèle](#6-construction-du-modèle)
7. [Évaluation et Validation](#7-évaluation-et-validation)
8. [Résultats et Interprétation](#8-résultats-et-interprétation)
9. [Annexes Techniques](#9-annexes-techniques)

---

## 1. Introduction et Contexte

### 1.1 Objectif du Projet

Ce projet vise à construire un modèle de **régression logistique** pour prédire si un emprunteur remboursera son prêt. Il s'agit d'un problème de **classification binaire** fondamental dans le domaine du crédit et de la gestion des risques financiers.

### 1.2 Pourquoi la Régression Logistique ?

La régression logistique est choisie pour plusieurs raisons :

- ✅ **Simplicité** : Modèle linéaire facile à comprendre et à implémenter
- ✅ **Interprétabilité** : Les coefficients ont une signification claire
- ✅ **Efficacité** : Rapide à entraîner, même sur de grands datasets
- ✅ **Probabilités** : Fournit des scores de probabilité, pas seulement des classes
- ✅ **Baseline** : Excellent point de départ avant des modèles plus complexes

### 1.3 Importance Pratique

Dans le secteur financier, prédire le remboursement des prêts permet de :

1. **Réduire les risques** : Identifier les emprunteurs à risque
2. **Optimiser les décisions** : Approuver les bons prêts, refuser les mauvais
3. **Gérer le capital** : Allouer efficacement les ressources
4. **Respecter les régulations** : Justifier les décisions de crédit

---

## 2. Fondamentaux de la Régression Logistique

### 2.1 Principe Général

La régression logistique transforme une combinaison linéaire de features en une **probabilité** entre 0 et 1.

**Processus complet** :

```
Étape 1: Combinaison Linéaire
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b

Étape 2: Fonction Sigmoïde
P(y=1|x) = σ(z) = 1 / (1 + e⁻ᶻ)

Étape 3: Décision
Classe prédite = {1 si P ≥ 0.5
                 {0 si P < 0.5
```

### 2.2 La Fonction Sigmoïde

**Équation** :
```
σ(z) = 1 / (1 + e^(-z))
```

**Propriétés mathématiques** :
- Domaine : z ∈ ℝ (tous les réels)
- Image : σ(z) ∈ [0, 1] (probabilité)
- Point d'inflexion : σ(0) = 0.5
- Limites : lim(z→+∞) σ(z) = 1, lim(z→-∞) σ(z) = 0

**Visualisation de la courbe sigmoïde** :

```
P(y=1)
  1.0 │           ┌────────────
      │         ╱
      │       ╱
  0.5 │     ╱    ← Seuil de décision
      │   ╱
      │ ╱
  0.0 │┘
      └──────────────────────────→ z
       -6  -4  -2   0   2   4   6
```

### 2.3 Fonction de Coût (Log Loss)

Pour entraîner le modèle, on minimise la **log loss** (entropie croisée binaire) :

```
J(w, b) = -1/m Σᵢ₌₁ᵐ [yᵢ·log(ŷᵢ) + (1-yᵢ)·log(1-ŷᵢ)]

où :
  - m = nombre d'exemples
  - yᵢ = vraie classe (0 ou 1)
  - ŷᵢ = probabilité prédite P(y=1|xᵢ)
```

**Intuition** :
- Si y=1 et ŷ→1 : log(ŷ)→0, coût faible ✓
- Si y=1 et ŷ→0 : log(ŷ)→-∞, coût élevé ✗
- Le modèle est pénalisé pour les prédictions confiantes mais fausses

### 2.4 Optimisation : Gradient Descent

**Algorithme** :

```
Initialiser w et b aléatoirement
Pour chaque itération jusqu'à convergence :
    1. Calculer les prédictions : ŷ = σ(Xw + b)
    2. Calculer le gradient : ∂J/∂w = (1/m)Xᵀ(ŷ - y)
    3. Mettre à jour : w := w - α·(∂J/∂w)
                       b := b - α·(∂J/∂b)
où α = learning rate
```

### 2.5 Régularisation

Pour éviter le **sur-apprentissage**, on ajoute un terme de pénalité :

**Régularisation L2 (Ridge)** :
```
J_reg(w) = J(w) + λ·||w||²

où :
  - λ = paramètre de régularisation
  - ||w||² = somme des carrés des coefficients
```

**En scikit-learn** : Le paramètre `C` est l'inverse de λ
```python
C = 1/λ
# C petit → forte régularisation
# C grand → faible régularisation
```

---

## 3. Exploration des Données

### 3.1 Structure du Dataset

**Caractéristiques générales** :

| Dataset | Nombre de lignes | Fichier |
|---------|------------------|---------|
| Entraînement | 10,000 | `train.csv` |
| Test | 5,000 | `test.csv` |
| **Total** | **15,000** | - |

### 3.2 Variables du Dataset

#### Variables Numériques

| Variable | Type | Plage | Description |
|----------|------|-------|-------------|
| `id` | Identifiant | 0 - 9,999 | Identifiant unique |
| `credit_score` | Discret | 300 - 849 | Score de crédit |
| `debt_to_income_ratio` | Continu | 0.10 - 0.60 | Ratio dette/revenu |
| `interest_rate` | Continu | 3% - 15% | Taux d'intérêt du prêt |

#### Variables Catégorielles

| Variable | Modalités | Description |
|----------|-----------|-------------|
| `gender` | Male, Female | Genre de l'emprunteur |
| `marital_status` | Single, Married, Divorced | Statut matrimonial |
| `education_level` | High School, Bachelors, Masters, PhD | Niveau d'éducation |
| `employment_status` | Employed, Unemployed, Self-Employed | Statut d'emploi |
| `loan_purpose` | Home, Car, Education, Other | But du prêt |
| `grade_subgrade` | A1, B2, C3, D4 | Grade du prêt |

#### Variable Cible

| Variable | Type | Valeurs | Description |
|----------|------|---------|-------------|
| **`loan_paid_back`** | **Binaire** | **0, 1** | **0 = Non remboursé, 1 = Remboursé** |

### 3.3 Statistiques Descriptives

#### Variables Numériques

```
credit_score:
  ├─ count: 10,000
  ├─ mean:  574.88
  ├─ std:   159.00
  ├─ min:   300
  ├─ 25%:   436
  ├─ 50%:   573
  ├─ 75%:   714
  └─ max:   849

debt_to_income_ratio:
  ├─ count: 10,000
  ├─ mean:  0.3499
  ├─ std:   0.1436
  ├─ min:   0.1000
  ├─ 25%:   0.2260
  ├─ 50%:   0.3510
  ├─ 75%:   0.4730
  └─ max:   0.5999

interest_rate:
  ├─ count: 10,000
  ├─ mean:  9.01%
  ├─ std:   3.45%
  ├─ min:   3.00%
  ├─ 25%:   6.09%
  ├─ 50%:   9.01%
  ├─ 75%:   12.01%
  └─ max:   15.00%

loan_paid_back (CIBLE):
  ├─ count: 10,000
  ├─ mean:  0.4968
  ├─ std:   0.5000
  ├─ min:   0
  ├─ 25%:   0
  ├─ 50%:   0
  ├─ 75%:   1
  └─ max:   1
```

### 3.4 Distribution de la Variable Cible

**Répartition des classes** :

```
Classe 0 (Non remboursé) : 5,032 exemples (50.32%)
Classe 1 (Remboursé)     : 4,968 exemples (49.68%)
─────────────────────────────────────────────────
Total                    : 10,000

Visualisation:
┌────────────────────────────────────┐
│ Not Paid Back │ Paid Back          │
│     5032      │    4968            │
│ ████████████  │ ████████████       │
│ ████████████  │ ████████████       │
│ ████████████  │ ████████████       │
└────────────────────────────────────┘
      50.32%          49.68%
```

**Observation critique** :
- ✅ **Classes parfaitement équilibrées** (~50/50)
- ✅ Pas besoin de techniques de rééchantillonnage (SMOTE, undersampling)
- ✅ Métrique accuracy pertinente (pas de classe majoritaire)

### 3.5 Cardinalité des Variables

```
Nombre de valeurs uniques par variable:

id                      : 10,000  (tous uniques)
credit_score            : 550     (discret)
gender                  : 2       (binaire)
marital_status          : 3       (catégoriel)
debt_to_income_ratio    : 10,000  (continu)
education_level         : 4       (ordinal)
employment_status       : 3       (catégoriel)
loan_purpose            : 4       (catégoriel)
grade_subgrade          : 4       (ordinal)
interest_rate           : 10,000  (continu)
loan_paid_back          : 2       (binaire - CIBLE)
```

---

## 4. Analyse de Corrélation

### 4.1 Matrice de Corrélation des Variables Numériques

**Variables analysées** :
- `credit_score`
- `debt_to_income_ratio`
- `interest_rate`
- `loan_paid_back` (cible)

**Résultats (valeurs approximatives)** :

```
Corrélation avec loan_paid_back:
├─ credit_score          : +0.05  (corrélation très faible positive)
├─ debt_to_income_ratio  : -0.03  (corrélation très faible négative)
└─ interest_rate         : -0.02  (corrélation très faible négative)

Intercorrélations:
├─ credit_score ↔ interest_rate        : -0.10
├─ credit_score ↔ debt_to_income_ratio : +0.02
└─ interest_rate ↔ debt_to_income_ratio: +0.03
```

### 4.2 Interprétation

**Constatations principales** :

1. **Faibles corrélations linéaires** : Aucune variable numérique n'a une corrélation forte avec la cible
   - Cela suggère que les relations sont **non-linéaires** ou que les **variables catégorielles** sont plus importantes

2. **Absence de multicolinéarité** : Les variables explicatives ne sont pas fortement corrélées entre elles
   - ✅ Bon pour la stabilité du modèle
   - ✅ Pas de redondance d'information

3. **Importance des variables catégorielles** : Les features comme `grade_subgrade`, `employment_status`, etc. pourraient avoir plus de pouvoir prédictif

### 4.3 Visualisation

```
Heatmap de corrélation (représentation textuelle):

                      credit  debt_to  interest  loan_paid
                      score   income   rate      back
credit_score          1.00    0.02    -0.10     0.05
debt_to_income_ratio  0.02    1.00     0.03    -0.03
interest_rate        -0.10    0.03     1.00    -0.02
loan_paid_back        0.05   -0.03    -0.02     1.00

Légende:
  1.00 : Corrélation parfaite
  0.70+: Corrélation forte
  0.30+: Corrélation modérée
  0.10+: Corrélation faible
  0.00 : Aucune corrélation
```

---

## 5. Prétraitement des Données

### 5.1 Vue d'Ensemble du Pipeline

```
┌─────────────────┐
│  Données Brutes │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│ 1. Séparation Cible/ID  │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ 2. Identification Types │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ 3. One-Hot Encoding     │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ 4. Standardisation      │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Données Prêtes pour ML │
└─────────────────────────┘
```

### 5.2 Étape 1 : Séparation Cible et Identifiants

**Code** :
```python
# Extraction de la variable cible
y = X_full.pop('loan_paid_back')

# Conservation des IDs test pour la soumission finale
testID = X_test.pop('id')

# Suppression de l'ID du dataset d'entraînement
X_full.drop('id', axis=1, inplace=True)
```

**Résultat** :
- `y` : vecteur de 10,000 valeurs (0 ou 1)
- `X_full` : DataFrame de 10,000 × 9 features
- `testID` : vecteur de 5,000 IDs pour reconstituer la soumission

### 5.3 Étape 2 : Identification des Types de Variables

**Décision de traitement** :

```python
# Variables à encoder (catégorielles + pseudo-numériques)
cat_cols = [
    'credit_score',           # Discret → Catégoriel
    'gender',                 # Catégoriel
    'marital_status',         # Catégoriel
    'debt_to_income_ratio',   # Continu → Catégoriel (choix de l'auteur)
    'education_level',        # Catégoriel
    'employment_status',      # Catégoriel
    'loan_purpose',           # Catégoriel
    'grade_subgrade'          # Catégoriel
]

# Variables numériques continues
num_cols = ['interest_rate']  # Seule variable vraiment continue
```

**Justification** :
- `credit_score` : Bien que numérique, a seulement 550 valeurs distinctes → traité comme catégoriel
- `debt_to_income_ratio` : Choix de l'auteur de le traiter comme catégoriel
- Cette approche peut capturer des **relations non-linéaires** plus complexes

### 5.4 Étape 3 : One-Hot Encoding

#### Principe

Le **One-Hot Encoding** transforme chaque variable catégorielle en plusieurs colonnes binaires (0/1).

**Exemple** :

```
Avant encoding:
┌────────┐
│ gender │
├────────┤
│ Male   │
│ Female │
│ Male   │
└────────┘

Après encoding:
┌────────────┬──────────────┐
│ gender_Male│ gender_Female│
├────────────┼──────────────┤
│     1      │      0       │
│     0      │      1       │
│     1      │      0       │
└────────────┴──────────────┘
```

#### Implémentation

```python
from sklearn.preprocessing import OneHotEncoder

# Initialisation
Oh = OneHotEncoder(
    handle_unknown='ignore',  # Ignore les catégories inconnues dans le test
    sparse_output=False       # Retourne un array dense (pas sparse)
)

# Encoding des données d'entraînement
X_encoded = pd.DataFrame(Oh.fit_transform(X_full[cat_cols]))

# Encoding des données de test (avec les mêmes catégories)
test_encoded = pd.DataFrame(Oh.transform(X_test[cat_cols]))
```

#### Résultat

**Dimensions** :

```
Avant encoding:
  X_full: 10,000 × 8 colonnes (cat_cols)

Après encoding:
  X_encoded: 10,000 × N colonnes
  où N = nombre total de modalités de toutes les variables catégorielles

Exemple de calcul:
  gender (2) + marital_status (3) + education_level (4) + 
  employment_status (3) + loan_purpose (4) + grade_subgrade (4) +
  credit_score (550) + debt_to_income_ratio (10,000)
  = Beaucoup de colonnes !
```

**Note** : Le nombre exact de colonnes dépend des modalités présentes dans le dataset.

### 5.5 Étape 4 : Jonction avec Variables Numériques

```python
# Ajout de la variable numérique continue
X = X_encoded.join(X_full[num_cols])
test = test_encoded.join(X_test[num_cols])

# Conversion des noms de colonnes en string (pour éviter les erreurs)
X.columns = X.columns.astype(str)
test.columns = test.columns.astype(str)
```

### 5.6 Étape 5 : Standardisation (Scaling)

#### Importance Critique

**Pourquoi standardiser ?**

1. **Échelles différentes** : Les variables ont des plages très différentes
   - `credit_score` : [300, 849]
   - `interest_rate` : [3, 15]
   - Colonnes one-hot : [0, 1]

2. **Gradient descent** : La convergence est plus rapide avec des features standardisées

3. **Régularisation** : La pénalité L2 doit traiter toutes les features équitablement

#### Formule de Standardisation (Z-score)

```
Pour chaque feature j:

x'_ij = (x_ij - μ_j) / σ_j

où :
  x_ij  : valeur originale (exemple i, feature j)
  μ_j   : moyenne de la feature j
  σ_j   : écart-type de la feature j
  x'_ij : valeur standardisée

Résultat: x' ~ N(0, 1) (distribution approximativement normale)
```

#### Implémentation

```python
from sklearn.preprocessing import StandardScaler

# Initialisation du scaler
scaler = StandardScaler()

# Fit sur train, transform train et test
X_scaled = pd.DataFrame(scaler.fit_transform(X))
test_scaled = pd.DataFrame(scaler.transform(test))
```

**⚠️ Important** :
- Le scaler est **fit sur train uniquement** (pas sur test)
- Puis appliqué à train ET test avec les mêmes paramètres (μ et σ du train)
- Cela évite le **data leakage**

#### Exemple de Transformation

```
Variable: interest_rate

Avant standardisation:
  min = 3.00, max = 15.00, mean = 9.01, std = 3.45

Après standardisation:
  Pour x = 3.00  : z = (3.00 - 9.01) / 3.45 = -1.74
  Pour x = 9.01  : z = (9.01 - 9.01) / 3.45 =  0.00
  Pour x = 15.00 : z = (15.00 - 9.01) / 3.45 = +1.74

Nouvelle distribution:
  min ≈ -1.74, max ≈ +1.74, mean = 0, std = 1
```

---

## 6. Construction du Modèle

### 6.1 Split Train/Validation

**Objectif** : Séparer les données pour évaluer le modèle sur un ensemble qu'il n'a jamais vu.

```python
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,      # 20% pour validation
    random_state=42     # Pour reproductibilité
)
```

**Répartition** :

```
Dataset Total: 10,000 exemples
├─ Train      : 8,000 exemples (80%)
│              └─ Utilisés pour apprendre les coefficients w
│
└─ Validation : 2,000 exemples (20%)
               └─ Utilisés pour évaluer la performance
```

### 6.2 Configuration du Modèle

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    C=1e-3,         # Paramètre de régularisation (λ = 1/C = 1000)
    max_iter=1000,  # Nombre max d'itérations du gradient descent
    solver='lbfgs', # Algorithme d'optimisation (par défaut)
    random_state=42 # Pour reproductibilité
)
```

#### Paramètre C (Régularisation)

**Relation** : C = 1/λ

```
C = 1e-3 = 0.001
  ↓
λ = 1/C = 1000

Effet:
├─ λ élevé (C faible) → FORTE régularisation
│                      → Coefficients w proches de 0
│                      → Modèle simple, moins de sur-apprentissage
│
└─ λ faible (C grand) → FAIBLE régularisation
                       → Coefficients w peuvent être grands
                       → Modèle complexe, risque de sur-apprentissage
```

**Choix C=1e-3** :
- ✅ Régularisation très forte
- ✅ Adapté pour éviter le sur-apprentissage avec beaucoup de features (one-hot encoding)
- ✅ Privilégie la généralisation sur l'ajustement parfait aux données d'entraînement

### 6.3 Entraînement du Modèle

```python
# Entraînement
model.fit(X_train, y_train)

# Le modèle apprend les coefficients w et le biais b
# en minimisant la log loss régularisée :
# J(w) = -1/m Σ[y·log(ŷ) + (1-y)·log(1-ŷ)] + λ·||w||²
```

**Processus interne** :

```
Itération 1:
  1. Initialiser w et b aléatoirement
  2. Calculer prédictions: ŷ = σ(Xw + b)
  3. Calculer coût: J(w)
  4. Calculer gradients: ∂J/∂w
  5. Mettre à jour: w := w - α·(∂J/∂w)

Itération 2:
  [répéter étapes 2-5]
  ...

Itération N:
  [convergence atteinte ou max_iter=1000]
```

---

## 7. Évaluation et Validation

### 7.1 Métriques pour la Classification Binaire

#### ROC AUC (Métrique Principale)

**Définition** : Area Under the Receiver Operating Characteristic Curve

**Interprétation** :

```
AUC = 1.0   → Modèle parfait (sépare parfaitement les classes)
AUC = 0.9+  → Excellent modèle
AUC = 0.8+  → Très bon modèle
AUC = 0.7+  → Bon modèle
AUC = 0.6+  → Modèle moyen
AUC = 0.5   → Modèle aléatoire (inutile)
AUC < 0.5   → Modèle pire que aléatoire
```

**Avantages** :
- ✅ Insensible au déséquilibre des classes (mais ici équilibrées)
- ✅ Mesure la capacité de discrimination globale
- ✅ Indépendant du seuil de décision choisi

#### Courbe ROC

**Construction** :

```
Pour chaque seuil t de 0 à 1:
  1. Classer y_pred ≥ t comme classe 1
  2. Calculer TPR (True Positive Rate) = TP / (TP + FN)
  3. Calculer FPR (False Positive Rate) = FP / (FP + TN)
  4. Tracer le point (FPR, TPR)

AUC = aire sous la courbe tracée
```

**Visualisation** :

```
TPR │
1.0 │ ┌─────────┐  AUC = 0.92
    │ │         │
0.8 │ │         │
    │ │         │
0.6 │ │         │
    │╱          │
0.4 │           │
    │           │
0.2 │           │
    │           │
0.0 │───────────┘
    └─────────────────────
    0   0.2  0.4  0.6  0.8  1.0
                FPR

Légende:
  Ligne bleue: Notre modèle (AUC=0.92)
  Diagonale: Modèle aléatoire (AUC=0.5)
  Coin supérieur gauche: Modèle parfait (AUC=1.0)
```

### 7.2 Validation Hold-Out

**Méthode** : Évaluer sur l'ensemble de validation séparé

```python
# Prédictions sur validation
y_pred_proba = model.predict_proba(X_val)[:, 1]  # Probabilités classe 1

# Calcul du score
from sklearn.metrics import roc_auc_score
score_val = roc_auc_score(y_val, y_pred_proba)

print(f"ROC AUC (Validation): {score_val:.4f}")
# Résultat attendu: ~0.92
```

**Limites** :
- ❌ Un seul split : le score peut varier selon le split
- ❌ Perte de données (20% non utilisés pour l'entraînement)

### 7.3 Validation Croisée (Cross-Validation)

**Méthode** : K-Fold Cross-Validation (K=5)

```
Principe:
┌──────────────────────────────────────┐
│         Dataset complet (10,000)     │
└──────────────────────────────────────┘
         ↓ Division en 5 folds
┌────────┬────────┬────────┬────────┬────────┐
│ Fold 1 │ Fold 2 │ Fold 3 │ Fold 4 │ Fold 5 │
└────────┴────────┴────────┴────────┴────────┘
  2,000    2,000    2,000    2,000    2,000

Itération 1: Train [2,3,4,5], Test [1]
Itération 2: Train [1,3,4,5], Test [2]
Itération 3: Train [1,2,4,5], Test [