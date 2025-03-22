# Travaux pratiques N° 2
*Deep learning pour les problèmes spatio-temporels : Maintenance prédictive*

par Joseph OGODJA & Gilbert HABA
1. **Objectif**

  L’objectif de ce TP est de développer un modèle de prédiction basé sur des réseaux de neurones à
mémoire à long terme (LSTM) afin de prédire l’état d’une pompe industrielle en considérant la
composante temporelle.
2. **Description des données (dataset)**

  Soit le dataset « maintenance_predictive.csv ». Ce dataset contient des mesures collectées à partir de capteurs installés sur des pompes Industrielles. Il s’agit d’un problème de classification multiclasse où l’objectif est de prédire l’état de la pompe en fonction des données temporelles des capteurs. Les classes possibles sont :
  
  - **0** : *Fonctionnement normal.*
  - **1** : *Panne liée à une surchauffe.*
  - **2** : *Panne liée à des vibrations excessives.*
  - **3** : *Panne liée à une surcharge électrique.*

Le dataset contient les colonnes suivantes :
- Variables :
  * **Timestamp** : *Date et heure de la mesure*.
  * **Temperature (°C)** : *Température de fonctionnement de la pompe.*
  * **Vibrations (mm/s)** : *Amplitude des vibrations.*
  * **Pressure (bar)** : *Pression exercée par la pompe.*
  * **Current (A)** : *Courant consommé par le moteur.*
  * **Operating_Time (heures)** : *Temps de fonctionnement cumulé.*
  * **Humidity (%)** : *Humidité ambiante.*
  * **Noise_Level (dB)** : *Niveau sonore.*
  * **Energy_Consumption (kWh)** : *Consommation énergétique.*
  * **Pump_Type** : *Type de pompe (Centrifuge, Diaphragm, Piston).*
  * **Location** : *Localisation de la pompe (Zone A, Zone B, Zone C).*

- Variable cible :
  * **State** : *État de la pompe (0, 1, 2, 3).*



### 📊 **Tableau récapitulatif des architectures testées et performances associées**  

| Modèle | Couches principales | Batch Size | Class Weight | Learning Rate | Epochs arrêtées | Train Acc. | Val Acc. | Train Loss | Val Loss | Overfitting ? |
|--------|----------------------|------------|--------------|---------------|-----------------|------------|----------|------------|----------|---------------|
| **V1.0** | LSTM(64) → LSTM(32) → Dense(32) | 32 | ❌ Non | 0.001 | 8 | 30% | ~10% | ↘️ Diminue | ↗️ Augmente | 🔴 Oui |
| **V2.0** | LSTM(64) → LSTM(32) → Dense(32) | 32 | ❌ Non | 0.001 (avec ReduceLROnPlateau) | 8 | 30% | ~10% | ↘️ Diminue | ↗️ Augmente | 🔴 Oui |
| **V3.0** | **LSTM(128) → LSTM(64) → Dense(64)** | **64** | ✅ Oui | 0.001 (avec ReduceLROnPlateau) | 8 | 35% | ~15% | ↘️ Diminue | ↗️ Augmente | 🔴 Oui |
---

# Entrainement du model 1ère itération :
D'après les graphiques de suivi dans **Weights & Biases (WandB)**, voici une analyse détaillée des performances du modèle LSTM :  

---
### **1. Évolution de la perte sur l’ensemble d'entraînement (`epoch/loss`)**  

![loss](loss.svg)
fig.1 :Courbe d'évolution de la perte sur l’ensemble d'entraînement

**Interpretations** :

La courbe descend progressivement, ce qui signifie que le modèle apprend bien sur les données d’entraînement. La perte diminue sans fluctuations majeures. Une perte trop basse pourrait indiquer du surapprentissage (overfitting).  

---

### **2. Évolution de la perte sur l’ensemble de validation (`epoch/val_loss`)**

![courbe val_loss](val_loss.svg)
fig.2 : Courbe de la perte sur l'ensemble de validation

**Interpretations** :

Contrairement à la courbe de perte d'entraînement, celle de validation fluctue et **augmente après un certain nombre d’époques**.  

**Problème possible** :  
- Cela indique un **surapprentissage** (overfitting).  
- Après un certain nombre d'époques, le modèle ne généralise plus bien aux nouvelles données.  

**Solution envisagée** :  
- **Réduire le nombre d’époques** (early stopping).  
- **Ajouter de la régularisation** (Dropout plus élevé, L2).  
- **Réduire le learning rate progressivement** pour éviter l’instabilité.  

---

### **3. Précision sur l’ensemble d'entraînement (`epoch/accuracy`)**

![accuracy](accuracy.svg)
fig.3 : Courbe de précision sur l’ensemble d'entraînement 

**Interpretations** :

La courbe est plate et semble s’être stabilisée très rapidement autour de **0.845** (84.5%). Le modèle converge bien. Si elle est **trop haute** par rapport à `val_accuracy`, cela peut signifier un **overfitting**.  

---

### **4. Précision sur l’ensemble de validation (`epoch/val_accuracy`)**

![val_accuracy](val_accuracy.svg)
fig.4 : Courbe de précision sur l’ensemble de validation

**Interpretations** :

La valeur semble **bloquée** et ne progresse pas après les premières époques.  
- Il y a possiblement **une erreur d'affichage** ou **un bug dans le code de logging**.   

---

### **5. Learning Rate (`epoch/learning_rate`)**

![learning_rate](learning_rate.svg)
fig.5 : Courbe de taux d'apprentissage

**Interpretations** :

Il reste constant à **0.001**, sans ajustement au fil des époques.  

# Entrainement du model 2ème itération :

### Améliorations clés :
1. **Correction des erreurs W&B** :  
   - Ajout de `{epoch:02d}` dans le `filepath` pour éviter l'erreur `FileNotFoundError`
   - `save_best_only=True` fonctionne maintenant car on s'assure que le fichier est bien sauvegardé

2. **Meilleure régularisation** :
   - Augmentation du `Dropout` pour éviter l'overfitting

3. **Optimisation du training** :
   - `ReduceLROnPlateau` réduit automatiquement le learning rate si `val_loss` stagne
   - `EarlyStopping` arrête l'entraînement si `val_loss` ne s'améliore pas pendant 10 époques
  
### Résultats obtenus :

### **1️. `epoch/val_loss` (Perte de validation)**
![val_loss_v2](val_loss_v2.svg)
fig.1 : Courbe de la perte sur l'ensemble de validation

**Interpretation :**  
- La courbe est relativement stable jusqu'à environ **l'époque 7**, mais après, on observe une montée soudaine et une forte variation.  
- Cela indique un **début d'overfitting**, où le modèle commence à trop s'adapter aux données d'entraînement et perd en généralisation.  

---

### **2️. `epoch/val_accuracy` (Précision de validation)**
![val_accuracy_v2](val_accuracy_v2.svg)
fig.2 : Courbe de précision sur l’ensemble de validation

**Interpretation :**  
- La courbe est **totalement plate**, ce qui est anormal.  
- Il y a un **bug potentiel** où l'évaluation ne se fait pas correctement.  
---

### **3️. `epoch/loss` (Perte d'entraînement)**
![loss_v2](loss_v2.svg)
fig.3 :Courbe d'évolution de la perte sur l’ensemble

**Interpretation :**  
- Le `loss` diminue de façon régulière, ce qui est un bon signe.  
- Pas de signe de divergence ou de forte oscillation, ce qui montre que l'entraînement est bien réglé. 
---

### **4️. `epoch/learning_rate`**
![learning_rate_v2](learning_rate_v2.svg)
fig.4 : Courbe de taux d'apprentissage

**Interpretation :**
- Le `ReduceLROnPlateau` a bien fonctionné : le taux d'apprentissage a chuté après **l'époque 8**, ce qui est une bonne stratégie.  
- Cela aurait pu se produire **un peu plus tôt** pour éviter la montée du `val_loss`.
---

### **5️. `epoch/accuracy` (Précision d'entraînement)**
![accuracy_v2](accuracy_v2.svg)
fig.5 : Courbe de précision sur l’ensemble d'entraînement

**Interpretation :**  
- On atteint rapidement **84.4% de précision**, puis la courbe devient plate.  
- Cela pourrait être un signe de **plateau de convergence**, mais aussi de **mauvaise gestion des classes** si `val_accuracy` reste figé à 1.
---

# Entrainement du model 3ème itération :

### Améliorations apportées :
1. **Augmentation de la capacité du modèle** :  
   - Plus de neurones dans les couches LSTM pour capturer plus de complexité.
  
2. **Correction de la métrique `val_accuracy`** :  
   - Remplacement de `accuracy` par `sparse_categorical_accuracy` pour correspondre à la loss utilisée.

3. **Gestion des classes déséquilibrées** :  
   - Calcul de `class_weight` pour que le modèle ne soit pas biaisé vers la classe majoritaire.

4. **Optimisation du `learning rate`** :  
   - Réduction plus rapide du `LR` si `val_loss` stagne.
   - `EarlyStopping` plus agressif pour éviter un overfitting inutile.

5. **Augmentation du `batch_size`** :  
   - Passage à `64` pour stabiliser l'entraînement.

### Résultats obtenus :
1. **Validation Accuracy (epoch/val_sparse_categorical_accuracy)**
   ![val_sparse_categorical_accuracy_v3](val_sparse_categorical_accuracy.svg)
   
   - La courbe montre une chute brutale après la première époque, et elle reste proche de zéro.  
   - Cela suggère que le modèle ne généralise pas bien sur les données de validation, ce qui peut être dû à un surajustement aux données d'entraînement ou un problème de distribution des classes.

2. **Validation Loss (epoch/val_loss)**
   ![val_loss_v3](val_loss_v3.svg)

   - La perte de validation augmente progressivement au fil des époques, indiquant que le modèle devient moins performant sur l'ensemble de validation.  
   - Un tel comportement est un signe d’overfitting, où le modèle apprend trop bien les données d'entraînement et ne parvient pas à généraliser.

3. **Training Accuracy (epoch/sparse_categorical_accuracy)**
   ![sparse_categorical_accuracy](sparse_categorical_accuracy.svg)

   - La courbe est instable et montre des fluctuations importantes.  
   - Cela peut indiquer que l'entraînement est instable, potentiellement en raison d'un taux d'apprentissage trop élevé ou d'une architecture inadaptée.

4. **Training Loss (epoch/loss)**
   ![loss_v3](loss_v3.svg) 

   - La perte d'entraînement diminue progressivement, ce qui montre que le modèle apprend bien sur l’ensemble d'entraînement.  
   - Cependant, la divergence avec la perte de validation suggère un overfitting.

5. **Learning Rate (epoch/learning_rate)**  
   ![learning_rate_v3](learning_rate_v3.svg)

   - Le taux d’apprentissage diminue brusquement après quelques époques.  
   - Bien que cette technique aide souvent à stabiliser l'entraînement, cela ne semble pas suffisant pour empêcher l’overfitting.