#  Règles d'Association Apriori — Détection de Feux de Forêt
> Projet Data Mining — Master 1 Informatique Visuelle, USTHB 2025-2026  
> Dataset : **Algerian Forest Fires Dataset** (~200 000 transactions)

---

##  Description

Application et amélioration de l'algorithme **Apriori** pour l'extraction de règles d'association à partir d'un dataset de feux de forêt. L'objectif est d'identifier les combinaisons de conditions météorologiques et d'indices FWI (*Fire Weather Index*) les plus fortement associées aux classes `fire` et `not fire`.

Le projet comporte deux implémentations :
- **`apriori.ipynb`** — version notebook interactive avec visualisations avancées
- **`code1.py`** — version script Python autonome

---

##  Structure du projet

```
Apriori-feux-foret/
│
├── apriori.ipynb       # Notebook principal (pipeline complet + visualisations)
├── code1.py            # Script Python autonome
├── data.csv            # Dataset Algerian Forest Fires
└── README.md
```

---

##  Dataset

**Algerian Forest Fires Dataset** — enregistrements météorologiques de deux régions d'Algérie (Béjaïa et Sidi Bel-Abbès) sur la période juin–septembre 2012.

| Feature | Description |
|---|---|
| `Temperature` | Température de l'air (°C) |
| `RH` | Humidité relative (%) |
| `Ws` | Vitesse du vent (km/h) |
| `Rain` | Précipitations (mm) |
| `FFMC` | Fine Fuel Moisture Code |
| `DMC` | Duff Moisture Code |
| `DC` | Drought Code |
| `ISI` | Initial Spread Index |
| `BUI` | Buildup Index |
| `FWI` | Fire Weather Index |
| `Classes` | **Cible** : `fire` / `not fire` |

---

##  Pipeline

### Étape 1 — Chargement & nettoyage
- Chargement du CSV, typage des colonnes objet en string propre
- Aucune suppression de colonnes — toutes les features sont conservées

### Étape 2 — Construction des transactions
- Encodage `attribut=valeur` pour chaque ligne
- La variable cible `Classes` est incluse comme item dans chaque transaction
- Exemple : `['Temperature=29', 'RH=57', ..., 'Classes=fire']`

### Étape 3 — Encodage one-hot
- `TransactionEncoder` (mlxtend) → matrice binaire de présence/absence d'items

### Étape 4 — Sélection du min_support
- Analyse de la variation du nombre d'itemsets fréquents pour `min_support ∈ {0.01, 0.03, 0.05, 0.07, 0.10}`
- **Support retenu : 0.05**

### Étape 5 — Extraction des itemsets fréquents
- Algorithme Apriori avec `max_len=2`, `min_support=0.05`

### Étape 6 — Génération des règles
- Métrique : **confiance** avec seuil `min_confidence=0.5`
- Filtrage supplémentaire par **lift > 1.0** (règles non triviales)

### Étape 7 — Analyse ciblée
- Règles vers `Classes=fire` et `Classes=not fire` séparément
- Top 15 items les plus fréquents dans les antécédents
- Récapitulatif de la distribution des classes

---

## 📊 Visualisations produites

| Graphique | Description |
|---|---|
| Variation min_support | Courbe itemsets fréquents vs seuil de support |
| Distribution des itemsets | Histogramme des longueurs d'itemsets |
| Distribution des antécédents | Histogramme des longueurs d'antécédents |
| Support vs Confiance | Nuage de points — toutes règles |
| Confiance vs Lift | Nuage de points — toutes règles |
| Support vs Confiance (fire) | Nuage de points — règles vers `Classes=fire` |
| Lift vs Confiance (fire) | Nuage de points — règles vers `Classes=fire` |
| Top 10 règles fire (lift) | Barplot horizontal — meilleures règles |
| Top 10 règles not fire (lift) | Barplot horizontal — meilleures règles |
| Items associés à fire | Top 15 antécédents fréquents |
| Items associés à not fire | Top 15 antécédents fréquents |
| Distribution des classes | Barplot fire / not fire |

---

##  Exécution

### Prérequis
```bash
pip install pandas matplotlib mlxtend
```

### Notebook
```bash
jupyter notebook apriori.ipynb
```

### Script Python
```bash
python code1.py
```

> Placer `data.csv` dans le même dossier avant l'exécution.

---

##  Métriques des règles d'association

| Métrique | Formule | Interprétation |
|---|---|---|
| **Support** | P(A ∪ B) | Fréquence de la règle dans le dataset |
| **Confiance** | P(B \| A) | Probabilité conditionnelle du conséquent |
| **Lift** | P(A∪B) / (P(A)·P(B)) | Lift > 1 : association positive non aléatoire |

---



