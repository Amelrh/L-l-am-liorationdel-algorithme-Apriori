# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from collections import Counter

plt.rcParams['figure.dpi'] = 120

# ----------------------------------------------------
# 1. Charger le dataset complet
# ----------------------------------------------------
df = pd.read_csv("data.csv")

print("Shape du dataset:", df.shape)
print("Colonnes:", df.columns.tolist())

# ----------------------------------------------------
# 2. Nettoyage / typage simple
# ----------------------------------------------------
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].astype(str).str.strip()

# ----------------------------------------------------
# 3. Construire les transactions (attribut=valeur)
# ----------------------------------------------------
feature_cols = [c for c in df.columns if c != "Classes"]
target_col = "Classes"

transactions = []
for _, row in df.iterrows():
    items = []
    for col in feature_cols:
        items.append(f"{col}={row[col]}")
    items.append(f"{target_col}={row[target_col]}")
    transactions.append(items)

print("Exemple de transaction:")
print(transactions[0])

# ----------------------------------------------------
# 4. Encodage one-hot
# ----------------------------------------------------
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_onehot = pd.DataFrame(te_ary, columns=te.columns_)

print("Shape one-hot:", df_onehot.shape)

# ----------------------------------------------------
# 5. Effet de min_support sur le nombre d'itemsets
# ----------------------------------------------------
minsups = [0.01, 0.03, 0.05, 0.07, 0.10]
counts = []

for ms in minsups:
    fi = apriori(df_onehot, min_support=ms, use_colnames=True, max_len=2)
    counts.append(len(fi))

plt.figure(figsize=(6,4))
plt.plot(minsups, counts, marker='o')
plt.xlabel('min_support')
plt.ylabel("Nombre d'itemsets fréquents (max_len=2)")
plt.title("Variation du nombre d'itemsets fréquents selon min_support")
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------------------------------
# 6. Itemsets fréquents pour min_support choisi
# ----------------------------------------------------
min_support_base = 0.05  # support choisi après analyse
frequent_itemsets = apriori(
    df_onehot,
    min_support=min_support_base,
    use_colnames=True,
    max_len=2
)
frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(len)

print("Nombre d'itemsets fréquents (min_support=0.05):",
      len(frequent_itemsets))
print(frequent_itemsets.sort_values("support",
                                    ascending=False).head(10))

# Histogramme de la longueur des itemsets
plt.figure(figsize=(6,4))
frequent_itemsets["length"].hist(bins=[1,2,3])
plt.xlabel("Longueur des itemsets")
plt.ylabel("Nombre d'itemsets")
plt.title("Distribution de la longueur des itemsets fréquents")
plt.tight_layout()
plt.show()

# ----------------------------------------------------
# 7. Règles d'association (toutes classes confondues)
# ----------------------------------------------------
rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.5
)

print("Nombre total de règles:", len(rules))
print(rules[['antecedents', 'consequents',
             'support', 'confidence', 'lift']].head(10))

# Longueur des règles
rules['antecedent_len'] = rules['antecedents'].apply(len)
rules['rule_len'] = rules['antecedents'].apply(len) \
                     + rules['consequents'].apply(len)

print("\nStatistiques longueur des antécédents:")
print(rules['antecedent_len'].describe())
print("\nStatistiques longueur des règles complètes:")
print(rules['rule_len'].describe())

plt.figure(figsize=(6,4))
rules['antecedent_len'].hist(bins=[1,2,3,4,5])
plt.xlabel("Longueur des antécédents")
plt.ylabel("Nombre de règles")
plt.title("Distribution de la longueur des antécédents")
plt.tight_layout()
plt.show()

# Nuage support vs confiance (toutes règles)
plt.figure(figsize=(7,5))
plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('Support')
plt.ylabel('Confiance')
plt.title("Toutes règles : support vs confiance")
plt.grid(True)
plt.tight_layout()
plt.show()

# Nuage confiance vs lift (toutes règles)
plt.figure(figsize=(7,5))
plt.scatter(rules['confidence'], rules['lift'], alpha=0.5)
plt.xlabel('Confiance')
plt.ylabel('Lift')
plt.title("Toutes règles : confiance vs lift")
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------------------------------
# 8. Filtrage par lift (règles intéressantes)
# ----------------------------------------------------
rules_filtered = rules[rules['lift'] > 1.0]
print("Nombre de règles avec lift > 1.0 :", len(rules_filtered))

# ----------------------------------------------------
# 9. Focus sur Classes=fire et Classes=not fire
# ----------------------------------------------------
rules_fire = rules_filtered[
    rules_filtered['consequents'].astype(str).str.contains('Classes=fire')
]
rules_not_fire = rules_filtered[
    rules_filtered['consequents'].astype(str).str.contains('Classes=not fire')
]

print("Nombre de règles vers 'Classes=fire':", len(rules_fire))
print("Nombre de règles vers 'Classes=not fire':", len(rules_not_fire))

# --- FIRE : graphs dédiés
if len(rules_fire) > 0:
    # support vs confiance
    plt.figure(figsize=(7,5))
    plt.scatter(rules_fire['support'], rules_fire['confidence'], alpha=0.7)
    plt.xlabel('Support')
    plt.ylabel('Confiance')
    plt.title("Règles vers 'Classes=fire' : support vs confiance")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # confiance vs lift
    plt.figure(figsize=(7,5))
    plt.scatter(rules_fire['confidence'], rules_fire['lift'], alpha=0.7)
    plt.xlabel('Confiance')
    plt.ylabel('Lift')
    plt.title("Règles vers 'Classes=fire' : lift vs confiance")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # top 10 par lift
    top_fire = rules_fire.sort_values('lift', ascending=False).head(10)
    plt.figure(figsize=(9,5))
    plt.barh(range(len(top_fire)),
             top_fire['lift'],
             tick_label=top_fire['antecedents'].astype(str))
    plt.xlabel('Lift')
    plt.title("Top 10 règles vers 'Classes=fire' (par lift)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

# --- NOT FIRE : graphs dédiés
if len(rules_not_fire) > 0:
    # support vs confiance
    plt.figure(figsize=(7,5))
    plt.scatter(rules_not_fire['support'], rules_not_fire['confidence'], alpha=0.7)
    plt.xlabel('Support')
    plt.ylabel('Confiance')
    plt.title("Règles vers 'Classes=not fire' : support vs confiance")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # confiance vs lift
    plt.figure(figsize=(7,5))
    plt.scatter(rules_not_fire['confidence'], rules_not_fire['lift'], alpha=0.7)
    plt.xlabel('Confiance')
    plt.ylabel('Lift')
    plt.title("Règles vers 'Classes=not fire' : lift vs confiance")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # top 10 par lift
    top_not_fire = rules_not_fire.sort_values('lift', ascending=False).head(10)
    plt.figure(figsize=(9,5))
    plt.barh(range(len(top_not_fire)),
             top_not_fire['lift'],
             tick_label=top_not_fire['antecedents'].astype(str))
    plt.xlabel('Lift')
    plt.title("Top 10 règles vers 'Classes=not fire' (par lift)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

# ----------------------------------------------------
# 10. Items les plus fréquents dans les antécédents
# ----------------------------------------------------
def plot_top_antecedent_items(rules_subset, title):
    antecedent_items = []
    for ants in rules_subset['antecedents']:
        for a in ants:
            antecedent_items.append(a)
    counter_items = Counter(antecedent_items)
    top_items = counter_items.most_common(15)

    print(title)
    for item, cnt in top_items:
        print(item, ":", cnt)

    if len(top_items) > 0:
        labels = [x[0] for x in top_items]
        values = [x[1] for x in top_items]

        plt.figure(figsize=(9,5))
        plt.barh(range(len(labels)), values, tick_label=labels)
        plt.xlabel("Fréquence dans les antécédents")
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

if len(rules_fire) > 0:
    plot_top_antecedent_items(
        rules_fire,
        "Items les plus associés à 'Classes=fire'"
    )

if len(rules_not_fire) > 0:
    plot_top_antecedent_items(
        rules_not_fire,
        "Items les plus associés à 'Classes=not fire'"
    )

# ----------------------------------------------------
# 11. Récapitulatif sur la variable cible
# ----------------------------------------------------
n_total = len(df)
class_counts = df['Classes'].value_counts()

print("Nombre total d'observations:", n_total)
print("Distribution de la classe:")
print(class_counts)

plt.figure(figsize=(4,4))
class_counts.plot(kind='bar')
plt.title("Distribution de la classe (fire / not fire)")
plt.ylabel("Nombre d'observations")
plt.tight_layout()
plt.show()
