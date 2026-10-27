# Level 2 — Explication complète de mon code

Ce document complète le [README.md](README.md). Il explique **chaque mot-clé**, **chaque bloc de code** et **chaque modèle** du notebook [level2.ipynb](level2.ipynb).

Le Level 2 reprend la recette du Level 1. Les notions de base (CNN, CLIP, NPR, epoch, loss, AdamW…) sont expliquées en détail dans [../level1/EXPLICATION_CODE.md](../level1/EXPLICATION_CODE.md). Ici, j'insiste sur **ce qui change** pour passer de 2 à 3 classes.

---

## Partie A — Où et comment j'ai lancé le notebook

- **Plateforme** : Kaggle Notebook.
- **Matériel** : 2 GPU **Tesla T4**, mais **un seul utilisé** (`USE_TWO_GPUS = False`).
- **Données** : `/kaggle/input/competitions/gensivana-real-or-fake-level-2-classification`.
- **Sorties** : `/kaggle/working` (submission, checkpoints, caches, audit).
- **Durée visée** : moins de 7 heures (limite Kaggle).
- **Reprise** : si le run est coupé, les caches `.npz` (prédictions) et `.pt` (poids) sont rechargés automatiquement (`RESUME`).

---

## Partie B — Les modèles et pourquoi

Les mêmes 3 experts qu'au Level 1, adaptés à **3 sorties** :

| Expert | Modèle | Entrée | Pourquoi |
|---|---|---|---|
| `deep_rgb` | DeepGuard `king1oo1/deepfake-model` (SigLIP2, Transformer) | 224 px | Déjà spécialisé real/fake, vision **globale** |
| `cnn_rgb` | ConvNeXt-V2 Tiny `convnextv2_tiny.fcmae_ft_in22k_in1k_384` (CNN) | 384 px | Textures et contenu |
| `cnn_npr` | Même ConvNeXt-V2, entrée **NPR**, 5 vues | 384 px | Traces de pixels laissées par les générateurs |

Plus un **stacker** : une **régression logistique multinomiale** (multi-classe) qui combine les 9 probabilités des experts.

**Ce qui change pour DeepGuard** : il a été créé pour 2 classes (real / fake). Je remplace sa tête par une tête à 3 sorties, en **réutilisant ses anciens poids** (voir cellule 8).

**Résultats OOF (Macro F1)** : deep_rgb 0.8934, cnn_rgb 0.9930, cnn_npr 0.9950, mean 0.9969, **stack 0.99707 → retenu**.

Contrairement au Level 1, ici **la combinaison bat chaque expert seul**. Distinguer « généré » et « retouché » demande à la fois la vue globale et les détails.

---

## Partie C — Le code, cellule par cellule

### Cellule 1 — Installation

```python
%pip install -q imagehash timm transformers
```

`%pip` installe les librairies dans l'environnement du notebook. Ici, les versions ne sont **pas bloquées** (contrairement au Level 1). C'est un petit risque pour la reproductibilité. Les versions réellement utilisées sont quand même notées dans l'audit.

### Cellule 2 — Imports et configuration

Mêmes librairies qu'au Level 1, plus :

- `time` : mesurer la durée du run ;
- `classification_report` (sklearn) : affiche précision, rappel et F1 **par classe**.

**Nouvelles constantes :**

| Constante | Valeur | Signification |
|---|---|---|
| `N_CLASSES` | 3 | Nombre de classes |
| `CLASS_NAMES` | `["real", "generated", "local_edit"]` | Noms des classes 0, 1, 2 |
| `META_FOLDS` | 5 | Folds pour évaluer le stacker et les biais |
| `USE_TWO_GPUS` | False | Multi-GPU désactivé (voir encadré) |
| `NUM_WORKERS` | 2 | 2 processus chargent les images en parallèle (plus rapide que 0 au Level 1) |
| `EPOCHS_DEEP`, `EPOCHS_CNN` | 4, 4 | Epochs maximum |
| `LABEL_SMOOTHING` | 0.02 | Lissage des labels (voir cellule 8) |
| `RUN_TAG` | `level2_v2c_s20260726_f3` | Nom du run pour les fichiers de cache |
| `START_TIME` | `time.monotonic()` | Heure de départ, pour calculer la durée |

> **Le bug multi-GPU**
> Le kernel mourait (`c10::AcceleratorError`) dès qu'un 2ᵉ modèle utilisait le 2ᵉ GPU. Cause : le lien direct entre les 2 GPU T4 (**P2P**, *peer-to-peer*) est instable avec `nn.DataParallel` sur Kaggle.
> Solution : un seul GPU. Les batchs passent de 24 → 12 et de 8 → 4, mais l'accumulation passe de 2 → 4 et de 3 → 6. Le **batch effectif reste identique** (48 et 24). La recette d'entraînement ne change pas.

**`seed_worker(worker_id)`** : nouveau. Avec `NUM_WORKERS = 2`, chaque processus de chargement a son propre générateur aléatoire. Cette fonction lui donne une graine fixe (`SEED + worker_id`) pour que les augmentations soient **reproductibles**.

### Cellule 4 — Données et groupes anti-doublons

**`find_root()`** : cherche un dossier qui contient **les 4 fichiers obligatoires** (`train/ground_truth.csv`, `val/ground_truth.csv`, `test/images`, `sample_submission.csv`).

**`read_split(root, name, labelled)`** :
- renomme la colonne `classe` → `class` → `target` ;
- vérifie que toutes les images existent (sinon `FileNotFoundError`) ;
- vérifie que les labels sont **exactement** {0, 1, 2} (sinon `ValueError`).

Sortie : `{0: 2619, 1: 2619, 2: 2619}` → classes parfaitement équilibrées.

**pHash et groupes — ce qui change par rapport au Level 1 :**

```python
keys = [
    (integer >> 51) & 8191,   # bits 51 à 63  (13 bits)
    (integer >> 38) & 8191,   # bits 38 à 50  (13 bits)
    (integer >> 25) & 8191,   # bits 25 à 37  (13 bits)
    (integer >> 12) & 8191,   # bits 12 à 24  (13 bits)
    integer & 4095,           # bits 0 à 11   (12 bits)
]
```

| Mot-clé | Signification |
|---|---|
| `>>` (décalage à droite) | Pousse les bits vers la droite, pour amener le morceau voulu au début |
| `& 8191` | Garde seulement les 13 derniers bits (8191 = 2¹³ − 1 = treize « 1 » en binaire) |
| **5 morceaux** | Si 2 empreintes ont ≤ 4 bits différents, au moins **1 morceau sur 5 est identique** (principe des tiroirs : 4 différences ne peuvent pas toucher 5 morceaux). Aucune paire n'est ratée. |
| Insertion progressive | Chaque image est comparée aux images **déjà rangées**, puis rangée à son tour. Chaque paire n'est testée qu'une fois. |

Puis Union-Find (comme au Level 1) → `groups`.

**Contrôle anti-fuite :**

```python
overlap = set(groups[train_index]) & set(groups[valid_index])
if overlap:
    raise RuntimeError(f"Fuite de groupes dans le fold {fold}")
```

`&` entre deux ensembles = **intersection**. Si un groupe est des deux côtés, le notebook s'arrête.

### Cellule 6 — Transformations et Dataset

- Transformations : **identiques au Level 1** (RandomResizedCrop, flip, ColorJitter, flou, normalisation).
- `ImageDataset` : comme `ImageDS` au Level 1, mais le calcul NPR est rangé dans une méthode `_represent()`.
- Le label est maintenant un **entier** (0, 1 ou 2) et non un nombre décimal. La cross-entropy l'exige.

**`make_loader(...)`** — nouveautés :

| Paramètre | Signification |
|---|---|
| `generator` avec `manual_seed(SEED)` | L'ordre de mélange des images est toujours le même |
| `worker_init_fn=seed_worker` | Graine fixe pour chaque processus |
| `persistent_workers=True` | Garde les processus ouverts entre les epochs (plus rapide) |
| `prefetch_factor=2` | Chaque processus prépare 2 batchs à l'avance |

### Cellule 8 — Modèles et moteur d'entraînement

**`DeepModel` — passer de 2 à 3 classes :**

```python
new_head = nn.Linear(classifier.in_features, N_CLASSES)   # nouvelle tête à 3 sorties
new_head.weight[0].copy_(old_weight[real_index])          # classe 0 (real)       ← poids "real"
new_head.weight[1].copy_(old_weight[fake_index])          # classe 1 (generated)  ← poids "fake"
new_head.weight[2].copy_(old_weight[fake_index])          # classe 2 (local_edit) ← poids "fake"
new_head.weight[2].add_(0.001 * torch.randn_like(...))    # + tout petit bruit
```

| Mot-clé | Signification |
|---|---|
| `nn.Linear(in, out)` | Couche « linéaire » : multiplie par une matrice de poids et ajoute un biais |
| `in_features` | Taille du vecteur qui arrive dans la tête |
| `copy_` | Copie des valeurs (le `_` final = modification **sur place**) |
| `torch.randn_like` | Bruit aléatoire de la même forme |
| `torch.no_grad()` | Ces copies ne doivent pas être suivies pour le calcul des gradients |
| **Pourquoi le bruit ?** | Sans bruit, les classes 1 et 2 auraient exactement les mêmes poids, donc les mêmes gradients : elles apprendraient la même chose et ne se sépareraient jamais. Le bruit **casse la symétrie**. |

Je mets aussi à jour `config.num_labels`, `id2label` et `label2id` pour que le modèle sache qu'il a 3 classes nommées.

**`CnnModel`** : `timm.create_model(..., num_classes=3, drop_rate=0.2)`. timm crée directement une tête à 3 sorties.

**`expert_settings(kind)`** : renvoie en une fois la transformation, la taille de batch, l'accumulation et le LR du corps pour un expert. Évite de répéter ces `if`.

**`predict(...)`** — ce qui change :

| Level 1 | Level 2 |
|---|---|
| `torch.sigmoid(z)` → 1 probabilité | `logits.softmax(1)` → **3 probabilités** qui font 1 |
| 5 vues : `0.75 × moyenne + 0.25 × max` | Pareil, puis **`probs / probs.sum(1)`** : on renormalise pour que la somme refasse 1 |

**Softmax** : transforme 3 scores en 3 probabilités positives dont la somme vaut 1. Exemple : scores (2, 1, 0) → probabilités (0.67, 0.24, 0.09).

**`macro_f1(targets, probabilities)`** : prend l'`argmax` (la classe la plus probable), puis `f1_score(..., average="macro")`.

**`fit_one(...)`** — ce qui change :

```python
loss = F.cross_entropy(logits, targets, label_smoothing=LABEL_SMOOTHING) / accum
```

| Mot-clé | Signification |
|---|---|
| **Cross-entropy** | Loss standard pour la classification multi-classe. Elle punit fort quand la bonne classe a une faible probabilité. |
| **Label smoothing 0.02** | Au lieu de viser (0, 0, 1), on vise environ (0.007, 0.007, 0.987). Le modèle n'est jamais poussé à 100 % de certitude. Résultat : moins d'overfitting, probabilités plus raisonnables. |
| Checkpoint | Gardé si le **Macro F1** du fold progresse de plus de 0.0001 |
| `torch.cuda.synchronize()` | Force le GPU à finir son travail. Si une erreur GPU existe, elle apparaît **ici**, comme une erreur Python normale, au lieu de tuer tout le kernel plus tard. |

Le reste (AdamW 2 LR, cosine + warmup, AMP, accumulation, clipping, early stopping) est **identique au Level 1**.

### Cellule 10 — Prédictions OOF

- `oof` : tableau de forme **(7857, 3, 3)** = images × experts × classes.
- `hold` : pareil pour `val/`, moyenne des 3 modèles de fold.
- Contrôles :
  - `valid_probs.shape != (len(valid_index), N_CLASSES)` → erreur si la forme est mauvaise ;
  - `np.isfinite(...)` → pas de `NaN` ni d'infini ;
  - `np.allclose(oof.sum(2), 1.0)` → chaque ligne de probabilités fait bien 1.

### Cellule 12 — Ensemble et décision

**`meta_features(probabilities)`** :
```python
np.log(np.clip(probabilities, 1e-6, 1.0)).reshape(len(clipped), -1)
```
- `clip(1e-6, 1)` : évite `log(0)` ;
- `log` : les log-probabilités sont plus faciles à combiner pour une régression logistique ;
- `reshape(n, -1)` : aplatit 3 experts × 3 classes = **9 colonnes** par image.

**Les biais de décision — l'idée :**

Au lieu de `argmax(probas)`, je fais `argmax(log(probas) + biais)`. Un biais positif sur une classe la rend **plus facile à choisir**.

| Fonction | Rôle |
|---|---|
| `centered_bias(values)` | Soustrait la moyenne des biais. Ajouter la même valeur aux 3 classes ne change pas l'argmax, donc on fixe la moyenne à 0. |
| `apply_bias(probabilities, bias)` | `argmax(log(p) + biais)` → classe prédite |
| `optimize_bias(probabilities, targets)` | Cherche les biais qui maximisent le Macro F1 |

**Comment marche `optimize_bias` :**
1. Départ : biais = (0, 0, 0).
2. Pour chaque classe, j'essaie de modifier son biais avec plusieurs valeurs. Je garde la valeur si le Macro F1 monte.
3. Je fais ça avec 3 grilles, du **grossier au fin** : ±0.45 → ±0.08 → ±0.02.
4. Au maximum 3 passages par grille, j'arrête si plus rien ne s'améliore.

On appelle ça une **recherche par coordonnées** (*coordinate search*) : on améliore un paramètre à la fois.

**Le stacking :**
- 5 folds groupés (`meta_splits`).
- `meta_oof` : prédictions OOF du stacker (pour l'évaluer honnêtement).
- `stack` : le stacker final, entraîné sur tout l'OOF.
- `LogisticRegression(C=0.08)` : avec 3 classes, sklearn fait une régression logistique **multinomiale** (une seule formule qui sort 3 probabilités).

**Les candidats :**

| Candidat | Description |
|---|---|
| `stack` | Régression logistique sur les 9 log-probas |
| `mean` | Moyenne simple des 3 experts |
| `deep_rgb`, `cnn_rgb`, `cnn_npr` | Chaque expert seul |

**Évaluation honnête des biais** : pour chaque candidat et chaque meta-fold, j'apprends les biais sur 4 folds et je les applique au 5ᵉ. Le score est calculé sur ces prédictions croisées. Le biais final est la **médiane** des 5 biais.

**Résultat** : `stack` retenu, biais `[-0.19167, -0.19167, +0.38333]` → **favorise `local_edit`**.

**Contrôle sur `val/`** (ouvert une seule fois) :
- Macro F1 = **0.99591** ;
- `classification_report` : F1 de 0.9969 (real), 0.9954 (generated), 0.9954 (local_edit) ;
- alerte automatique si le holdout est plus de **1,5 point** sous l'OOF. Pas déclenchée.

### Cellule 14 — Réentraînement final

- `train + val` = 8 835 images.
- Epochs = médiane des meilleures epochs OOF (4 pour chaque expert).
- Les 3 experts sont réentraînés, et **tous les 3 servent** (le stack les utilise tous).
- Test : `stack.predict_proba(meta_features(final_test))`, puis `apply_bias(..., FINAL_BIAS)`.

### Cellule 16 — Soumission et audit

- `merge` sur `sample_submission.csv` avec `validate="one_to_one"`.
- Contrôles avec des messages clairs : nombre de lignes, valeurs manquantes, doublons, classes dans {0, 1, 2}.
- **Audit** `audit_level2_v2.json` : candidat choisi, scores de tous les candidats et de chaque fold, biais, F1 par classe, matrice de confusion, epochs, **nombre de prédictions par classe sur le test**, durée en heures, versions.
- Affiche la matrice de confusion avec les noms de classes.

---

## Partie D — Dictionnaire des mots-clés propres au Level 2

| Mot | Explication |
|---|---|
| **Multi-classe** | Plus de 2 classes possibles, une seule réponse par image. |
| **Softmax** | Transforme des scores en probabilités qui font 1 au total. |
| **Argmax** | Indice de la plus grande valeur = classe choisie. |
| **Cross-entropy** | Loss pour la classification multi-classe. |
| **Label smoothing** | Viser un peu moins que 100 % pour la bonne classe. |
| **Macro F1** | Moyenne simple des F1 de chaque classe. |
| **Un-contre-tous** | Pour une classe : elle = positive, les autres = négatives. |
| **Log-probabilité** | `log(p)`. Transforme les multiplications en additions, plus stable. |
| **Biais de décision** | Bonus/malus ajouté à une classe avant l'argmax. |
| **Recherche par coordonnées** | Optimiser un paramètre à la fois, puis passer au suivant. |
| **Régression logistique multinomiale** | Régression logistique qui sort une probabilité par classe. |
| **Casser la symétrie** | Ajouter un peu de bruit pour que deux neurones identiques apprennent des choses différentes. |
| **P2P** | Communication directe entre deux GPU. |
| **Worker** | Processus qui charge les images en parallèle. |
| **`classification_report`** | Tableau sklearn avec précision / rappel / F1 par classe. |

---

## Partie E — Points honnêtes à connaître

1. **Versions non bloquées** dans `%pip install`. Une future version pourrait changer les résultats. Le Level 1 et le Level 3 les bloquent.
2. **DeepGuard est faible** (0.89) : il a été pré-entraîné en binaire et distingue mal « généré » et « retouché ». Il reste utile dans le stack.
3. **L'OOF est un peu optimiste**, car l'epoch est choisie sur le fold qui sert à l'OOF. Mais le holdout confirme (0.9971 → 0.9959, écart de 0.1 point).
4. **Un seul GPU sur deux** à cause des plantages P2P.
5. **Le biais favorise `local_edit`** : c'est un choix appris sur les données. Si le test avait beaucoup moins de retouches locales, ça pourrait créer des faux positifs sur cette classe.

---

## Partie F — Fiche ultra-courte

- **Tâche** : 0 real, 1 generated, 2 local_edit. Métrique : **Macro F1**.
- **Anti-fuite** : pHash en **5 morceaux** (aucune paire ratée) + Union-Find + `StratifiedGroupKFold` + **contrôle automatique**.
- **3 experts à 3 sorties** : DeepGuard (tête 2→3 avec transfert de poids + bruit), ConvNeXt-V2 RGB, ConvNeXt-V2 NPR.
- **Entraînement** : **cross-entropy + label smoothing 0.02**, softmax, le reste comme au Level 1.
- **Décision** : stack (régression logistique sur 9 log-probas) + **biais par classe** (médiane sur 5 folds) → favorise `local_edit`.
- **`val/` une seule fois** : Macro F1 **0.99591**, 4 erreurs sur 978.
- **Final** : réentraînement train + val, stack et biais figés, soumission vérifiée, audit JSON.
