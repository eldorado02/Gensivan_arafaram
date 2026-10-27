# Level 1 — Explication complète de mon code

Ce document complète le [README.md](README.md). Il explique **chaque mot-clé**, **chaque bloc de code** et **chaque modèle** du notebook [level1.ipynb](level1.ipynb).

Objectif : savoir expliquer **ce que fait mon code et pourquoi**, sans être un expert.

---

## Partie A — Où et comment j'ai lancé le notebook

- **Plateforme** : Kaggle Notebook (obligatoire : le règlement demande un notebook Kaggle reproductible).
- **Matériel** : 2 GPU **NVIDIA Tesla T4** (16 Go de mémoire chacun). En pratique, j'en utilise **un seul** (voir `MULTI_GPU_TRAINING`).
- **Données** : le dataset de la compétition est ajouté au notebook. Il est lu dans `/kaggle/input/competitions/gensivana-real-or-fake-level-1-detection`.
- **Sorties** : tout est écrit dans `/kaggle/working` (submission, checkpoints, caches, audit).
- **Pour relancer** : activer le GPU, ajouter le dataset, puis « Run All ». Si le run est coupé, les caches `.npz` et `.pt` permettent de **reprendre** sans tout recalculer (message `RESUME` dans les logs).

**Historique de versions** : ce notebook est la **v7**. La v6 avait un défaut : certains choix (seuil, poids) étaient réglés en regardant `val/`. Le score de `val/` n'était donc plus honnête. La v7 corrige ça : **toutes les décisions sont prises sur le train (OOF)**, et `val/` sert seulement au contrôle final.

---

## Partie B — Les modèles que j'ai utilisés et pourquoi

J'utilise **2 architectures** pour construire **3 experts**.

### 1. DeepGuard — `king1oo1/deepfake-model`

| | |
|---|---|
| D'où il vient | Hugging Face Hub, chargé avec la librairie `transformers` |
| Type | **Vision Transformer** basé sur **SigLIP2** (modèle de Google) |
| Taille | ≈ 90 millions de paramètres (fichier de 372 Mo) |
| Entrée | 224 × 224 pixels |
| Déjà entraîné pour | dire si une image est **real** ou **fake** |

**C'est quoi SigLIP2 ?** Un modèle entraîné sur des milliards de paires image + texte. Il comprend très bien le **sens** d'une image (« c'est un chat sur un canapé »). Il est de la même famille que **CLIP** (OpenAI).

**Pourquoi je l'ai choisi :**
- Il est **déjà spécialisé en deepfake** : il apporte une connaissance « a priori » (un *prior*) sur le réel et le faux.
- Il regarde l'image **globalement**, comme un humain. C'est complémentaire des deux autres experts, qui regardent la texture.
- Inspiration : l'article *Raising the Bar of AI-generated Image Detection with CLIP* (CVPRW 2024) montre que les modèles de type CLIP détectent bien les images IA.

**Comment je l'entraîne** : je **gèle** la majorité du modèle (`requires_grad = False`). J'entraîne seulement la tête de classification et les **6 derniers blocs**. Comme ça, je ne casse pas ce qu'il sait déjà et ça demande moins de mémoire.

**Résultat** : c'est le plus faible des 3 (F1 OOF ≈ 0.923). Il est bon sur le sens global, mais il rate les traces fines.

### 2. ConvNeXt-V2 Tiny — `convnextv2_tiny.fcmae_ft_in22k_in1k_384`

Ce nom long se lit morceau par morceau :

| Morceau | Signification |
|---|---|
| `convnextv2` | Architecture **ConvNeXt V2** (Meta, 2023). Un **CNN** (réseau à convolutions) modernisé. |
| `tiny` | La plus petite version (≈ 28 millions de paramètres, fichier de 115 Mo). |
| `fcmae` | Pré-entraîné en **auto-supervisé** : on cache des morceaux de l'image et le modèle apprend à les reconstruire (*Fully Convolutional Masked AutoEncoder*). |
| `ft_in22k_in1k` | Puis **fine-tuné** sur ImageNet-22k (14 millions d'images, 22 000 classes), puis sur ImageNet-1k. |
| `384` | Entraîné à la résolution **384 × 384**. |

Chargé avec la librairie **`timm`**.

**Pourquoi je l'ai choisi :**
- Un **CNN** est très bon pour les **textures et détails locaux**. C'est là que se cachent les traces de génération.
- **384 px** au lieu de 224 : on garde plus de détails fins.
- **Tiny** : assez petit pour tenir sur un T4 et finir dans le temps Kaggle, assez puissant pour ce problème.
- Pré-entraîné sur beaucoup d'images : il démarre avec de bonnes bases.

Je l'utilise **deux fois**, avec deux entrées différentes :

- **Expert B — `cnn_rgb`** : l'image normale en couleurs.
- **Expert C — `cnn_npr`** : l'image transformée en **NPR** (voir plus bas). Inspiration : l'article **NPR** (CVPR 2024) sur les traces d'*upsampling* des générateurs.

**Résultat** : `cnn_rgb` F1 OOF ≈ 0.9953, **`cnn_npr` ≈ 0.9988 → le meilleur, c'est lui qui est retenu.**

### 3. Régression logistique — le « stacker »

Un modèle très simple de `scikit-learn`. Il prend les 3 réponses des experts et apprend à les combiner.

**Pourquoi un modèle aussi simple ?** Il n'a que 3 nombres en entrée. Un modèle compliqué ferait de l'**overfitting** (il apprendrait le bruit). Avec `C=0.08`, je le régularise fortement : il reste prudent.

Inspiration générale de la combinaison sémantique + traces bas niveau : **AIDE** (ICLR 2025), qui mélange elle aussi un modèle de type CLIP et des détails de pixels.

**Résultat** : le stack fait 0.9981, un peu moins bien que NPR seul. La sélection automatique a donc choisi NPR seul.

---

## Partie C — Le code, cellule par cellule

### Cellule 1 — Installation

```python
%env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
!pip install -q "transformers>=4.53,<5" "timm>=1.0.15,<2" "scikit-learn>=1.4" imagehash
```

- `%env ...` : variable d'environnement. `expandable_segments:True` aide PyTorch à mieux gérer la mémoire GPU (moins de fragmentation, moins d'erreurs « out of memory »).
- `!pip install -q` : installe des librairies. `-q` = silencieux.
- Les versions sont **bornées** (`>=4.53,<5`) pour la **reproductibilité** : une nouvelle version majeure pourrait casser le code.

### Cellule 2 — Imports et configuration

**Les librairies :**

| Librairie | À quoi elle sert dans mon code |
|---|---|
| `numpy` (`np`) | Calcul sur des tableaux de nombres |
| `pandas` (`pd`) | Tableaux de données (lire les CSV, `train_df`…) |
| `PIL` (`Image`) | Ouvrir les images |
| `imagehash` | Calculer les pHash (empreintes d'images) |
| `torch` | **PyTorch**, la librairie de deep learning |
| `torch.nn` (`nn`) | Les briques des réseaux de neurones |
| `torch.nn.functional` (`F`) | Fonctions (loss, pooling, interpolation…) |
| `torchvision.transforms` | Transformations d'images (resize, flip…) |
| `timm` | Bibliothèque de modèles de vision pré-entraînés (ConvNeXt-V2) |
| `transformers` | Bibliothèque Hugging Face (DeepGuard / SigLIP2) |
| `sklearn` | **scikit-learn** : régression logistique, métriques, folds |
| `gc` | *Garbage collector* : libérer la mémoire |
| `json`, `platform` | Écrire l'audit, noter les versions |

**Les constantes de configuration :**

| Constante | Valeur | Signification |
|---|---|---|
| `SEED` | 20260723 | Graine aléatoire. Même graine = mêmes résultats. |
| `N_FOLDS` | 3 | Nombre de folds de validation croisée |
| `DEVICE` | `'cuda'` | Calcul sur GPU (`cuda` = GPU NVIDIA), sinon `cpu` |
| `AMP` | True | Active la *mixed precision* (calcul en float16) |
| `OUT` | `/kaggle/working` | Dossier de sortie |
| `FAST` | False | Mode test rapide (900 images, 2 epochs). Désactivé pour le vrai run. |
| `DEEP_NAME` | `king1oo1/deepfake-model` | Nom du modèle DeepGuard |
| `CNN_NAME` | `convnextv2_tiny...384` | Nom du ConvNeXt-V2 |
| `IMG_DEEP`, `IMG_CNN` | 224, 384 | Taille des images en entrée |
| `MULTI_GPU_TRAINING` | False | Pas de multi-GPU : le lien entre les 2 T4 de Kaggle était instable et faisait planter le kernel |
| `BATCH_DEEP`, `BATCH_CNN` | 12, 4 | Images par mini-batch |
| `ACCUM_DEEP`, `ACCUM_CNN` | 4, 6 | Accumulation de gradients → batch effectif 48 et 24 |
| `NUM_WORKERS` | 0 | Chargement des images dans le processus principal (plus lent mais plus stable) |
| `EPOCHS_DEEP`, `EPOCHS_CNN` | 5, 5 | Nombre maximum d'epochs |
| `PATIENCE` | 2 | Early stopping après 2 epochs sans progrès |
| `RUN_TAG` | `v7c_safe_s20260723_f3` | Nom du run, utilisé dans les noms de fichiers de cache |
| `FORCE_RETRAIN` | False | Si True, ignore les caches et réentraîne tout |

**Autres lignes :**

- `ImageFile.LOAD_TRUNCATED_IMAGES = True` : accepte d'ouvrir une image même si le fichier est un peu abîmé, au lieu de planter.
- `seed_all()` : fixe la graine de `random`, `numpy`, `torch` et CUDA.
- `cudnn.benchmark = False` et `cudnn.deterministic = True` : force le GPU à faire **toujours les mêmes calculs**. Un peu plus lent, mais reproductible.

### Cellule 4 — Données et groupes anti-doublons

**`find_root()`** : cherche le dossier du dataset parmi plusieurs chemins possibles (Kaggle, Colab, local). Si rien n'est trouvé : erreur claire.

**`read_split(name, labelled)`** :
- lit `ground_truth.csv` (train, val) ou `sample_submission.csv` (test) ;
- renomme la colonne du label en `target` ;
- construit le chemin de chaque image (`path`) ;
- **vérifie** qu'aucune image ne manque (`assert`) et que les labels sont bien 0 ou 1.

Sortie : `7857 978 993 {1: 0.667, 0: 0.333}` → tailles de train / val / test, et 2/3 de truquées.

**Le pHash et les groupes :**

| Mot-clé | Signification |
|---|---|
| **pHash** (*perceptual hash*) | Empreinte de 64 bits qui résume l'apparence de l'image. Deux images qui se ressemblent ont des empreintes proches. Écrite en 16 caractères hexadécimaux. |
| **Cache** `train_phash.npy` | Les pHash sont sauvegardés pour ne pas les recalculer au prochain run. |
| **Distance de Hamming** | Nombre de bits différents entre deux empreintes. Calculée avec `(a ^ b).bit_count()` : `^` (XOR) met un 1 là où les bits diffèrent, `bit_count()` compte les 1. |
| **Seuil ≤ 4** | Si 4 bits ou moins diffèrent, les deux images sont considérées comme quasi-doublons. |
| **Buckets (seaux)** | Comparer toutes les paires = 30 millions de comparaisons. J'ai découpé chaque empreinte en 4 morceaux de 4 caractères. Je compare seulement les images qui partagent un morceau identique. Beaucoup plus rapide. |
| `if len(ids) > 80: continue` | Si un seau est trop rempli, je le saute pour éviter une explosion du temps de calcul. |
| **Union-Find** | Algorithme pour fusionner des éléments en groupes. `find(x)` donne le « chef » du groupe de x. `union(a, b)` fusionne deux groupes. |
| **Compression de chemin** | `parent[x] = parent[parent[x]]` : raccourcit les chemins pour que `find` soit rapide. |
| `groups` | Pour chaque image, le numéro de son groupe. |

Sortie : `pHash groups 6933 largest 6` → 6 933 groupes, le plus gros contient 6 images.

**`StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED)`** :
- **K-Fold** : coupe les données en K = 3 parties.
- **Stratified** : chaque partie garde la proportion 1/3 – 2/3.
- **Group** : un groupe n'est jamais coupé entre deux parties. → **pas de fuite**.
- `shuffle=True` + `random_state` : mélange, mais toujours de la même façon.

### Cellule 6 — Transformations et Dataset

**Normalisation :**

- `IM_MEAN = (0.485, 0.456, 0.406)` et `IM_STD = (0.229, 0.224, 0.225)` : moyenne et écart-type des couleurs d'**ImageNet**. ConvNeXt a appris avec ces valeurs, donc je normalise pareil : `(pixel − mean) / std`.
- `deep_processor.image_mean / image_std` : DeepGuard a ses propres valeurs, je les lis directement depuis le modèle.

**`make_tf(size, mean, std, train)`** construit la suite de transformations.

En entraînement (`train=True`), ce sont les **augmentations** (des variations aléatoires pour rendre le modèle robuste) :

| Transformation | Ce qu'elle fait |
|---|---|
| `RandomResizedCrop(size, scale=(0.55, 1.0))` | Prend un morceau aléatoire de 55 % à 100 % de l'image, puis le redimensionne |
| `ratio=(0.75, 1.333)` | Le morceau peut être un peu plus large ou plus haut |
| `InterpolationMode.BICUBIC` | Méthode de redimensionnement de bonne qualité |
| `RandomHorizontalFlip()` | Retourne l'image gauche-droite (une fois sur deux) |
| `ColorJitter(0.1, 0.1, 0.06, 0.015)`, `p=0.3` | Change un peu luminosité, contraste, saturation, teinte (30 % du temps) |
| `RandomChoice([identité, GaussianBlur])` | Une fois sur deux, ajoute un léger flou |
| `ToTensor()` | Transforme l'image en **tenseur** PyTorch (valeurs entre 0 et 1) |
| `Normalize(mean, std)` | Normalisation |

En évaluation : seulement `Resize` + `ToTensor` + `Normalize`. **Pas d'aléatoire**, pour que les prédictions soient stables.

**Classe `ImageDS(Dataset)`** : explique à PyTorch comment lire **une** image.

- `__len__` : nombre d'images.
- `__getitem__(i)` : ouvre l'image `i`, la convertit en RGB, applique les transformations, renvoie `(x, y, i)` = (image, label, numéro).
- `multiview=True` : ajoute 4 crops de 72 % dans les 4 coins → 5 vues au total.
- `mode='npr'` : calcule la représentation NPR.

**Le calcul NPR, ligne par ligne :**

```python
raw = x.clone()                                   # copie de l'image normalisée
up = F.interpolate(F.avg_pool2d(raw[None], 2),    # 1) moyenne par carrés de 2×2 → image 2× plus petite
                   size=raw.shape[-2:],           # 2) ré-agrandie à la taille d'origine
                   mode='nearest')[0]             #    en copiant chaque pixel (nearest)
npr = raw - up                                    # 3) différence = les micro-détails entre voisins
x = torch.tanh(2.5 * npr)                         # 4) amplifie (×2.5) puis écrase entre -1 et 1
```

**Pourquoi ça marche ?** Les générateurs d'images IA fabriquent l'image en l'agrandissant plusieurs fois (*upsampling*). Ça crée des **motifs réguliers entre pixels voisins**, invisibles à l'œil. NPR enlève le contenu (le chat, le canapé) et garde seulement ces motifs.

⚠️ Petite différence à connaître : le texte markdown de la cellule 5 parle de `pixel_shuffle`. **Le code réel** utilise `avg_pool2d` + `interpolate(nearest)`. C'est le code qui compte. L'idée reste la même : comparer chaque pixel à ses voisins.

**`loader(...)`** crée un **DataLoader** : il prépare les images par paquets (batchs).

| Paramètre | Signification |
|---|---|
| `batch_size` | Nombre d'images par paquet |
| `shuffle` | Mélanger les images (oui à l'entraînement) |
| `num_workers` | Nombre de processus pour charger les images |
| `pin_memory` | Accélère le transfert CPU → GPU |
| `drop_last` | Jette le dernier paquet s'il est incomplet (à l'entraînement) |

### Cellule 8 — Modèles et moteur d'entraînement

**`unwrap(m)`** : si le modèle est enveloppé dans `nn.DataParallel` (multi-GPU), renvoie le vrai modèle à l'intérieur.

**`class DeepModel(nn.Module)`** :
- charge DeepGuard avec `AutoModelForImageClassification.from_pretrained(DEEP_NAME)` ;
- lit `id2label` (la correspondance numéro → nom de classe) pour trouver quelle sortie veut dire « fake » ;
- `forward(x)` : renvoie **un seul logit** = `logit_fake − logit_real`. Plus il est grand, plus l'image est suspecte.

**`class CNNModel(nn.Module)`** :
- `timm.create_model(CNN_NAME, pretrained=True, num_classes=1, drop_rate=0.2)` ;
- `num_classes=1` : une seule sortie (le logit « manipulée ») ;
- `drop_rate=0.2` : **dropout** de 20 %. On éteint au hasard des neurones à l'entraînement pour éviter l'overfitting.
- `.flatten()` : met le résultat à plat (un nombre par image).

**`freeze_deep(model)`** : met `requires_grad = False` partout (= gelé, pas appris), puis le remet à `True` pour la tête, les 6 derniers blocs, `post_layernorm` et `head`.

**`make_model(kind)`** : crée l'expert voulu et l'envoie sur le GPU avec `.to(DEVICE)`.

**`predict(model, df, kind, mode)`** : calcule une probabilité pour chaque image.

| Mot-clé | Signification |
|---|---|
| `model.eval()` | Mode évaluation : désactive dropout et aléatoire |
| `torch.inference_mode()` | Pas de calcul de gradients → plus rapide, moins de mémoire |
| `torch.autocast(..., dtype=torch.float16)` | Calcul en demi-précision (AMP) |
| `torch.sigmoid(z)` | Transforme un logit en probabilité entre 0 et 1 |
| `0.75 * pv.mean(1) + 0.25 * pv.max(1).values` | Pour NPR : combine les 5 vues (moyenne + un peu du maximum) |

**`best_threshold(y, p)`** : teste **361 seuils** entre 0.05 et 0.95, calcule le F1 pour chacun, renvoie le meilleur seuil et son F1.

**`fit_one(kind, mode, tr, va, fold, epochs)`** : entraîne **un** expert sur **un** fold. C'est le cœur de l'entraînement.

| Mot-clé | Signification |
|---|---|
| **Learning rate (LR)** | Taille des pas quand le modèle corrige ses poids |
| `body_lr` = 8e-6 (deep) / 2e-5 (cnn) | LR **petit** pour le corps pré-entraîné : on le modifie doucement |
| `head_lr` = 2e-4 | LR **plus grand** pour la tête, qui est nouvelle |
| **AdamW** | Optimiseur (algorithme qui met à jour les poids). Version d'Adam avec un meilleur *weight decay*. |
| `weight_decay=0.0002` | Pénalise les poids trop grands → régularisation |
| `get_cosine_schedule_with_warmup` | Le LR **monte** pendant les 8 premiers % des steps (*warmup*), puis **descend en cosinus** jusqu'à 0 |
| **Step** | Une mise à jour des poids |
| `GradScaler` | Évite que les petits gradients deviennent 0 en float16 |
| `F.binary_cross_entropy_with_logits(z, y)` | **BCE** : la loss pour un problème binaire. Grande si le modèle se trompe avec confiance. |
| `/ accum` | Divise la loss pour l'accumulation de gradients |
| `.backward()` | **Rétropropagation** : calcule les gradients (dans quel sens corriger chaque poids) |
| `(step + 1) % accum == 0` | Tous les `accum` mini-batchs, on fait une vraie mise à jour |
| `clip_grad_norm_(..., 1.0)` | **Gradient clipping** : limite la taille des gradients pour éviter des sauts brutaux |
| `scaler.step(opt)` | Mise à jour des poids |
| `opt.zero_grad()` | Remet les gradients à zéro avant le prochain calcul |
| `sch.step()` | Avance le scheduler (change le LR) |
| `torch.cuda.empty_cache()` | Libère la mémoire GPU inutilisée |
| **Checkpoint** (`.pt`) | Sauvegarde des poids du modèle (`state_dict`) |
| `score > best + 0.0001` | Si le F1 progresse, je sauvegarde ce checkpoint |
| **Early stopping** | Si pas de progrès pendant `PATIENCE` = 2 epochs → j'arrête |
| `torch.load(ck, weights_only=True)` | Recharge le meilleur checkpoint. `weights_only=True` = chargement sécurisé (seulement des poids, pas de code). |

À la fin, la fonction renvoie le meilleur modèle, l'historique et le numéro de la meilleure epoch.

### Cellule 10 — Entraînement OOF des 3 experts

```python
experts = [('deep', 'rgb', 5), ('cnn', 'rgb', 5), ('cnn', 'npr', 5)]
```

Pour **chaque expert** et **chaque fold** (3 × 3 = 9 entraînements) :

1. `tr` = 2 folds pour entraîner, `va` = 1 fold pour valider.
2. Si le cache `..._pred.npz` existe → je recharge (`RESUME`). Sinon j'entraîne avec `fit_one`.
3. Je prédis `va` → ça remplit `oof[iv, j]` (**OOF** = *out-of-fold*).
4. Je prédis `val/` → `hold[:, j] += pvh / N_FOLDS` (moyenne des 3 modèles).
5. Je sauvegarde dans un `.npz` (`np.savez_compressed`).
6. Je libère la mémoire (`model.cpu()`, `del`, `gc.collect()`).

Résultat : un tableau `oof` de taille `7857 × 3` (une proba par image et par expert), sans aucune triche.

### Cellule 12 — Choix du modèle et du seuil

**`logit(p)`** : `log(p / (1 − p))`. `np.clip(p, 1e-5, 1 − 1e-5)` évite `log(0)` qui donne l'infini.

**Le stacking :**
- `make_pipeline(StandardScaler(), LogisticRegression(C=0.08))` :
  - **Pipeline** : enchaîne plusieurs étapes en un seul objet ;
  - **StandardScaler** : centre et réduit chaque colonne (moyenne 0, écart-type 1) ;
  - **LogisticRegression** : le modèle de combinaison ;
  - `C=0.08` : **petit C = forte régularisation** ;
  - `max_iter=2000` : nombre max d'itérations pour converger.
- `meta_cv` : 5 folds groupés pour évaluer le stack lui-même en OOF (`meta_oof`).
- `predict_proba(...)[:, 1]` : probabilité de la classe 1 (manipulée).
- `stack` final : entraîné sur tout l'OOF, utilisé pour `val/` et `test/`.

**Sélection automatique :**
1. 4 candidats : `stack`, `deep_rgb`, `cnn_rgb`, `cnn_npr`.
2. Pour chacun : meilleur seuil sur chacun des 5 folds → **médiane** des 5 seuils.
3. F1 de chaque candidat avec ce seuil médian.
4. `SELECTED = max(...)` → **`cnn_npr`**, `THRESH = 0.4425`.

**Contrôle sur `val/` :**
- `f1_score` → **0.99847**
- `balanced_accuracy_score` = (rappel classe 0 + rappel classe 1) / 2 → **0.99693**
- `confusion_matrix` → `[[324, 2], [0, 652]]`
- Alerte automatique si le F1 de `val/` est plus de 4 points sous l'OOF (signe d'overfitting). Pas déclenchée.

### Cellule 14 — Réentraînement final

- `all_df = pd.concat([train_df, val_df])` : train + val = 8 835 images.
- Epochs = **médiane des meilleures epochs** trouvées en OOF (deep 4, cnn_rgb 4, cnn_npr 3).
- Même recette d'entraînement que `fit_one`, mais **sans validation** (toutes les données servent à apprendre).
- Prédiction du test pour les 3 experts, puis je prends le candidat choisi (`cnn_npr`) et le seuil figé.
- `pred = (p_test >= THRESH).astype(int)` → 0 ou 1.

### Cellule 16 — Soumission et audit

- `merge(..., validate='one_to_one')` : joint mes prédictions sur le `sample_submission.csv` officiel. `one_to_one` vérifie qu'il n'y a pas de doublon.
- `assert` : bon nombre de lignes, pas de vide, valeurs dans {0, 1}.
- `sub.to_csv('submission.csv', index=False)` : écrit le fichier.
- **Audit** `audit_level1_v7.json` : seed, modèles, tailles, scores OOF et holdout, seuils, coefficients du stack, epochs, versions des librairies. C'est la **preuve** que le run est reproductible.
- `ensemble_scores_level1_v7.npz` : toutes les probabilités, pour analyser les erreurs plus tard.

---

## Partie D — Dictionnaire général des mots-clés

### Bases du machine learning

| Mot | Explication |
|---|---|
| **Machine learning** | Le programme apprend à partir d'exemples au lieu de règles écrites à la main. |
| **Deep learning** | Machine learning avec des réseaux de neurones à beaucoup de couches. |
| **Classification binaire** | Choisir entre 2 classes (0 ou 1). |
| **Label / target** | La bonne réponse connue pour un exemple. |
| **Train / validation / test** | Apprendre / vérifier / prédire pour la soumission. |
| **Overfitting** | Le modèle apprend par cœur et marche mal sur de nouvelles données. |
| **Régularisation** | Techniques qui empêchent l'overfitting (dropout, weight decay, `C` petit…). |
| **Data leakage (fuite)** | Information du test qui « fuit » dans l'entraînement. Score faussement bon. |
| **Validation croisée (K-Fold)** | Couper en K parties, chaque partie sert une fois de validation. |
| **OOF (out-of-fold)** | Prédiction d'une image par un modèle qui ne l'a pas vue. |
| **Holdout** | Données mises de côté jusqu'à la fin. |
| **Hyperparamètre** | Réglage choisi par moi, pas appris (LR, epochs, seuil…). |
| **Reproductibilité** | Relancer = obtenir les mêmes résultats. |

### Réseaux de neurones

| Mot | Explication |
|---|---|
| **Poids / paramètres** | Les nombres que le réseau apprend. |
| **Tenseur** | Tableau de nombres à plusieurs dimensions (une image = 3 × hauteur × largeur). |
| **CNN** | Réseau à convolutions : de petits filtres glissent sur l'image. Bon pour les textures. |
| **Transformer / ViT** | Réseau qui découpe l'image en petits carrés et regarde les liens entre eux. Bon pour le sens global. |
| **Pré-entraîné** | Déjà entraîné sur beaucoup de données. |
| **Fine-tuning** | Continuer l'entraînement sur mes données. |
| **Transfer learning** | Réutiliser ce qu'un modèle a appris ailleurs. |
| **Geler (freeze)** | Empêcher certaines couches d'être modifiées. |
| **Tête (head)** | Dernière couche qui donne la réponse finale. |
| **Corps (body / backbone)** | Tout le reste, qui extrait les caractéristiques. |
| **Logit** | Score brut avant la sigmoid. |
| **Sigmoid** | Transforme un logit en probabilité entre 0 et 1. |
| **Loss** | Mesure de l'erreur, que l'on cherche à minimiser. |
| **Gradient** | Direction dans laquelle changer les poids pour réduire la loss. |
| **Rétropropagation** | Calcul des gradients de la fin vers le début du réseau. |
| **Optimiseur** | Algorithme qui met à jour les poids (ici AdamW). |
| **Epoch** | Un passage complet sur les données. |
| **Batch** | Petit paquet d'images traitées ensemble. |
| **Batch effectif** | `batch × accumulation`. |
| **Dropout** | Éteindre des neurones au hasard pendant l'entraînement. |
| **Augmentation** | Modifier aléatoirement les images pour rendre le modèle robuste. |

### GPU et performance

| Mot | Explication |
|---|---|
| **GPU** | Carte graphique, très rapide pour le deep learning. |
| **CUDA** | Technologie NVIDIA pour calculer sur GPU. |
| **T4** | Modèle de GPU NVIDIA (16 Go) fourni par Kaggle. |
| **AMP / mixed precision** | Calculer en float16 (moitié moins de mémoire) quand c'est possible. |
| **float16 / float32** | Nombres à 16 ou 32 bits. 16 = plus rapide, moins précis. |
| **OOM** | *Out Of Memory* : plus de mémoire GPU. |
| **DataParallel** | Utiliser plusieurs GPU en même temps (désactivé ici). |

### Métriques

| Mot | Explication |
|---|---|
| **TP / FP / FN / TN** | Vrai positif / faux positif / faux négatif / vrai négatif. |
| **Précision** | Parmi ce que j'ai dit « truqué », combien le sont vraiment. |
| **Rappel** | Parmi les vrais truqués, combien j'en ai trouvé. |
| **F1** | Moyenne harmonique de précision et rappel. |
| **Balanced accuracy** | Moyenne des rappels des deux classes. |
| **Matrice de confusion** | Tableau vraie classe × classe prédite. |
| **Seuil** | Valeur qui transforme une proba en 0 ou 1. |

---

## Partie E — Points honnêtes à connaître (si on te pousse)

1. **L'OOF est un tout petit peu optimiste** : dans `fit_one`, je choisis la meilleure epoch sur le même fold qui sert ensuite à l'OOF. Mais le score de `val/` (jamais vu, 0.99847) est presque identique (OOF 0.99876) : l'effet est négligeable.
2. **Les 3 experts sont réentraînés à la fin**, alors que seul `cnn_npr` est utilisé pour le test. C'est du temps GPU en plus. Je l'ai gardé pour pouvoir comparer les candidats dans l'audit.
3. **La recherche de doublons** coupe l'empreinte en 4 morceaux. Une paire avec exactement 4 bits différents, un dans chaque morceau, peut être ratée. Corrigé au Level 2 (5 morceaux).
4. **Le texte de la cellule 5** parle de `pixel_shuffle`, mais le code utilise `avg_pool2d` + `interpolate`. L'idée est la même.
5. **Un seul GPU utilisé sur deux**, à cause de plantages en multi-GPU. Plus lent, mais stable.

---

## Partie F — Fiche ultra-courte à relire avant l'entretien

- **Tâche** : 0 = vraie, 1 = truquée. Métrique : F1.
- **Anti-fuite** : pHash + Union-Find → groupes → `StratifiedGroupKFold` à 3 folds.
- **3 experts** : DeepGuard/SigLIP2 (global), ConvNeXt-V2 RGB (texture), ConvNeXt-V2 NPR (traces de pixels).
- **NPR** : image − image réduite puis ré-agrandie → micro-détails laissés par les générateurs.
- **Entraînement** : BCE, AdamW 2 LR, cosine + warmup, AMP, accumulation, clipping, early stopping.
- **Décisions sur l'OOF uniquement** : candidat `cnn_npr`, seuil médian 0.4425.
- **Contrôle `val/` une seule fois** : F1 0.99847, 2 erreurs sur 978.
- **Final** : réentraînement train + val, même seuil, soumission vérifiée, audit JSON.
