# Level 3 — Explication complète de mon code

Ce document complète le [README.md](README.md). Il explique **chaque mot-clé**, **chaque bloc de code** et **chaque modèle** du notebook [level3.ipynb](level3.ipynb).

Les notions de base (CNN, epoch, loss, AdamW, pHash, Union-Find…) sont expliquées dans [../level1/EXPLICATION_CODE.md](../level1/EXPLICATION_CODE.md). Le Level 3 est **très différent** des deux premiers : ce n'est plus de la classification mais de la **segmentation** (une réponse pour chaque pixel).

---

## Partie A — Où et comment j'ai lancé le notebook

- **Plateforme** : Kaggle Notebook.
- **Matériel** : **un seul GPU Tesla T4**. Le notebook **refuse de tourner** s'il n'y a pas exactement 1 GPU visible.
- **Données** : `/kaggle/input/competitions/gensivana-real-or-fake-level-3-localization`.
- **Sorties** : `/kaggle/working`.
- **Limite de temps** : moins de 7 h. Le notebook surveille lui-même l'horloge (voir les budgets de temps).
- **Durée réelle** : **5,7 h**, pic de mémoire GPU 2,7 Go.
- **Origine** : notebook « autonome » dérivé de ma version **V2**, qui avait obtenu **0.87076** sur le leaderboard public. Il compare cette V2 à une V3 améliorée et choisit tout seul la meilleure.

---

## Partie B — Les modèles et pourquoi

### Le modèle principal : ForensicFPN (construit par moi)

Ce n'est pas un modèle téléchargé tel quel. Je l'ai **assemblé** à partir de 3 briques :

| Brique | Rôle | Pourquoi |
|---|---|---|
| **ConvNeXt-V2 Tiny** (`convnextv2_tiny.fcmae_ft_in22k_in1k_384`, pré-entraîné) | Encodeur : comprend le **contenu** de l'image | Un CNN moderne, bon sur les détails, assez léger pour un T4 |
| **Branche SRM** (filtres fixes + petites convolutions) | Lit le **bruit** de l'image | Une zone retouchée n'a souvent pas le même bruit que le reste de la photo |
| **FPN** (*Feature Pyramid Network*) | Combine les informations de **toutes les échelles** | Le global aide à comprendre, le fin aide à dessiner des bords précis |

Et **3 sorties** : un **masque** (chaque pixel est-il retouché ?), un **contour** (les bords de la zone) et une **classe** (l'image entière est-elle retouchée ?).

### Deux versions comparées

| | V2 baseline | V3 enhanced |
|---|---|---|
| Résolution | 512 px | 640 px |
| SRM | moyenne simple des couleurs | vraie luminance + bords « reflect » |
| Classifieur image | branche SRM seule | moyenne SRM + RGB |
| Têtes auxiliaires | non | oui (3) |
| Tête de raffinement | non | oui |
| Loss | de base | + Lovász + contour équilibré + auxiliaires |

**Résultat** : V3 gagne (0.9392 contre 0.9307 sur `selection`), avec 0 fausse alerte.

### Pourquoi pas un modèle de segmentation « tout fait » (U-Net…) ?

Un modèle classique ne voit que le contenu. Une bonne retouche a un contenu **crédible**. Il faut aussi regarder les **traces techniques** (le bruit). C'est le rôle de la branche SRM, que les modèles classiques n'ont pas.

---

## Partie C — Le code, cellule par cellule

### Cellule 1 — Installation

```python
%pip install -q imagehash==4.3.2 timm==1.0.26
%pip install -q albumentations==2.0.8 opencv-python-headless
```

- Versions **bloquées** (`==`) → reproductible.
- **albumentations** : librairie d'augmentations qui transforme **l'image et le masque ensemble** (indispensable en segmentation).
- **opencv-python-headless** : OpenCV (`cv2`), la grande librairie de traitement d'images. *headless* = sans interface graphique (pas besoin sur un serveur).

### Cellule 2 — Imports et configuration

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
```

- `CUDA_VISIBLE_DEVICES = "0"` : PyTorch ne voit que le 1er GPU. Ça doit être écrit **avant** `import torch`. C'est pour ça que les imports suivants ont `# noqa: E402` (ça dit à l'outil de style : « oui, je sais, les imports ne sont pas en haut, c'est voulu »).
- `expandable_segments` : meilleure gestion de la mémoire GPU.

**Nouvelles librairies :** `copy` (copier un modèle pour l'EMA), `zipfile` (compresser les cartes), `cv2` (OpenCV), `albumentations` (`A`), `WeightedRandomSampler` (échantillonnage équilibré).

**Constantes :**

| Constante | Valeur | Signification |
|---|---|---|
| `BASE_SIZE`, `V3_SIZE`, `EVAL_SIZE` | 512, 640, 640 | Résolution V2, V3, et taille des cartes pour l'évaluation |
| `DEV_EPOCHS`, `FINAL_EPOCHS` | 6, 8 | Epochs pour la comparaison et pour le final |
| `MIN_V3_GAIN` | 0.002 | Gain minimum pour préférer V3 à V2 |
| `MIN_TTA_GAIN` | 0.001 | Gain minimum pour utiliser 8 vues au lieu de 2 |
| `MIN_RETRIEVAL_GAIN` | 0.0005 | Gain minimum pour garder le retrieval |
| `MAX_COMPARE_HOURS` | 3.75 | Si dépassé avant V3 → arrêt |
| `MAX_REFIT_START_HOURS` | 4.25 | Le réentraînement final doit commencer avant |
| `MAX_START_EPOCH_HOURS` | 5.50 | Pas de nouvelle epoch finale après (sauf les 4 premières) |
| `MAX_TOTAL_HOURS` | 6.75 | Limite avant la prédiction du test |
| `POLYGON_SEPARATOR` | `chr(59)` | Le caractère `;` (code 59), séparateur entre polygones |
| `WINNER_SOURCE_SHA256` | `bd5dae…` | **Empreinte SHA-256** du notebook V2 gagnant. Elle prouve de quelle version exacte je suis parti (traçabilité). |
| `WARNINGS`, `PHASE_TIMES` | listes / dict | Avertissements et durées de chaque phase, écrits dans l'audit |

`elapsed_hours()` : nombre d'heures depuis le début du notebook.

### Cellule 4 — Données et codec polygone ↔ masque

| Fonction | Rôle |
|---|---|
| `is_blank(value)` | Vrai si la valeur est vide ou ne contient que des espaces |
| `parse_polygons(value)` | Coupe le texte sur `;`, puis lit les nombres par paires (x, y). Garde seulement les polygones valides (≥ 3 points, nombre pair de valeurs, valeurs finies), ramenés entre 0 et 1. |
| `polygon_to_mask(value, h, w)` | Multiplie x par la largeur et y par la hauteur, arrondit, puis remplit le polygone avec `cv2.fillPoly` → masque de 0 et 1 |
| `mask_to_polygon(mask, epsilon, min_area)` | L'inverse : trouve les contours, les simplifie, les normalise, les colle avec `;`. Masque vide → `" "`. |
| `official_iou(truth, prediction)` | IoU avec la règle officielle : si l'union est vide (les deux masques vides), IoU = 1 |

**Mots-clés OpenCV dans `mask_to_polygon` :**

| Mot-clé | Signification |
|---|---|
| `cv2.findContours` | Trouve le contour de chaque tache blanche du masque |
| `RETR_EXTERNAL` | Garde seulement les contours **extérieurs** (pas les trous à l'intérieur) |
| `CHAIN_APPROX_SIMPLE` | Compresse les lignes droites (garde seulement les coins) |
| `cv2.contourArea` | Surface d'un contour. Les contours de moins de 3 pixels sont ignorés. |
| `cv2.arcLength` | Périmètre du contour |
| `cv2.approxPolyDP` | Simplifie le contour avec moins de points. `epsilon × périmètre` = tolérance. `epsilon = 0` → pas de simplification. |
| `f"{x:.6f}"` | Écrit chaque coordonnée avec 6 chiffres après la virgule |

**`read_split(root, name, labelled)`** :
- `keep_default_na=False` : **important**. Sans ça, pandas transformerait les cases vides en `NaN`. Ici, une case vide doit rester une chaîne vide (= pas de masque).
- Pour le test : `polygon = " "` (pas de vérité terrain).
- Lit la hauteur et la largeur de chaque image (toutes en **640 × 640**).
- `has_mask` : vrai si l'image a un polygone (= Cat 2).

**Tests automatiques du codec :**
```python
test_mask[2:12, 3:18] = 1      # rectangle 1
test_mask[22:38, 31:49] = 1    # rectangle 2
assert official_iou(test_mask, polygon_to_mask(encoded, 41, 53)) > 0.90   # aller-retour fidèle
assert mask_to_polygon(np.zeros((8, 9), np.uint8)) == " "                  # vide → " "
assert POLYGON_SEPARATOR in encoded                                        # 2 polygones séparés par ;
```

C'est un **test unitaire** : il vérifie que mes fonctions marchent **avant** de lancer des heures de calcul.

### Cellule 6 — Groupes perceptuels et splits

**`compute_hashes(frame, cache_name)`** : calcule les pHash, ou les recharge depuis un cache `.npy`.

**`perceptual_groups(hashes, maximum_distance=4)`** : Union-Find comme avant, mais l'empreinte est coupée en **8 morceaux de 8 bits** (`(value >> shift) & 255`). Avec ≤ 4 bits différents, au moins 4 morceaux sont identiques : aucune paire ratée.

**Les 3 sous-ensembles :**

```
train (7 857)
 ├── StratifiedGroupKFold(7) → 1 part = calib (1 134)
 └── le reste = development
       ├── StratifiedGroupKFold(6) → 1 part = selection (1 127)
       └── le reste = fit (5 596)
```

| Ensemble | Sert à |
|---|---|
| `fit` | **Entraîner** V2 et V3 |
| `selection` | **Choisir** entre V2 et V3, et entre snapshot « last » et « average » |
| `calib` | **Régler** le post-traitement (seuils, porte…) |

- Stratification sur `has_mask` (même proportion d'images retouchées partout).
- `assert sets[0].isdisjoint(sets[1]) and ...` : vérifie qu'**aucun groupe** n'est partagé. `isdisjoint` = « aucun élément en commun ».

Sortie : `fit/selection/calib 5596 1127 1134 groups 6898`.

### Cellule 8 — Dataset et architecture

**`transforms(size, training)`** avec albumentations :

| Transformation | Ce qu'elle fait |
|---|---|
| `A.Resize(size, size)` | Redimensionne |
| `A.HorizontalFlip(p=0.5)` | Miroir gauche-droite |
| `A.RandomRotate90(p=0.25)` | Rotation de 90°, 180° ou 270° |
| `A.OneOf([...], p=0.35)` | Une seule au hasard parmi : |
| ↳ `A.ImageCompression(quality_range=(75, 98))` | Recompression **JPEG** (les retouches réelles circulent souvent compressées) |
| ↳ `A.GaussianBlur` | Léger flou |
| ↳ `A.NoOp` | Ne rien faire |
| `A.ColorJitter(..., p=0.25)` | Variations de couleur |

**Point clé** : `self.tf(image=image, mask=mask)` applique **la même transformation** à l'image et au masque. Si l'image tourne, le masque tourne aussi. Sinon, le masque ne correspondrait plus.

**`MaskDataset.__getitem__`** renvoie 4 choses : l'image normalisée, le masque, le label image (`has_mask` : 0 ou 1), et le numéro de ligne.

**`make_loader(...)`** — nouveauté : **`WeightedRandomSampler`**

```python
weights = np.where(pos, 0.5 / pos.sum(), 0.5 / (~pos).sum())
```

Chaque image retouchée reçoit un poids, chaque image vide un autre poids, pour que **50 % des images tirées soient retouchées et 50 % vides**. Sans ça, le modèle verrait 2/3 d'images vides et apprendrait surtout à dire « rien ». `replacement=True` : une image peut être tirée plusieurs fois.

**`SRMBranch` — la branche bruit :**

**SRM** = *Spatial Rich Model*, une méthode classique de **forensique d'image** (et de stéganalyse). On utilise des filtres **passe-haut** qui suppriment le contenu et gardent le bruit.

```
Filtre 1 (÷4)         Filtre 2 (÷12)            Filtre 3 (÷4)
 0  0  0  0  0        -1   2  -2   2  -1          0  0  0  0  0
 0 -1  2 -1  0         2  -6   8  -6   2          0  0 -1  0  0
 0  2 -4  2  0        -2   8 -12   8  -2          0 -1  4 -1  0
 0 -1  2 -1  0         2  -6   8  -6   2          0  0 -1  0  0
 0  0  0  0  0        -1   2  -2   2  -1          0  0  0  0  0
```

Chaque filtre compare un pixel à ses voisins. Sur une zone lisse, le résultat ≈ 0. Il ne reste que le **bruit** et les micro-détails.

| Mot-clé | Signification |
|---|---|
| `register_buffer("kernels", ...)` | Les filtres sont stockés dans le modèle mais **pas appris** (fixes) |
| `F.conv2d(..., self.kernels, padding=2)` | Applique les 3 filtres → 3 cartes de bruit |
| `corrected=True` (V3) | 1) annule la normalisation pour retrouver les vraies couleurs ; 2) calcule la **luminance** `0.299 R + 0.587 G + 0.114 B` (la formule standard de l'œil humain) ; 3) bords en mode `reflect` (miroir) au lieu de zéros, pour ne pas créer de faux bords |
| 4 blocs `Conv2d → BatchNorm2d → GELU` (×2) | Petites couches **apprises** qui lisent ces cartes de bruit |
| Canaux 32, 64, 128, 256 | Nombre de cartes à chaque échelle |
| `stride 2` | Divise la taille par 2. Le 1er bloc divise par 4, puis par 2 à chaque bloc → tailles 1/4, 1/8, 1/16, 1/32 de l'image. **Exactement les mêmes échelles que ConvNeXt**, pour pouvoir les fusionner. |
| `BatchNorm2d` | Normalise les valeurs entre les couches (entraînement plus stable) |
| `GELU` | Fonction d'activation (variante douce de ReLU) |

**`ForensicFPN` — le modèle complet :**

| Partie | Ce qu'elle fait |
|---|---|
| `timm.create_model(..., features_only=True, out_indices=(0,1,2,3))` | ConvNeXt renvoie ses **4 cartes intermédiaires** au lieu d'une classe |
| `set_grad_checkpointing(True)` | **Gradient checkpointing** : PyTorch oublie certains calculs et les refait pendant la rétropropagation. Plus lent, mais **beaucoup moins de mémoire**. |
| `lateral` (conv 1×1) | À chaque échelle : colle ConvNeXt + SRM (`torch.cat`) et ramène à 128 canaux |
| Boucle `for i in range(2, -1, -1)` | **Top-down** du FPN : on part de l'échelle la plus grossière, on l'agrandit (`F.interpolate` bilinéaire) et on l'**ajoute** à l'échelle plus fine. L'information globale descend vers le détail. |
| `smooth` (conv 3×3 + GELU) | Lisse chaque niveau après l'addition |
| `fused` | Toutes les échelles agrandies à 1/4 de la taille et collées → 4 × 128 = **512 canaux** |
| `mask_head` | 512 → 128 → **1** canal : le masque (avec `Dropout2d(0.1)`) |
| `boundary_head` | 512 → 64 → **1** : les contours |
| `classifier` | Moyenne globale (`AdaptiveAvgPool2d(1)`) de l'échelle la plus grossière → `LayerNorm` → `Dropout(0.25)` → `Linear(128, 1)` : **l'image est-elle retouchée ?** |
| `rgb_classifier` (V3) | Même idée, mais sur la dernière carte de ConvNeXt seul. V3 fait la **moyenne** des deux classifieurs. |
| `aux_heads` (V3) | 3 petites têtes qui prédisent aussi le masque sur les niveaux 0, 1, 2. C'est la **supervision profonde** : chaque niveau apprend directement. |
| `refine` (V3) | Reprend `fused` + masque de base + contour (512 + 1 + 1 = **514** canaux) pour un masque final plus précis |
| Sortie | Tout est ré-agrandi à la taille de l'image d'entrée |

### Cellule 10 — Losses, EMA, snapshots et TTA

**`boundary_target(mask)`** :
```python
F.max_pool2d(mask, 3, 1, 1) + F.max_pool2d(-mask, 3, 1, 1)
```
- `max_pool2d(mask)` = **dilatation** (la tache grossit d'un pixel) ;
- `max_pool2d(-mask)` = moins l'**érosion** (la tache rétrécit d'un pixel) ;
- dilatation − érosion = **un anneau autour du bord**. C'est la cible de la tête contour.

**Les losses :**

| Fonction | Explication |
|---|---|
| `focal_bce(logits, target, gamma=2)` | BCE multipliée par `(1 − pt)²`, où `pt` est la probabilité de la bonne réponse. Les pixels **faciles** (pt proche de 1) comptent presque zéro, les **difficiles** comptent beaucoup. Utile car la plupart des pixels sont faciles (fond). |
| `soft_iou(logits, target, positive)` | IoU calculé avec des **probabilités** au lieu de 0/1 → dérivable, donc utilisable comme loss. `1 − (inter + 1) / (union + 1)` : le `+1` évite la division par 0. Seulement sur les images retouchées. |
| `lovasz_hinge_flat(logits, labels)` | **Lovász hinge** : une autre approximation mathématique de l'IoU, plus fidèle. Trie les erreurs et les pondère selon leur effet sur l'IoU. |
| `balanced_boundary(logits, target)` | BCE avec `pos_weight` (les pixels de contour sont rares, on leur donne jusqu'à 25× plus de poids) + **Dice loss** (proche de l'IoU) |
| `combined_loss(outputs, mask, label, enhanced)` | La somme pondérée de toutes les losses (voir le tableau du README) |

*Bug corrigé dans Lovász* : `grad[1:] -= grad[:-1]` lit et écrit dans des morceaux qui se chevauchent du même tenseur. PyTorch refuse. `.clone()` copie d'abord la partie lue.

Pour V3, Lovász est calculé à **1/4 de résolution** (`scale_factor=0.25`) parce que le tri de tous les pixels en 640 × 640 coûte cher.

**EMA et snapshots :**

| Fonction | Rôle |
|---|---|
| `update_ema(ema, model, decay=0.995)` | `ema = 0.995 × ema + 0.005 × poids actuels`. Une **moyenne glissante** des poids, plus stable que les poids bruts. Appelée après chaque mise à jour. |
| `cpu_state(model)` | Copie les poids sur le CPU (libère la mémoire GPU) |
| `average_states(states)` | Moyenne de plusieurs snapshots de poids |

**TTA (Test-Time Augmentation) :**

| Fonction | Rôle |
|---|---|
| `transform_image(x, index)` | Applique une des 8 transformations **dièdres** : 4 rotations (0°, 90°, 180°, 270°) × avec/sans miroir. `index // 2` = rotation, `index % 2` = miroir. |
| `invert_map(x, index)` | **Annule** la transformation sur la carte prédite, pour la remettre dans le bon sens |
| `predict(model, loader, views)` | Pour chaque vue : transformer → prédire → dé-transformer → additionner. Puis moyenne. Avec `views=2` : image normale + miroir. Cartes redimensionnées en 640 × 640 et stockées en `float16` (moins de mémoire). |

`@torch.no_grad()` : décorateur qui désactive le calcul des gradients dans toute la fonction.

### Cellule 12 — Entraînement comparatif V2 / V3

**`safe_batch(enhanced, size)`** : essaie un faux entraînement avec 4 images, puis 2, puis 1. Renvoie la plus grande taille qui ne provoque pas `torch.cuda.OutOfMemoryError`. Le notebook s'adapte tout seul à la mémoire disponible.

**`train_development(name, enhanced, size)`** :

| Élément | Détail |
|---|---|
| Accumulation | `16 // batch` → batch effectif de **16** |
| `ema = copy.deepcopy(model)` | Copie indépendante du modèle pour l'EMA, gelée (`requires_grad = False`) |
| 2 groupes de LR | Encodeur pré-entraîné : **3e-5**. Nouvelles couches (SRM, FPN, têtes) : **1.2e-4**. |
| `id(x)` | Identifiant unique d'un objet Python. Sert à séparer les paramètres de l'encodeur des autres. |
| `CosineAnnealingLR` | LR qui descend en cosinus (pas de warmup ici) |
| Snapshots | EMA sauvegardée aux 3 dernières epochs (4, 5, 6) |
| Renvoie | `last` (dernier snapshot) et `average` (moyenne des 3) |

Ensuite : `BASELINE = train_development("v2_baseline", False, 512)` puis, si le temps le permet, `V3 = train_development("v3_enhanced", True, 640)`.

*Note* : la loss de V3 (0.37) est plus haute que celle de V2 (0.18). **Ce n'est pas comparable** : V3 additionne plus de termes de loss. Seul l'IoU compte pour comparer.

### Cellule 14 — Retrieval et calibration

**Le retrieval (recherche de l'original) :**

| Fonction | Rôle |
|---|---|
| `build_hash_index(frame, hashes)` | Range dans des seaux uniquement les images **sans retouche** (galerie de références) |
| `retrieval_probability(path, target_hash, gallery, buckets)` | Cherche l'original d'une image et calcule ce qui a changé |

Étapes de `retrieval_probability` :
1. Trouve les candidats qui partagent un morceau de pHash, garde les **4 plus proches**.
2. Si le plus proche a plus de **10 bits** de différence → pas d'original trouvé.
3. **Alignement** : `cv2.findTransformECC` avec `MOTION_AFFINE` trouve le décalage / rotation / zoom entre les deux images (calcul en 256 × 256 gris, puis remis à l'échelle). `cv2.warpAffine` applique cet alignement.
4. **Différence de couleur** dans l'espace **LAB** (proche de la perception humaine), ÷ 45.
5. **Différence de bords** avec le **Laplacien** (`cv2.Laplacian`, un filtre qui détecte les bords), ÷ 0.35.
6. Mélange `0.75 × couleur + 0.25 × bords`, puis flou gaussien → carte de changement.
7. **Confiance** = corrélation de l'alignement × `(1 − distance / 12)`. Gardée seulement si ≥ 0.35.

Résultat : un original n'est trouvé que pour **3,8 %** des images. Le gain était trop faible, donc le retrieval a reçu un **poids de 0** dans le réglage final.

**Les fonctions de post-traitement :**

| Fonction | Rôle |
|---|---|
| `remove_small(mask, area)` | `cv2.connectedComponentsWithStats` trouve chaque tache séparée et sa surface. Supprime celles plus petites que `area` pixels. |
| `top_evidence(probability)` | Moyenne des **0,1 % de pixels les plus suspects**. `np.partition` trouve ces valeurs sans tout trier (plus rapide). |
| `native_iou(frame, index, mask, epsilon)` | Masque → **polygone** → masque à la taille d'origine → IoU officiel. `TRUTH_CACHE` garde les vrais masques en mémoire pour ne pas les recalculer. |
| `evidence_values(maps, classes, alpha)` | Preuve par image = `alpha × proba_image + (1 − alpha) × top_evidence` |
| `evaluate_config(...)` | Applique une configuration complète (porte, seuil, nettoyage, fermeture) à toutes les images et renvoie l'IoU moyen |
| `calibrate(...)` | Teste **384 configurations**, garde la meilleure, vérifie si le retrieval apporte vraiment quelque chose, puis teste 3 valeurs d'`epsilon` |

Dans `evaluate_config` :
- mode `rank` : `np.partition(evidence, -n/3)` trouve la valeur du **tiers supérieur**. Seul le tiers des images les plus suspectes peut recevoir un masque.
- `cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ...)` : **fermeture morphologique** = dilatation puis érosion. Bouche les petits trous dans le masque.

### Cellule 16 — Sélection automatique et validation officielle

**`assess_candidate(candidate)`** :
1. Pour les snapshots `last` et `average` : prédit `selection` (2 vues), calcule l'IoU et le **taux de faux positifs** (FPR = images vides où j'ai dessiné quelque chose).
2. Garde le meilleur snapshot.
3. Prédit `calib` et lance `calibrate`.

**La règle de choix V2 / V3** (écrite avant de voir les résultats) :
```python
v3_allowed = (V3 score >= V2 score + 0.002) and (V3 FPR <= V2 FPR)
```
→ V3 : 0.9392 contre 0.9307, FPR 0 % contre 0.14 % → **V3 retenu**, snapshot `last`.

**Test du TTA 8 vues** : seulement s'il reste du temps. Gardé seulement s'il gagne au moins +0.001. Pas assez de gain → **2 vues**.

**Validation officielle** (`val/`, ouverte une seule fois) :
- retrieval contre tout le train ;
- prédiction avec le modèle et la configuration figés ;
- IoU moyen **0.9385**, IoU sur les images retouchées **0.8217**, porte `[[650, 2], [2, 324]]`.

### Cellule 18 — Réentraînement final, soumission et audit

1. **Garde** : si on a dépassé 4,25 h → arrêt (pas assez de temps pour finir proprement).
2. Réentraîne V3 sur **train + val** (8 835 images) avec EMA, 8 epochs prévues.
3. **Garde par epoch** : après l'epoch 4, si on dépasse 5,5 h → on arrête. C'est arrivé après **5 epochs** (`"Refit arrêté par garde temporelle"` dans les warnings).
4. Snapshot `last` → c'est l'EMA finale qui est utilisée.
5. Sauvegarde des poids (`level3_v3_self_select_s20260728_final.pt`).
6. Prédiction du test (2 vues) + retrieval contre train + val.
7. Application de la configuration figée : porte (rank) → seuil pixel 0.45 → polygone.
8. **Contrôles** : 993 lignes, pas de doublon, mêmes `image_id` que le sample.
9. `submission.csv` : **328 images** avec polygone, 665 avec `" "`.
10. Sauvegardes : `candidate_comparison.csv`, cartes de probabilités en `uint8` (0-255, plus léger) dans un **zip**, audit `audit_level3_v3.json` (scores, configuration, durée de chaque phase, mémoire GPU max, warnings).

---

## Partie D — Dictionnaire des mots-clés propres au Level 3

### Segmentation

| Mot | Explication |
|---|---|
| **Segmentation** | Donner une réponse pour **chaque pixel**. |
| **Masque** | Image de 0 et 1 : 1 = pixel retouché. |
| **Polygone** | Forme décrite par une liste de points (x, y). |
| **Coordonnées normalisées** | x ÷ largeur, y ÷ hauteur → entre 0 et 1. |
| **Contour** | Bord d'une forme. |
| **IoU** | Intersection ÷ union de deux masques. |
| **Encodeur** | Partie du réseau qui extrait les caractéristiques de l'image. |
| **Carte de caractéristiques** | Sortie intermédiaire d'un réseau : une « image » de détecteurs. |
| **Échelle 1/4, 1/8…** | Taille de la carte par rapport à l'image. |
| **FPN** | Réseau qui combine les cartes de toutes les échelles (du global vers le fin). |
| **Top-down** | Du grossier vers le fin. |
| **Supervision profonde** | Ajouter des sorties intermédiaires qui apprennent aussi. |
| **Tête de raffinement** | Couche qui améliore un premier masque. |

### Forensique d'image

| Mot | Explication |
|---|---|
| **Forensique** | Analyse technique pour trouver des preuves de manipulation. |
| **SRM** | Filtres fixes qui extraient le bruit d'une image. |
| **Passe-haut** | Filtre qui garde les variations rapides (détails, bruit). |
| **Luminance** | Luminosité perçue : `0.299 R + 0.587 G + 0.114 B`. |
| **Retrieval** | Retrouver une image similaire dans une base. |
| **ECC** | *Enhanced Correlation Coefficient* : méthode d'alignement de deux images. |
| **Transformation affine** | Déplacement + rotation + zoom + étirement. |
| **LAB** | Espace de couleurs proche de la perception humaine. |
| **Laplacien** | Filtre qui détecte les bords. |

### Morphologie (opérations sur les masques)

| Mot | Explication |
|---|---|
| **Dilatation** | La forme grossit. |
| **Érosion** | La forme rétrécit. |
| **Fermeture** | Dilatation puis érosion : bouche les petits trous. |
| **Composante connexe** | Une tache séparée des autres. |

### Entraînement et inférence

| Mot | Explication |
|---|---|
| **Focal loss** | Loss qui insiste sur les exemples difficiles. |
| **Soft IoU / Dice** | Versions dérivables de l'IoU, utilisables comme loss. |
| **Lovász hinge** | Approximation mathématique de l'IoU pour l'optimiser. |
| **`pos_weight`** | Poids supplémentaire pour la classe rare. |
| **Échantillonnage pondéré** | Tirer plus souvent certaines images pour équilibrer. |
| **EMA** | Moyenne glissante des poids du modèle. |
| **Snapshot** | Copie des poids à un moment donné. |
| **Gradient checkpointing** | Recalculer au lieu de stocker → moins de mémoire. |
| **TTA** | Prédire sur plusieurs versions (rotation, miroir) et faire la moyenne. |
| **Dièdre** | Les 8 combinaisons rotation × miroir d'un carré. |
| **Gate (porte)** | Décider si l'image est retouchée avant de dessiner. |
| **Calibration** | Réglage des seuils de post-traitement sur des données à part. |
| **FPR** | Taux de faux positifs : images vides où j'ai dessiné un masque. |
| **Garde temporelle** | Le code vérifie l'heure et s'arrête proprement avant la limite. |
| **SHA-256** | Empreinte unique d'un fichier. Un seul caractère change → empreinte totalement différente. |
| **Test unitaire** | Petit test automatique qui vérifie une fonction. |

---

## Partie E — Points honnêtes à connaître

1. **Écart validation / leaderboard** : la V2 faisait ≈ 0.93 en interne mais **0.87** en public. Le test public est peut-être plus difficile, ou il contient peu d'images. Je ne peux pas l'expliquer avec certitude.
2. **Refit coupé à 5 epochs sur 8** par la garde temporelle. Choix volontaire : mieux vaut un modèle un peu moins entraîné qu'un notebook qui ne finit pas.
3. **Le retrieval est calculé mais pas utilisé** (poids 0). Il prend du temps pour rien sur ce run. La règle de sélection l'a écarté honnêtement.
4. **La porte en mode `rank` suppose qu'1/3 du test est retouché.** Si le test a une autre proportion, c'est un risque.
5. **Le dessin reste le point faible** : IoU de 0.82 sur les images retouchées, alors que la porte est presque parfaite.
6. **Le sujet mentionne le format RLE**, mais la soumission demandée est en **polygones**. J'ai suivi le `sample_submission.csv`.

---

## Partie F — Fiche ultra-courte

- **Tâche** : dessiner la zone retouchée en polygone, `" "` si pas de retouche. Métrique : **IoU moyen** (vide + vide = 1).
- **2/3 des images sont vides** → une **porte** décide d'abord, puis on dessine.
- **Codec** polygone ↔ masque testé automatiquement. Scores mesurés **après l'aller-retour polygone**.
- **Anti-fuite** : pHash en 8 morceaux, 3 sous-ensembles disjoints : **fit / selection / calib**.
- **ForensicFPN** = ConvNeXt-V2 (contenu) + **SRM** (bruit) + **FPN** (échelles) → masque + contour + classe.
- **Losses** : focal BCE + soft IoU + BCE classe + contour (+ Lovász + auxiliaires en V3).
- **Entraînement** : échantillonnage 50/50, albumentations, batch auto, gradient checkpointing, EMA, snapshots.
- **Règle fixée à l'avance** : V3 retenu (0.9392 vs 0.9307, 0 fausse alerte).
- **Calibration** : 384 configurations → porte `rank`, alpha 0.85, seuil pixel 0.45, pas de retrieval, 2 vues TTA.
- **`val/` une seule fois** : IoU **0.9385**, porte avec 4 erreurs sur 978.
- **Final** : refit train + val (5 epochs, garde temporelle), 328 polygones sur 993, audit JSON.
