# Level 2 — Classification : vraie, générée ou retouchée ?

Notebook : [level2.ipynb](level2.ipynb)

---

## 1. Le problème en une phrase

Pour chaque image de `test/`, je dois choisir **une classe parmi trois** :

| Classe | Nom | Signification |
|---|---|---|
| **0** | `real` | Vraie photo, non modifiée |
| **1** | `generated` | Image **entièrement** créée par IA |
| **2** | `local_edit` | Vraie photo dont **une zone** a été modifiée (visage remplacé, objet effacé…) |

C'est une **classification multi-classe**. Au Level 1, les classes 1 et 2 étaient fusionnées. Ici, je dois les séparer.

**Pourquoi c'est plus dur ?** Une image `local_edit` est à 90 % une vraie photo. Seule une petite zone est fausse. Le modèle doit donc trouver un petit détail.

---

## 2. Les données

| Dossier | Images | Répartition |
|---|---|---|
| `train/` | 7 857 | 2 619 par classe (parfaitement équilibré) |
| `val/` | 978 | 326 par classe |
| `test/` | 993 | inconnue |

---

## 3. La métrique : le Macro F1

1. Pour **chaque classe**, je calcule un F1 en mode « un contre tous » (cette classe = positive, les deux autres = négatives).
2. Je fais la **moyenne simple** des 3 F1 :

```
Macro F1 = ( F1(real) + F1(generated) + F1(local_edit) ) / 3
```

**Pourquoi « Macro » ?** Chaque classe pèse **autant**, peu importe son nombre d'images. Si mon modèle ignore une classe, son F1 tombe à 0 et le score global chute d'un tiers. La classe `local_edit` est la plus dangereuse pour les fake news : on ne peut pas la négliger.

---

## 4. Mon approche en un schéma

C'est **la même recette que le Level 1**, adaptée à 3 classes.

```
          Images du train (7 857)
                   │
   ① Groupes de quasi-doublons (pHash + Union-Find)
                   │
   ② 3 folds sans fuite (StratifiedGroupKFold) + vérification
                   │
   ③ 3 experts, chacun sort 3 probabilités
        ├── DeepGuard / SigLIP2  (tête modifiée 2 → 3 classes)
        ├── ConvNeXt-V2 RGB
        └── ConvNeXt-V2 NPR      (5 vues)
                   │
   ④ Prédictions OOF
                   │
   ⑤ Stacking (régression logistique) + biais par classe, appris sur l'OOF
                   │
   ⑥ Vérification UNE SEULE FOIS sur val/
                   │
   ⑦ Réentraînement train + val → test → submission.csv
```

---

## 5. Les étapes en détail

### Étape ① — Les quasi-doublons (pHash)

Même idée qu'au Level 1 : des images presque identiques doivent rester dans le même fold, sinon c'est une **fuite de données**.

- pHash de 64 bits par image.
- Deux images sont « doublons » si leur distance de Hamming est **≤ 4 bits**.
- **Amélioration par rapport au Level 1** : je coupe l'empreinte en **5 morceaux** (au lieu de 4).
  Pourquoi ? Avec 4 bits différents et 5 morceaux, au moins un morceau est forcément identique (principe des tiroirs). Donc je ne rate aucune paire.
- Union-Find pour former les groupes.

### Étape ② — Les folds

- `StratifiedGroupKFold` à 3 folds : même proportion de classes dans chaque fold, et groupes jamais coupés.
- J'ajoute un **contrôle automatique** : si un même groupe apparaît dans le train et la validation d'un fold, le notebook s'arrête avec une erreur (`Fuite de groupes`).
- Taille des folds : ~5 200 images pour entraîner, ~2 600 pour valider.

### Étape ③ — Les 3 experts

| Expert | Modèle | Entrée | Rôle |
|---|---|---|---|
| `deep_rgb` | DeepGuard (`king1oo1/deepfake-model`, SigLIP2) | 224 px | Vue globale réel/faux |
| `cnn_rgb` | ConvNeXt-V2 Tiny | 384 px | Texture et contenu |
| `cnn_npr` | ConvNeXt-V2 Tiny | 384 px, NPR, 5 vues | Traces entre pixels voisins |

**Astuce pour DeepGuard (2 → 3 classes)**
DeepGuard sait dire « real » ou « fake ». Il me faut 3 sorties. Je crée une nouvelle tête à 3 sorties et je **réutilise ses anciens poids** :

- sortie 0 (`real`) ← copie des poids « real » ;
- sorties 1 (`generated`) et 2 (`local_edit`) ← copie des poids « fake » ;
- j'ajoute un **tout petit bruit** à la sortie 2. Sinon les sorties 1 et 2 seraient identiques et apprendraient exactement la même chose.

Le modèle démarre donc avec ce qu'il sait déjà, au lieu de partir de zéro.

**NPR** (rappel) : `image − version réduite puis ré-agrandie`, puis `tanh(2.5 × …)`. Ça garde seulement les micro-détails où l'IA laisse des traces.
**5 vues** : image entière + 4 coins. Je combine `0.75 × moyenne + 0.25 × max`, puis je renormalise pour que les 3 probabilités fassent 1.

### Les réglages d'entraînement

- **Loss** : `cross_entropy` (la loss standard multi-classe) avec **label smoothing 0.02**.
  Le label smoothing dit au modèle « sois sûr à 98 %, pas à 100 % ». Ça évite qu'il devienne trop confiant.
- **Sortie** : `softmax` → 3 probabilités qui font 1.
- AdamW avec 2 learning rates (corps petit, tête 2e-4), cosine + warmup, mixed precision, gradient accumulation, clipping, early stopping (patience 2), 4 epochs max.
- Le checkpoint retenu est celui qui a le **meilleur Macro F1** sur le fold de validation.

**Le bug GPU que j'ai corrigé**
Sur Kaggle (2 GPU T4), le kernel plantait (`c10::AcceleratorError`) dès qu'un 2ᵉ modèle utilisait le 2ᵉ GPU. La cause : la communication entre les deux GPU (P2P) est instable avec `nn.DataParallel`.
Solution : `USE_TWO_GPUS = False`. J'ai ajusté batch et accumulation pour garder **exactement le même batch effectif** (12 × 4 = 48 et 4 × 6 = 24). La recette d'entraînement ne change donc pas.
J'ai aussi ajouté `torch.cuda.synchronize()` pour qu'une erreur GPU apparaisse tout de suite, au lieu de tuer le kernel plus tard.

### Étape ④ — Prédictions OOF

Chaque image du train reçoit 3 probabilités par expert, prédites par un modèle qui ne l'a jamais vue.
Tableau final : `7 857 images × 3 experts × 3 classes`.
Je vérifie que les probabilités sont finies et que chaque ligne fait bien 1.

### Étape ⑤ — Combinaison et décision

**a) Le stacking**
J'ai 3 experts × 3 probabilités = **9 nombres** par image. Je prends leur log, et une **régression logistique** apprend à les combiner pour donner 3 nouvelles probabilités.
Pour évaluer ce stacking honnêtement, je le teste lui aussi en 5 folds.

**b) Les biais par classe**
Par défaut, je prends la classe qui a la plus grande probabilité (`argmax`). Mais parfois il vaut mieux **pousser un peu** une classe difficile.
J'ajoute un petit **biais** au log-probabilité de chaque classe, puis je fais l'`argmax`. Les biais sont cherchés par une recherche en grille, du plus grossier au plus fin, pour maximiser le Macro F1.
C'est l'équivalent du **seuil** du Level 1, mais pour 3 classes.

Pour rester honnête : j'apprends les biais sur 4 sous-parties, je les teste sur la 5ᵉ, et je garde la **médiane** des 5 biais.

**c) Le choix automatique**

| Candidat | Macro F1 OOF |
|---|---|
| deep_rgb | 0.8934 |
| cnn_rgb | 0.9930 |
| cnn_npr | 0.9950 |
| mean (moyenne des 3) | 0.9969 |
| **stack** ✅ | **0.99707** |

→ Gagnant : le **stack**, avec les biais `[-0.19, -0.19, +0.38]`.
Lecture : le biais pousse vers la classe `local_edit`. C'est logique : c'est la classe la plus difficile, et le modèle avait tendance à la sous-prédire.

Ici, contrairement au Level 1, **la combinaison des experts bat chaque expert seul**.

### Étape ⑥ — Le test honnête sur `val/`

Ouvert une seule fois, après avoir tout figé.

- **Macro F1 holdout = 0.99591**
- F1 par classe : real 0.9969, generated 0.9954, local_edit 0.9954

```
                 prédit real   prédit generated   prédit local_edit
real                 324              1                  1
generated              0            325                  1
local_edit             0              1                325
```

→ **4 erreurs sur 978 images**. L'écart OOF / holdout est très faible (0.997 vs 0.996). Le notebook affiche une alerte si l'écart dépasse 1,5 point : ce n'est pas le cas.

### Étape ⑦ — Final

- Réentraînement des 3 experts sur `train + val`, nombre d'epochs = médiane des meilleurs epochs OOF.
- Le stacker, le candidat et les biais restent **figés** (appris avant l'audit).
- `submission.csv` (colonnes `image_id`, `classe`) construit par une **jointure** sur `sample_submission.csv` (je ne suppose pas que l'ordre des images est le même).
- Contrôles : bon nombre de lignes, pas de doublon, pas de valeur manquante, classes dans {0, 1, 2}.
- Audit JSON `audit_level2_v2.json` : scores, biais, epochs, versions, durée.
- **Caches** `.npz` : si Kaggle coupe le run, je peux reprendre sans tout réentraîner.

---

## 6. Résumé des résultats

| Mesure | Valeur |
|---|---|
| Modèle retenu | Stacking (régression logistique) des 3 experts + biais |
| Macro F1 OOF | 0.99707 |
| **Macro F1 sur val (jamais vu)** | **0.99591** |
| Erreurs sur val | 4 sur 978 |

---

## 7. Différences avec le Level 1 (à bien retenir)

| | Level 1 | Level 2 |
|---|---|---|
| Nombre de classes | 2 | 3 |
| Loss | BCE | Cross-entropy + label smoothing |
| Sortie | sigmoid (1 proba) | softmax (3 probas) |
| Décision | seuil | argmax + biais par classe |
| Métrique | F1 | Macro F1 |
| Modèle gagnant | NPR seul | Stacking des 3 |
| pHash | 4 morceaux | 5 morceaux (aucune paire ratée) |

---

## 8. Petit dictionnaire

| Mot | Explication simple |
|---|---|
| **Softmax** | Transforme des scores en probabilités qui font 1 au total. |
| **Argmax** | Prendre la classe qui a la plus grande valeur. |
| **Label smoothing** | Au lieu de viser 100 %, on vise 98 % pour la bonne classe. Le modèle est moins arrogant. |
| **Biais de décision** | Petit bonus ou malus ajouté à une classe avant de choisir. |
| **Stacking** | Un modèle qui apprend à combiner les sorties d'autres modèles. |
| **Régression logistique** | Modèle simple et rapide. Ici, `C=0.08` = forte régularisation pour éviter l'overfitting. |
| **Matrice de confusion** | Tableau « vraie classe × classe prédite ». La diagonale = les bonnes réponses. |

---

## 9. Questions probables à l'entretien

**Q : Pourquoi le Macro F1 et pas l'accuracy ?**
R : Avec le Macro, chaque classe compte pareil. Un modèle qui raterait la classe `local_edit` serait fortement puni. C'est voulu, car c'est le cas le plus dangereux pour la désinformation.

**Q : Quelle est la classe la plus difficile et pourquoi ?**
R : `local_edit`, car l'image est presque entièrement réelle. Seule une petite zone est fausse. C'est pour ça que j'utilise les 4 coins en NPR (le `max` attrape une zone suspecte locale), et que le biais final favorise cette classe.

**Q : Comment tu as adapté DeepGuard qui était binaire ?**
R : J'ai remplacé sa tête par une tête à 3 sorties. J'ai copié les poids « real » vers la classe 0 et les poids « fake » vers les classes 1 et 2, avec un petit bruit sur la 2 pour casser la symétrie.

**Q : Pourquoi des biais et pas juste l'argmax ?**
R : L'argmax suppose que les probabilités sont parfaitement calibrées. En pratique, un petit décalage par classe améliore le Macro F1. Je l'ai appris sur l'OOF, avec la médiane sur 5 folds, pour qu'il soit stable.

**Q : Pourquoi tu n'utilises qu'un GPU alors que tu en as deux ?**
R : Le lien entre les deux T4 de Kaggle était instable avec `DataParallel` et faisait planter le kernel. J'ai désactivé le multi-GPU et adapté l'accumulation de gradients pour garder le même batch effectif. Le résultat est identique, juste plus lent.

**Q : Pourquoi DeepGuard est si faible (0.89) ?**
R : Il a été pré-entraîné pour réel/faux en binaire, sur des images globales à 224 px. Il distingue mal une image générée d'une retouche locale. Mais il apporte un avis différent, utile dans le stacking.

**Q : Que ferais-tu pour améliorer ?**
R : Analyser les 4 erreurs de `val/` à la main, tester plusieurs seeds, ou ajouter un expert spécialisé dans la localisation (comme au Level 3) pour mieux repérer les retouches locales.

---

## 10. Mon pitch en 30 secondes

> « Le Level 2 demandait de distinguer trois classes : photo réelle, image générée et retouche locale, avec le Macro F1.
> J'ai repris ma recette du Level 1 : groupes anti-doublons par pHash, folds sans fuite, et trois experts — DeepGuard que j'ai adapté de 2 à 3 classes en réutilisant ses poids, un ConvNeXt-V2 en RGB et un en NPR.
> Les prédictions out-of-fold servent à entraîner un stacking par régression logistique et à apprendre un petit biais par classe, qui favorise `local_edit`, la classe la plus difficile.
> Sur la validation, ouverte une seule fois, j'ai un Macro F1 de 0,996, soit 4 erreurs sur 978 images. »
