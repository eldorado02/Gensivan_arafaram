# Level 1 — Détection : vraie photo ou image truquée ?

Notebook : [level1.ipynb](level1.ipynb)

---

## 1. Le problème en une phrase

Pour chaque image du dossier `test/`, je dois répondre **0** (photo authentique) ou **1** (image manipulée).
« Manipulée » regroupe deux cas : l'image a été **entièrement générée par IA** (Cat 1) ou bien c'est une vraie photo **retouchée localement** (Cat 2).

C'est une **classification binaire** (deux réponses possibles).

---

## 2. Les données

| Dossier | Nombre d'images | Labels ? |
|---|---|---|
| `train/` | 7 857 | oui (`ground_truth.csv`) |
| `val/` | 978 | oui |
| `test/` | 993 | non (c'est ce qu'on doit prédire) |

- Dans le train : **1/3 d'images authentiques, 2/3 d'images manipulées** (car Cat 1 + Cat 2 sont fusionnées).
- Dans le val : 326 authentiques, 652 manipulées.

---

## 3. La métrique : le F1 score

On compte :

- **TP** (vrai positif) : image truquée, et j'ai dit « truquée » ✅
- **FP** (faux positif) : image vraie, mais j'ai dit « truquée » ❌ (fausse alerte)
- **FN** (faux négatif) : image truquée, mais j'ai dit « vraie » ❌ (truquage raté)

```
Précision = TP / (TP + FP)   → « quand je dis truqué, ai-je raison ? »
Rappel    = TP / (TP + FN)   → « est-ce que je trouve tous les truquages ? »
F1        = 2 · Précision · Rappel / (Précision + Rappel)
```

Le F1 est un compromis entre précision et rappel. 1.0 = parfait.

**Le seuil** : mon modèle donne une probabilité (ex. 0.73). Je dois la transformer en 0 ou 1 avec un seuil.
Si `proba >= seuil` → 1, sinon 0. Le choix du seuil m'appartient, et je l'ai choisi pour maximiser le F1.

---

## 4. Mon approche en un schéma

```
          Images du train (7 857)
                   │
   ① Regrouper les quasi-doublons (pHash)
                   │
   ② Découper en 3 folds sans fuite (StratifiedGroupKFold)
                   │
   ③ Entraîner 3 « experts » sur chaque fold
        ├── Expert A : DeepGuard / SigLIP2   (sens global de l'image)
        ├── Expert B : ConvNeXt-V2 en RGB     (texture, contenu)
        └── Expert C : ConvNeXt-V2 en NPR     (traces entre pixels voisins)
                   │
   ④ Prédictions OOF (chaque image prédite par un modèle qui ne l'a jamais vue)
                   │
   ⑤ Choisir le meilleur candidat + le seuil, UNIQUEMENT sur l'OOF
                   │
   ⑥ Vérifier UNE SEULE FOIS sur val/ (holdout intact)
                   │
   ⑦ Réentraîner sur train + val → prédire test → submission.csv
```

**L'idée clé à retenir** : toutes les décisions (quel modèle, quel seuil, combien d'epochs) sont prises sur le **train** grâce aux prédictions **out-of-fold**. Le dossier `val/` reste « sous scellé » jusqu'à la fin, pour avoir un score honnête.

---

## 5. Les étapes en détail

### Étape ① — Les quasi-doublons avec pHash

**Problème** : si deux images presque identiques se retrouvent l'une dans l'entraînement et l'autre dans la validation, le modèle « triche ». Il reconnaît l'image au lieu d'apprendre à détecter le truquage. On appelle ça une **fuite de données** (*data leakage*). Le score devient trop optimiste.

**Solution** :

1. Je calcule un **pHash** (perceptual hash) pour chaque image. C'est une empreinte de 64 bits. Deux images qui se ressemblent ont des empreintes presque identiques.
2. Je compare les empreintes avec la **distance de Hamming** (nombre de bits différents). Si la distance est **≤ 4**, je considère que ce sont des quasi-doublons.
3. Je fusionne les doublons dans un même **groupe** avec l'algorithme **Union-Find**.
4. Pour aller vite, je ne compare pas toutes les paires : je range les empreintes dans des « seaux » (*buckets*) par morceaux et je compare seulement les images d'un même seau.

Résultat : **6 933 groupes** pour 7 857 images. Le plus grand groupe contient 6 images.

### Étape ② — Les folds (StratifiedGroupKFold)

Je découpe le train en **3 parties (folds)** :

- **Stratified** : chaque fold garde la même proportion 1/3 vraies – 2/3 truquées.
- **Group** : un groupe de doublons reste entièrement dans le même fold. Pas de fuite possible.

### Étape ③ — Les 3 experts

J'ai choisi 3 modèles qui regardent l'image **de manière différente**. C'est ça qui les rend complémentaires.

| Expert | Modèle | Taille d'entrée | Ce qu'il regarde |
|---|---|---|---|
| A — `deep_rgb` | `king1oo1/deepfake-model` (DeepGuard, basé sur SigLIP2) | 224 px | Le sens global de l'image. Il est déjà pré-entraîné pour réel/faux. |
| B — `cnn_rgb` | ConvNeXt-V2 Tiny (pré-entraîné ImageNet-22k) | 384 px | Les textures et le contenu, en couleurs normales. |
| C — `cnn_npr` | Même ConvNeXt-V2, mais entrée NPR | 384 px | Les relations entre pixels voisins. |

**C'est quoi NPR ?** (*Neighboring Pixel Relationships*)
Les générateurs d'images IA agrandissent l'image par étapes (*upsampling*). Ça laisse des traces invisibles à l'œil entre pixels voisins.
Pour les faire ressortir, je calcule :

```
NPR = image − (image réduite 2×2 puis ré-agrandie)
x   = tanh(2.5 × NPR)
```

On enlève le « contenu » de l'image et on garde seulement les micro-détails (haute fréquence). Le `tanh` écrase les valeurs extrêmes.

**Les 5 vues pour NPR** : à la prédiction, l'expert C regarde l'image entière + 4 coins (crops de 72 %). Je combine :
`0.75 × moyenne + 0.25 × maximum`.
Le maximum aide à attraper une **retouche locale** cachée dans un seul coin.

**Fine-tuning de DeepGuard** : je gèle (*freeze*) presque tout le modèle. J'entraîne seulement la tête de classification et les **6 derniers blocs**. Ça évite de casser ce qu'il sait déjà et ça va plus vite.

### Les réglages d'entraînement (à savoir citer)

- **Loss** : `binary_cross_entropy_with_logits` (BCE), la loss classique pour du binaire.
- **Optimiseur** : AdamW avec **2 learning rates** : petit pour le corps du modèle (8e-6 ou 2e-5), plus grand pour la tête (2e-4). Le corps est déjà bon, la tête est nouvelle.
- **Scheduler** : cosine avec warmup (8 % des steps). Le LR monte doucement puis descend en courbe.
- **Mixed precision (AMP, float16)** : plus rapide et moins de mémoire GPU.
- **Gradient accumulation** : le GPU ne peut prendre que 4 images à la fois. J'accumule 6 mini-batchs → batch effectif de 24.
- **Gradient clipping** à 1.0 : évite les gradients qui explosent.
- **Early stopping** : patience de 2 epochs. Si le F1 ne monte plus, j'arrête et je garde le meilleur checkpoint.
- **Augmentations** : crop aléatoire, flip horizontal, variations de couleur, léger flou. Le modèle devient plus robuste.
- **Seed fixe** (`20260723`) et mode déterministe : le notebook est **reproductible**.

### Étape ④ — Les prédictions OOF (Out-Of-Fold)

Pour chaque fold :

- j'entraîne sur 2 folds ;
- je prédis le 3ᵉ fold (images jamais vues par ce modèle) ;
- je prédis aussi `val/`.

À la fin, **chaque image du train a une prédiction « honnête »**. Pour `val/`, je fais la moyenne des 3 modèles.

### Étape ⑤ — Choix du modèle et du seuil

J'ai 4 candidats :

1. **stack** : une régression logistique qui combine les 3 experts (c'est le *stacking*) ;
2. `deep_rgb` seul ;
3. `cnn_rgb` seul ;
4. `cnn_npr` seul.

Pour chacun, je cherche le meilleur seuil sur 5 sous-parties, puis je prends la **médiane** des 5 seuils. La médiane est plus stable qu'un seul seuil « parfait » qui pourrait être de la chance.

| Candidat | F1 OOF |
|---|---|
| deep_rgb | 0.9230 |
| cnn_rgb | 0.9953 |
| stack | 0.9981 |
| **cnn_npr** ✅ | **0.99876** |

→ Le gagnant est **`cnn_npr` seul**, avec un **seuil de 0.4425**.

Dans le stacking, le poids le plus fort allait aussi à NPR (coefficients 1.11 / 2.19 / 2.65). Ça confirme que **NPR est le signal le plus utile**.

### Étape ⑥ — Le test honnête sur `val/`

J'ouvre `val/` **une seule fois**, après avoir tout figé.

- **F1 holdout = 0.99847**
- Balanced accuracy = 0.99693
- Matrice de confusion :

```
                  prédit vrai   prédit truqué
vraie photo           324            2
image truquée           0          652
```

→ **0 truquage raté**, seulement **2 fausses alertes** sur 978 images.
Le F1 OOF (0.99876) et le F1 holdout (0.99847) sont très proches : **pas de surapprentissage** visible.

### Étape ⑦ — Réentraînement final et soumission

- Je fusionne `train + val` (plus de données = meilleur modèle).
- Nombre d'epochs = médiane des meilleurs epochs trouvés en OOF.
- Je garde **le même candidat et le même seuil** qu'avant. Je ne les ré-optimise pas.
- Je prédis `test/` et j'écris `submission.csv` (colonnes `image_id`, `classe`).
- Je vérifie le fichier : pas de doublon, toutes les images présentes, valeurs dans {0, 1}.
- Je sauvegarde un **audit JSON** (`audit_level1_v7.json`) avec tous les scores, seuils, versions des librairies. C'est la preuve de reproductibilité.

---

## 6. Résumé des résultats

| Mesure | Valeur |
|---|---|
| Modèle retenu | ConvNeXt-V2 Tiny en entrée NPR |
| Seuil | 0.4425 |
| F1 OOF (train) | 0.99876 |
| **F1 sur val (jamais vu)** | **0.99847** |
| Erreurs sur val | 2 sur 978 |

---

## 7. Petit dictionnaire

| Mot | Explication simple |
|---|---|
| **Pré-entraîné** | Modèle qui a déjà appris sur des millions d'images. Je le réutilise au lieu de partir de zéro. |
| **Fine-tuning** | Réentraîner un peu un modèle pré-entraîné sur mes données. |
| **Epoch** | Un passage complet sur toutes les images d'entraînement. |
| **Fold** | Une partie du dataset dans la validation croisée. |
| **OOF** | Prédiction faite par un modèle qui n'a jamais vu l'image. |
| **Holdout** | Données mises de côté, utilisées seulement à la fin pour vérifier. |
| **Fuite de données** | Le modèle voit, sans le vouloir, de l'information sur les données de test. Score faussement bon. |
| **Stacking** | Un petit modèle (ici régression logistique) qui apprend à combiner les réponses de plusieurs modèles. |
| **Logit** | `log(p / (1 − p))`. Transforme une probabilité en un nombre sans limite. Plus facile à combiner. |
| **Overfitting** | Le modèle apprend par cœur le train et marche mal sur de nouvelles données. |

---

## 8. Questions probables à l'entretien

**Q : Pourquoi le F1 et pas l'accuracy ?**
R : C'est la métrique imposée. Elle est aussi plus adaptée quand les classes ne sont pas équilibrées (ici 2/3 de truquées) : elle punit à la fois les fausses alertes et les truquages ratés.

**Q : Pourquoi avoir regroupé les images avec pHash ?**
R : Pour éviter la fuite de données. Sans ça, deux versions presque identiques d'une image pourraient être dans le train et la validation, et mon score serait trop optimiste.

**Q : Pourquoi ne pas avoir utilisé `val/` pour choisir le seuil ?**
R : Si je règle mes choix sur `val/`, son score n'est plus honnête. J'ai tout choisi sur l'OOF du train, puis `val/` sert de vrai contrôle final.

**Q : Pourquoi 3 experts différents ?**
R : Ils regardent des choses différentes : le sens global (SigLIP2), la texture (RGB), les traces de pixels (NPR). S'ils se trompent sur des images différentes, les combiner peut aider.

**Q : Mais au final tu n'as gardé que NPR, pourquoi ?**
R : La sélection est automatique : je prends le candidat avec le meilleur F1 OOF. NPR seul (0.99876) battait légèrement le stacking (0.99809). Je n'ai pas forcé l'ensemble juste parce qu'il « semble » mieux.

**Q : Comment tu as choisi le seuil ?**
R : J'ai testé 361 seuils entre 0.05 et 0.95 sur 5 sous-parties du train, et j'ai pris la médiane des 5 meilleurs seuils (0.4425). La médiane évite un seuil trop chanceux.

**Q : Pourquoi réentraîner sur train + val à la fin ?**
R : Plus de données = modèle un peu meilleur. Mais je garde le seuil et le choix de modèle figés avant, pour ne pas tricher.

**Q : Que ferais-tu pour améliorer ?**
R :
- Ne réentraîner à la fin que l'expert choisi (j'ai réentraîné les 3, ce qui coûte du temps GPU pour rien).
- Dans ce notebook, la recherche de doublons découpe l'empreinte en 4 morceaux : une paire avec exactement 4 bits différents peut être ratée. Au Level 2, je l'ai améliorée avec 5 morceaux.
- Tester plus de seeds et faire une moyenne.

---

## 9. Mon pitch en 30 secondes

> « Pour le Level 1, je devais dire si une image est vraie ou truquée, avec un F1 comme métrique.
> J'ai d'abord regroupé les quasi-doublons avec un pHash pour éviter les fuites entre folds.
> Ensuite, j'ai entraîné trois experts complémentaires : un modèle deepfake basé sur SigLIP2, un ConvNeXt-V2 sur l'image normale, et un ConvNeXt-V2 sur une représentation NPR qui fait ressortir les traces laissées par les générateurs.
> Tous mes choix — modèle, seuil, nombre d'epochs — ont été faits sur des prédictions out-of-fold. Le set de validation n'a été ouvert qu'une fois à la fin : j'obtiens un F1 de 0,998, avec seulement 2 erreurs sur 978 images. »
