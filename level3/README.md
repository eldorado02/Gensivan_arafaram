# Level 3 — Localisation : où est la retouche ?

Notebook : [level3.ipynb](level3.ipynb)

---

## 1. Le problème en une phrase

Pour chaque image de `test/`, je dois **dessiner la zone retouchée** sous forme de **polygone**.

- Image **retouchée localement** (Cat 2) → je donne les coordonnées du polygone autour de la zone modifiée.
- Image **vraie** (Cat 0) ou **entièrement générée** (Cat 1) → je donne une chaîne avec **un espace** (`" "`), c'est-à-dire « rien à signaler ».

Ce n'est plus de la classification : c'est de la **segmentation**. Le modèle doit répondre **pixel par pixel** : « ce pixel est-il retouché ? »

---

## 2. Les données

| Dossier | Images | Labels |
|---|---|---|
| `train/` | 7 857 | `segmentation.csv` (polygone ou vide) |
| `val/` | 978 | idem |
| `test/` | 993 | à prédire |

- Toutes les images font **640 × 640 pixels**.
- Environ **1/3 des images ont un masque** (Cat 2) et **2/3 sont vides** (Cat 0 et Cat 1).
  Sur `val/` : 326 avec masque, 652 vides.
- Les coordonnées des polygones sont **normalisées** entre 0 et 1 (x / largeur, y / hauteur). Plusieurs polygones sont séparés par `;`.

---

## 3. La métrique : l'IoU moyen

Pour chaque image, je compare mon masque prédit **P** au vrai masque **G** :

```
IoU = (pixels où G = 1 ET P = 1) / (pixels où G = 1 OU P = 1)
    = intersection / union
```

- IoU = 1 → mon masque est parfait.
- IoU = 0 → aucun recouvrement.

Le score final est la **moyenne des IoU** sur toutes les images.

**Le cas très important des images vides :**

| Vrai masque | Mon masque | IoU |
|---|---|---|
| vide | vide | **1.0** ✅ |
| vide | non vide | **0.0** ❌ (fausse alerte) |
| non vide | vide | **0.0** ❌ (retouche ratée) |

Comme **2/3 des images sont vides**, il est essentiel de **ne rien dessiner** sur les images non retouchées. Une seule tache dessinée sur une vraie photo donne 0 pour cette image.
→ C'est pour ça que mon pipeline a une **« porte »** (*gate*) qui décide d'abord : « cette image est-elle retouchée ? »

---

## 4. Mon approche en un schéma

```
            Images du train (7 857)
                     │
   ① Codec polygone ↔ masque (+ tests automatiques)
                     │
   ② Groupes pHash → 3 sous-ensembles sans fuite
        fit (5 596)  |  selection (1 127)  |  calib (1 134)
                     │
   ③ Entraîner 2 candidats sur « fit »
        ├── V2 baseline  : ForensicFPN, 512 px
        └── V3 enhanced  : ForensicFPN amélioré, 640 px
                     │
   ④ Sur « selection » : choisir V2 ou V3 (règle automatique)
                     │
   ⑤ Sur « calib » : régler le post-traitement (seuils, porte…)
                     │
   ⑥ Vérifier UNE SEULE FOIS sur val/
                     │
   ⑦ Réentraîner sur train + val → test → polygones → submission.csv
```

**Pourquoi 3 sous-ensembles ?** Chaque décision est prise sur des données que le modèle n'a pas vues, et **chaque étape utilise des données différentes**. Choisir l'architecture et régler les seuils sur les mêmes images donnerait un score trop optimiste.

---

## 5. Les étapes en détail

### Étape ① — Le codec polygone ↔ masque

Le modèle travaille avec des **masques** (images noir et blanc). La soumission demande des **polygones**. J'ai donc écrit deux fonctions :

- `polygon_to_mask` : lit le texte du polygone, remet les coordonnées en pixels, remplit la forme avec `cv2.fillPoly`.
- `mask_to_polygon` : trouve les contours du masque (`cv2.findContours`), peut les simplifier (`cv2.approxPolyDP` avec un paramètre `epsilon`), les normalise entre 0 et 1, et les sépare par `;`. Un masque vide donne `" "`.
- `official_iou` : l'IoU avec la règle « vide + vide = 1 ».

Je teste ces fonctions au début du notebook avec un faux masque à 2 rectangles. On vérifie que l'aller-retour masque → polygone → masque garde un IoU > 0.90.

**Point fort** : tous mes scores sont calculés **après l'aller-retour polygone**, à la résolution d'origine. Je mesure donc exactement ce que Kaggle va mesurer, pertes de conversion comprises.

### Étape ② — Les sous-ensembles sans fuite

- pHash de chaque image, groupes de quasi-doublons (distance ≤ 4 bits) avec Union-Find. Ici l'empreinte est coupée en **8 morceaux**.
- `StratifiedGroupKFold` à 7 folds → 1 fold devient **calib** (1 134 images).
- Le reste est recoupé en 6 → 1 partie **selection** (1 127), le reste **fit** (5 596).
- La stratification se fait sur « a un masque / n'a pas de masque ».
- Je vérifie avec un `assert` qu'aucun groupe n'est partagé entre fit, selection et calib.

### Étape ③ — Le modèle : ForensicFPN

C'est un modèle de segmentation que j'ai construit en assemblant plusieurs blocs.

```
                 image
            ┌──────┴──────┐
     ConvNeXt-V2       Branche SRM
   (ce qu'il y a      (le « bruit » de
    dans l'image)      l'image : filtres
            │          fixes forensiques)
            │               │
       4 échelles      4 échelles
            └──────┬────────┘
          fusion à chaque échelle
                   │
        FPN (on remonte du grossier au fin)
                   │
     ┌─────────────┼──────────────┐
  tête masque   tête contour   classifieur image
 (pixel retouché?) (bords)     (image retouchée ?)
```

**Les 3 morceaux importants :**

1. **Encodeur ConvNeXt-V2 Tiny** (pré-entraîné) : il comprend le **contenu** (visages, objets…). Il sort 4 cartes de caractéristiques, du détail fin au global.
2. **Branche SRM** (*Spatial Rich Model*) : 3 **filtres fixes passe-haut**, classiques en forensique d'image. Ils enlèvent le contenu et gardent le **bruit** de l'image. Une zone collée ou repeinte n'a souvent pas le même bruit que le reste de la photo. Ensuite, quelques couches de convolution apprennent à lire ce bruit.
3. **FPN** (*Feature Pyramid Network*) : je combine les informations de toutes les échelles. Le global aide à comprendre, le fin aide à dessiner des bords précis.

**Les 3 sorties :**

- **masque** : probabilité de retouche pour chaque pixel ;
- **contour** : les bords de la zone retouchée (aide le modèle à être précis sur les bords) ;
- **classe** : une probabilité pour toute l'image, « retouchée ou non ». Elle sert à la porte.

**V2 baseline vs V3 enhanced :**

| | V2 baseline | V3 enhanced |
|---|---|---|
| Résolution | 512 px | 640 px (taille réelle des images) |
| SRM | moyenne simple des couleurs | vraie luminance + bords « reflect » |
| Têtes auxiliaires | non | oui (supervision à plusieurs échelles) |
| Tête de raffinement | non | oui (utilise masque + contour) |
| Classifieur image | branche SRM seule | moyenne SRM + RGB |
| Loss | de base | + Lovász + contour équilibré + auxiliaires |

### La loss (fonction d'erreur), pièce par pièce

Mon modèle apprend avec une **somme de plusieurs losses** :

| Loss | Poids | À quoi elle sert |
|---|---|---|
| **Focal BCE** sur le masque | 0.65 | Erreur pixel par pixel. La version « focal » insiste sur les pixels difficiles. |
| **Soft IoU** (images avec masque seulement) | 0.85 | Optimise directement quelque chose qui ressemble à l'IoU, la métrique finale. |
| **BCE classe** | 0.35 | Apprend à dire si l'image entière est retouchée. |
| **Contour** | 0.15 | Apprend les bords de la zone. |
| *V3 :* **Lovász hinge** | 0.20 | Autre approximation de l'IoU, calculée à 1/4 de résolution pour aller vite. |
| *V3 :* **Têtes auxiliaires** | 0.12 | Chaque échelle apprend aussi à prédire le masque. |

*Petit bug corrigé dans Lovász* : `grad[1:] -= grad[:-1]` faisait une erreur PyTorch car les deux tranches se chevauchent. J'ai ajouté `.clone()` pour copier avant de soustraire.

### Les réglages d'entraînement

- **Échantillonnage équilibré** (`WeightedRandomSampler`) : 50 % d'images avec masque, 50 % sans, dans chaque batch.
- **Augmentations** (albumentations, appliquées **en même temps** à l'image et au masque) : flip, rotation 90°, compression JPEG, flou, couleur.
- **Batch automatique** (`safe_batch`) : j'essaie 4, puis 2, puis 1 image. Je garde la plus grande taille qui ne fait pas d'erreur mémoire (OOM). Puis j'accumule les gradients jusqu'à un batch effectif de 16.
- **Gradient checkpointing** sur l'encodeur : économise la mémoire GPU.
- AdamW : LR 3e-5 pour l'encodeur pré-entraîné, 1.2e-4 pour les nouvelles couches. Scheduler cosine. Mixed precision. Clipping.
- **EMA** (*Exponential Moving Average*, decay 0.995) : je garde une copie « lissée » des poids, moyenne des poids récents. Elle est souvent plus stable que les poids bruts.
- **Snapshots** : je garde les poids EMA des 3 dernières epochs. Plus tard, je compare « dernier » et « moyenne des 3 ».
- 6 epochs pour la comparaison.

### Étape ④ — Choisir V2 ou V3 (sur « selection »)

Règle **écrite à l'avance** : V3 n'est retenu que si

- il gagne **au moins +0.002** d'IoU sur V2, **et**
- il ne fait **pas plus de fausses alertes** que V2.

| | V2 | V3 |
|---|---|---|
| IoU sur selection | 0.9307 | **0.9392** |
| Taux de fausses alertes | 0.14 % | **0 %** |

→ **V3 est retenu** (snapshot « last »).

### Étape ⑤ — Régler le post-traitement (sur « calib »)

Le modèle sort des **probabilités**. Il faut les transformer en masque final. Je teste **384 combinaisons** de réglages :

| Réglage | Valeurs testées | Signification |
|---|---|---|
| `alpha` | 1.0 / 0.85 / 0.70 | Mélange entre la proba « image » et les pixels les plus suspects (top 0,1 %) |
| `gate_mode` | `threshold` ou `rank` | Comment la porte décide |
| `class_threshold` | 0.4 / 0.5 / 0.6 | Seuil de la porte (mode threshold) |
| `pixel_threshold` | 0.45 → 0.60 | À partir de quelle proba un pixel est « retouché » |
| `minimum_area` | 0 / 128 | Supprimer les petites taches |
| `retrieval_weight` | 0 / 0.25 | Utiliser ou non la méthode « retrieval » (voir plus bas) |
| `closing` | oui / non | Boucher les petits trous du masque |

Puis je règle `epsilon` (simplification du polygone).

**Comment marche la porte (gate) :**

1. Je calcule une **« preuve »** par image : `alpha × proba_image + (1 − alpha) × moyenne des 0,1 % de pixels les plus suspects`.
2. Mode `threshold` : si la preuve est sous le seuil → masque vide.
   Mode `rank` : je garde seulement le **tiers des images** avec la plus forte preuve (car environ 1/3 des images sont retouchées). Les autres → masque vide.
3. Pour les images qui passent la porte : `pixel >= pixel_threshold` → pixel retouché.

**Réglage choisi** : `alpha = 0.85`, porte en mode **rank**, seuil pixel **0.45**, pas de nettoyage, pas de retrieval, `epsilon = 0` (polygone non simplifié).

**Le « retrieval » (idée testée mais non retenue)**
Idée : si je trouve dans le train la **version originale non retouchée** d'une image (via pHash), je l'aligne (`cv2.findTransformECC`) et je calcule la **différence** de couleur et de bords. La zone qui change = la retouche.
Résultat : ça ne marche que pour **3,8 %** des images, et le gain était trop faible. La règle de sélection a donc mis son poids à 0. Je l'ai gardé dans le code, mais il n'est pas utilisé.

**TTA** (*Test-Time Augmentation*) : à la prédiction, je regarde l'image normale et retournée, puis je fais la moyenne (2 vues). J'ai aussi testé 8 vues (rotations + flips). Ça n'apportait pas assez (+0.001 exigé), donc je suis resté à 2 vues.

### Étape ⑥ — Le test honnête sur `val/`

Ouvert **une seule fois**, après tous les choix.

- **IoU moyen = 0.9385**
- IoU moyen sur les images **avec masque seulement** = 0.8217
- Porte (retouchée ou non) :

```
                    prédit vide   prédit masque
vraiment vide           650             2
vraiment retouchée        2           324
```

**Comment lire 0.9385 :**
- 650 images vides bien laissées vides → IoU = 1 chacune.
- 2 fausses alertes + 2 retouches ratées → IoU = 0.
- Les 326 images retouchées ont en moyenne un IoU de 0.82.
- `(650 × 1 + 326 × 0.8217) / 978 ≈ 0.9385`.

→ **La porte est presque parfaite** (4 erreurs sur 978). Le vrai point d'amélioration, c'est **la précision du dessin** sur les images retouchées (0.82).

### Étape ⑦ — Final

- Réentraînement de V3 sur `train + val`, 8 epochs prévues.
- **Garde temporelle** : Kaggle limite le temps de calcul. Le notebook surveille l'horloge (budgets de 3,75 h, 4,25 h, 5,5 h, 6,75 h). Le réentraînement s'est **arrêté à 5 epochs** pour être sûr de finir. Le notebook l'a noté dans les warnings de l'audit.
- Prédiction du test → porte → masque → polygone.
- **328 images sur 993** reçoivent un polygone, les autres `" "`.
- Contrôles : 993 lignes, pas de doublon, mêmes `image_id` que le sample.
- Audit JSON `audit_level3_v3.json` + sauvegarde des cartes de probabilité (zip) pour analyser les erreurs.
- Durée totale : **5,7 h** sur un seul GPU T4, pic mémoire 2,7 Go.

---

## 6. Résumé des résultats

| Mesure | Valeur |
|---|---|
| Architecture retenue | V3 enhanced (ForensicFPN, 640 px) |
| IoU selection (V2 → V3) | 0.9307 → 0.9392 |
| **IoU moyen sur val (jamais vu)** | **0.9385** |
| IoU sur les images retouchées seules | 0.8217 |
| Erreurs de la porte sur val | 4 sur 978 |
| Score public de la version V2 précédente | 0.87076 |

**Point de vigilance** : la V2 avait 0.87 sur le leaderboard public, alors qu'elle fait environ 0.93 en validation interne. Le test public peut donc être plus difficile que ma validation, ou le leaderboard public est calculé sur peu d'images. C'est une limite à connaître et à reconnaître honnêtement.

---

## 7. Petit dictionnaire

| Mot | Explication simple |
|---|---|
| **Segmentation** | Prédire une classe pour **chaque pixel**. |
| **Masque** | Image noir et blanc : blanc = zone retouchée. |
| **IoU** | Intersection / union entre deux masques. Mesure le recouvrement. |
| **Encodeur** | Partie du réseau qui « comprend » l'image et la résume en caractéristiques. |
| **FPN** | Façon de combiner des informations à plusieurs échelles (fin + global). |
| **SRM** | Filtres classiques de forensique qui extraient le bruit de l'image. |
| **Passe-haut** | Filtre qui garde les détails fins et enlève les grandes zones lisses. |
| **Focal loss** | Loss qui donne plus d'importance aux exemples difficiles. |
| **EMA** | Moyenne glissante des poids du modèle, plus stable. |
| **TTA** | Prédire sur plusieurs versions de l'image (flip, rotation) et faire la moyenne. |
| **Gate / porte** | Décision « l'image est-elle retouchée ? » avant de dessiner le masque. |
| **OOM** | *Out Of Memory* : la carte graphique n'a plus de mémoire. |

---

## 8. Questions probables à l'entretien

**Q : Pourquoi la « porte » est-elle si importante ?**
R : Deux tiers des images n'ont pas de retouche. Si je laisse un masque vide sur ces images, j'ai IoU = 1. Si je dessine ne serait-ce qu'un pixel, j'ai 0. La porte décide donc d'abord si l'image est retouchée, puis seulement je dessine.

**Q : Pourquoi le mode « rank » a gagné ?**
R : Il garde le tiers des images les plus suspectes, ce qui correspond à la proportion réelle de Cat 2 dans les données. C'est robuste, même si les probabilités ne sont pas parfaitement calibrées. Mais il suppose que le test a la même proportion. Si ce n'est pas le cas, c'est un risque.

**Q : Pourquoi ajouter une branche SRM au lieu d'utiliser juste ConvNeXt ?**
R : ConvNeXt voit le contenu de l'image. Une bonne retouche a un contenu crédible. Le SRM regarde le bruit du capteur et de la compression : une zone collée ou générée a souvent un bruit différent du reste. Les deux informations se complètent.

**Q : Pourquoi plusieurs losses ?**
R : La BCE apprend pixel par pixel. Soft IoU et Lovász optimisent quelque chose de proche de l'IoU, la vraie métrique. La loss de contour rend les bords précis. La loss de classe entraîne la porte.

**Q : Pourquoi 3 sous-ensembles (fit, selection, calib) ?**
R : Pour que chaque décision soit prise sur des données neuves : fit pour apprendre, selection pour choisir l'architecture, calib pour régler les seuils. `val/` ne sert qu'au contrôle final.

**Q : Pourquoi as-tu calculé l'IoU après conversion en polygone ?**
R : Kaggle reçoit des polygones, pas des masques. La conversion peut perdre des pixels. Mesurer après l'aller-retour donne exactement le score que Kaggle calculera.

**Q : Le sujet parle de RLE, mais tu rends des polygones ?**
R : La section « Tâche » et le `sample_submission.csv` demandent une colonne `polygon`. J'ai suivi le format de soumission réel. L'évaluation, elle, compare des masques : c'est pour ça que je reconvertis mes polygones en masques pour me noter.

**Q : Ton réentraînement s'est arrêté à 5 epochs sur 8, c'est grave ?**
R : C'est une sécurité voulue : Kaggle a une limite de temps, et un notebook qui ne finit pas ne produit pas de soumission. Le modèle à 5 epochs est un peu moins entraîné que prévu, mais il est sur plus de données (train + val). Pour améliorer, on pourrait réduire la phase de comparaison pour laisser plus de temps au final.

**Q : Que ferais-tu pour améliorer ?**
R :
- Améliorer le dessin sur les images retouchées (IoU 0.82) : plus grande résolution, encodeur plus gros, plus d'epochs.
- Comprendre l'écart entre la validation interne (~0.93) et le leaderboard public (0.87).
- Rendre la porte moins dépendante de la proportion « 1/3 » (mode rank).

---

## 9. Mon pitch en 30 secondes

> « Le Level 3 demandait de localiser la zone retouchée sous forme de polygone, avec l'IoU moyen comme métrique.
> Comme deux tiers des images n'ont pas de retouche, j'ai construit une porte qui décide d'abord si l'image est retouchée, puis un modèle de segmentation qui dessine la zone.
> Ce modèle, que j'appelle ForensicFPN, combine un ConvNeXt-V2 pour le contenu et une branche SRM pour le bruit de l'image, fusionnés dans un FPN avec trois sorties : masque, contour et classe.
> J'ai comparé deux versions avec une règle fixée à l'avance, réglé le post-traitement sur un jeu de calibration séparé, et mesuré tous mes scores après la vraie conversion en polygones.
> Sur la validation, ouverte une seule fois, j'obtiens un IoU moyen de 0,94, avec seulement 4 erreurs de porte sur 978 images. »
