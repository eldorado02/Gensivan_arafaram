# Level 3 — Explication très simple

Ce fichier est fait pour un **grand débutant**. Chaque mot technique est expliqué juste à côté, avec ce symbole : 📖.

Les mots de base (modèle, epoch, loss, CNN, pHash, fuite…) sont expliqués dans [../level1/EXPLICATION_SIMPLE.md](../level1/EXPLICATION_SIMPLE.md). Je les rappelle ici en une ligne quand ils reviennent.

---

## 1. Le jeu : entourer la retouche

Cette fois, je ne dois plus seulement dire « vrai » ou « faux ». Je dois **montrer où** se trouve la retouche, comme si j'**entourais la zone au feutre**.

- Si l'image est une vraie photo **retouchée à un endroit** → je dessine le contour de la zone retouchée.
- Si l'image est une **vraie photo normale** ou une image **entièrement générée** → je ne dessine **rien** (je réponds par un espace `" "`).

📖 **Polygone** = une forme fermée faite de points reliés par des traits (un triangle, un carré, ou une forme irrégulière). Je donne la **liste des points** du contour.

📖 **Coordonnées** = la position d'un point : x (de gauche à droite) et y (de haut en bas).

📖 **Normalisé** = au lieu de donner la position en pixels, je la donne entre 0 et 1. Exemple : x = 0.5 veut dire « au milieu de l'image ».

📖 **Segmentation** = répondre **pour chaque pixel** : « ce pixel est-il retouché, oui ou non ? ». C'est plus précis que de répondre une seule fois pour toute l'image.

📖 **Masque** = une image en noir et blanc de la même taille que la photo : **blanc = retouché**, **noir = normal**. Mon modèle dessine des masques, puis je les transforme en polygones.

---

## 2. Les données

| Paquet | Nombre |
|---|---|
| train | 7 857 |
| val | 978 |
| test | 993 |

- Toutes les images font **640 × 640 pixels** (📖 un pixel = un petit point de couleur).
- Environ **1 image sur 3 est retouchée**. Les **2 autres n'ont rien à entourer**.

---

## 3. Comment je suis noté : l'IoU

📖 **IoU** = *Intersection over Union* (« intersection sur union »). C'est une note entre 0 et 1 qui mesure **si ma zone dessinée tombe bien sur la vraie zone**.

Imagine deux feuilles de papier découpées, posées l'une sur l'autre : ma forme et la vraie forme.
- 📖 **Intersection** = la partie où les deux feuilles **se superposent**.
- 📖 **Union** = toute la surface couverte par **au moins une** des deux feuilles.

```
IoU = surface qui se superpose / surface totale couverte
```

- Mes deux formes sont identiques → **IoU = 1** (parfait).
- Elles ne se touchent pas du tout → **IoU = 0**.

La note finale est la **moyenne** des IoU de toutes les images.

**La règle très importante des images sans retouche :**

| La vraie réponse | Ma réponse | Note |
|---|---|---|
| rien à entourer | je n'entoure rien | **1** ✅ |
| rien à entourer | j'entoure quelque chose | **0** ❌ |
| il y a une retouche | je n'entoure rien | **0** ❌ |

**Conclusion** : 2 images sur 3 n'ont rien à entourer. Si je dessine même un tout petit point sur une vraie photo, j'ai **0** pour cette image. Il faut donc d'abord bien décider **s'il y a une retouche ou non**.

---

## 4. Le grand plan

1. **Écrire les outils de traduction** : masque ↔ polygone, et les vérifier.
2. **Enlever la triche** (images jumelles) et faire **3 paquets séparés** dans le train.
3. **Construire et entraîner 2 versions** de mon modèle (V2 et V3).
4. **Choisir la meilleure version** avec une règle fixée à l'avance.
5. **Régler les derniers boutons** (seuils, « porte ») sur un autre paquet.
6. **Examen blanc** une seule fois (val).
7. **Réviser tout**, puis le vrai examen (test).

---

## 5. Les étapes, une par une

### Étape 1 — Les outils de traduction

Mon modèle dessine des **masques**. Kaggle veut des **polygones**. J'ai donc écrit deux outils :
- **polygone → masque** : je « colorie » l'intérieur du polygone en blanc ;
- **masque → polygone** : je suis le bord de chaque tache blanche et je note les points.

📖 **OpenCV** (`cv2`) = une grande boîte à outils gratuite pour travailler sur les images. Elle sait colorier une forme, trouver un contour, etc.

📖 **Test unitaire** = un petit test automatique. Au début du code, je dessine 2 faux rectangles, je les transforme en polygones puis de nouveau en masque, et je vérifie que j'ai bien retrouvé les mêmes rectangles. Comme ça, je sais que mes outils marchent **avant** de lancer des heures de calcul.

**Point fort** : toutes mes notes sont calculées **après cette traduction**, exactement comme Kaggle le fera. Pas de surprise.

### Étape 2 — Enlever la triche et faire 3 paquets

📖 **pHash** (rappel) = l'empreinte visuelle d'une image. Les images jumelles vont dans la même famille.

Puis je coupe le train en **3 paquets séparés** :

| Paquet | Taille | Rôle | Comparaison avec l'école |
|---|---|---|---|
| **fit** | 5 596 | **Apprendre** | Les cours |
| **selection** | 1 127 | **Choisir** la meilleure version du modèle | Le contrôle pour choisir quel élève envoyer |
| **calib** | 1 134 | **Régler** les derniers boutons | L'entraînement aux conditions de l'examen |

📖 **Calibration** (*calib*) = le réglage fin des seuils, sur des images à part.

Une même famille d'images n'est **jamais** dans deux paquets différents. Le code le vérifie.

**Pourquoi 3 paquets ?** Chaque décision est prise sur des images **neuves** pour cette décision. Sinon, je me donnerais des notes trop belles.

### Étape 3 — Mon modèle : ForensicFPN

📖 **Forensique** = l'analyse technique pour trouver des **preuves**, comme la police scientifique. Ici : trouver des preuves de retouche dans une image.

J'ai **construit** ce modèle en assemblant 3 morceaux. Imagine **deux détectives** qui travaillent ensemble :

**Détective 1 — ConvNeXt-V2 : celui qui comprend ce qu'il voit**
- 📖 C'est un **CNN** (réseau qui fait glisser de petits filtres sur l'image).
- Il est **pré-entraîné** (📖 il a déjà appris sur des millions d'images).
- Il reconnaît les **objets** : visages, voitures, ciel…

**Détective 2 — la branche SRM : celui qui regarde le grain de l'image**
- 📖 **SRM** = *Spatial Rich Model*. Ce sont 3 filtres **fixes**, connus depuis longtemps en forensique.
- Ils **effacent le contenu** et ne gardent que le **bruit** de l'image.
- 📖 **Bruit** = les toutes petites variations aléatoires de pixels, comme le « grain » d'une photo. Chaque appareil photo et chaque compression laissent leur propre grain.
- **Pourquoi c'est utile** : une zone collée ou repeinte n'a souvent **pas le même grain** que le reste de la photo. Même si elle a l'air parfaite à l'œil.
- 📖 **Filtre passe-haut** = un filtre qui garde seulement les petits détails rapides (le grain) et enlève les grandes zones lisses.

**Le bureau commun — le FPN : là où les deux détectives mettent leurs notes ensemble**
- 📖 **FPN** = *Feature Pyramid Network* (« réseau en pyramide »).
- Les détectives regardent l'image **à plusieurs distances** : de très près (petits détails) et de loin (vue d'ensemble).
- Le FPN **mélange toutes ces vues**. La vue de loin aide à comprendre, la vue de près aide à **dessiner des bords précis**.

**Les 3 réponses du modèle :**
1. le **masque** : pour chaque pixel, la probabilité qu'il soit retouché ;
2. le **contour** : où sont les bords de la zone retouchée (ça aide à être précis) ;
3. la **classe** : une seule probabilité pour toute l'image, « est-elle retouchée ? ».

📖 **Probabilité** = un nombre entre 0 et 1 qui dit à quel point le modèle est sûr.

**Deux versions :**
- **V2** : la version de base, en 512 pixels.
- **V3** : une version améliorée, en **640 pixels** (la vraie taille des images), avec un SRM plus précis, des petites sorties en plus pour mieux apprendre, et une étape qui **repasse sur le masque** pour l'améliorer.

### Comment le modèle apprend

📖 **Loss** (rappel) = le nombre qui dit à quel point le modèle se trompe. Ici, j'en additionne plusieurs, chacune avec un rôle :

| Loss | En mots simples |
|---|---|
| **Focal loss** | Vérifie chaque pixel, en insistant sur les pixels **difficiles** (pas sur le fond facile) |
| **Soft IoU** | Essaie directement d'avoir **une bonne note IoU** |
| **Lovász** (V3) | Une autre façon, plus précise, de viser une bonne note IoU |
| **Loss de classe** | Apprend à dire si l'image entière est retouchée |
| **Loss de contour** | Apprend à bien placer les bords |

**Les astuces d'entraînement :**

📖 **Échantillonnage équilibré** = quand je donne les images au modèle, je mets **moitié retouchées, moitié normales**. Sinon il verrait surtout des images sans retouche et apprendrait à toujours dire « rien ».

📖 **Augmentation** = je modifie les images au hasard : je les retourne, je les tourne, je les compresse en **JPEG** (📖 le format photo le plus courant, qui abîme un peu l'image pour la rendre plus légère), j'ajoute un léger flou. **Important** : je fais **exactement la même modification** au masque, sinon il ne correspondrait plus à l'image.

📖 **EMA** = *Exponential Moving Average* (« moyenne glissante »). Je garde une **copie calme** du modèle, qui fait la moyenne de ses versions récentes. Comme quand on juge un élève sur sa moyenne plutôt que sur une seule note. Cette copie est plus stable.

📖 **Snapshot** (« photo instantanée ») = une sauvegarde du modèle à la fin d'une epoch (📖 un tour complet sur toutes les images).

📖 **Gradient checkpointing** = une astuce pour **économiser la mémoire** de la carte graphique : l'ordinateur recalcule certaines choses au lieu de tout garder en mémoire. Plus lent, mais ça tient.

📖 **OOM** = *Out Of Memory* (« plus de mémoire »). Mon code essaie d'abord avec 4 images à la fois, puis 2, puis 1, et garde la plus grande taille qui ne provoque pas d'OOM.

### Étape 4 — Choisir entre V2 et V3

J'ai écrit la règle **avant** de voir les résultats. V3 n'est choisie que si :
- elle est **meilleure d'au moins 0.002**, **et**
- elle ne fait **pas plus de fausses alertes** que V2.

📖 **Fausse alerte** (ou **faux positif**, 📖 **FPR** = *False Positive Rate*, le taux de fausses alertes) = dessiner une zone sur une image qui n'a pas de retouche.

| | V2 | V3 |
|---|---|---|
| Note (IoU) | 0.931 | **0.939** |
| Fausses alertes | 0.14 % | **0 %** |

→ **V3 est choisie.**

### Étape 5 — Régler les derniers boutons

Mon modèle donne des probabilités. Il faut décider à partir de quand on dessine. J'ai testé **384 combinaisons** de réglages sur le paquet `calib`, et j'ai gardé la meilleure.

**Le bouton le plus important : la porte**

📖 **Porte** (*gate* en anglais) = une décision prise **avant** de dessiner : « cette image est-elle retouchée ? ». Si non → je ne dessine rien.

Comme un vigile à l'entrée : il laisse passer seulement les images suspectes.

Mode choisi : 📖 **rank** (« classement »). Je classe toutes les images de la plus suspecte à la moins suspecte, et je laisse passer **seulement le tiers le plus suspect**. Pourquoi un tiers ? Parce qu'environ 1 image sur 3 est retouchée.

**Les autres boutons :**
- **seuil des pixels = 0.45** : un pixel est colorié en blanc si sa probabilité est au moins 0.45 ;
- 📖 **seuil** = la limite pour passer de la probabilité à la décision oui/non.

**Deux idées testées, puis écartées parce qu'elles n'apportaient pas assez :**

📖 **Retrieval** (« retrouver ») = chercher dans le train **la photo originale non retouchée**, puis comparer les deux pour voir ce qui a changé. Ça ne marchait que pour 3,8 % des images → **pas utilisé**.

📖 **TTA** = *Test-Time Augmentation*. Je montre au modèle l'image **et des versions retournées**, puis je fais la moyenne des réponses. J'ai testé 2 versions et 8 versions : 8 n'apportait pas assez → **2 versions gardées**.

### Étape 6 — L'examen blanc (val)

📖 **Holdout** = le paquet mis de côté, ouvert **une seule fois** à la fin.

Résultats :
- **note IoU moyenne = 0.94** ;
- la **porte** se trompe seulement **4 fois sur 978** ;
- sur les images retouchées, mes dessins ont une note moyenne de **0.82**.

**Comment lire ces résultats** : la porte est presque parfaite. Ce qui peut encore s'améliorer, c'est la **précision du dessin** sur les images retouchées.

### Étape 7 — Réviser tout, puis le vrai examen

- Je réentraîne V3 sur **train + val**.
- 📖 **Garde temporelle** = Kaggle limite le temps de calcul à environ 7 heures. Mon code **surveille l'horloge** et s'arrête proprement avant la limite. Ici, il a fait **5 tours sur 8 prévus** pour être sûr de finir.
- Le modèle répond sur le test : **328 images** reçoivent un polygone, les autres un espace `" "`.
- J'écris `submission.csv` (📖 le fichier de réponses) et l'**audit** (📖 le carnet de bord avec toutes les notes).
- Durée totale : **5,7 heures** sur une seule carte graphique (📖 **GPU** T4).

---

## 6. Ce qui change par rapport aux Levels 1 et 2

| | Levels 1 et 2 | Level 3 |
|---|---|---|
| Je réponds | une fois par image | **pour chaque pixel** |
| Ma réponse | un chiffre (0, 1 ou 2) | un **polygone** (ou rien) |
| Note | F1 / Macro F1 | **IoU** |
| Modèle | modèles existants | **un modèle que j'ai assemblé** (ForensicFPN) |
| Ce qui regarde le « grain » | NPR | **SRM** |
| Paquets dans le train | 3 folds | **fit / selection / calib** |

---

## 7. À retenir en 5 phrases

1. Je dois **entourer la zone retouchée**, ou ne rien dessiner s'il n'y a pas de retouche.
2. Comme **2 images sur 3** n'ont rien à entourer, une **porte** décide d'abord si l'image est retouchée.
3. Mon modèle a **deux détectives** : un qui comprend le contenu (ConvNeXt), un qui regarde le grain de l'image (SRM), et le **FPN** mélange leurs notes.
4. J'ai choisi V3 et réglé les seuils sur des paquets **séparés**, avec des règles fixées à l'avance.
5. À l'examen blanc : **0.94**, avec seulement 4 erreurs de porte sur 978.
