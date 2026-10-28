# Level 2 — Explication très simple

Ce fichier est fait pour un **grand débutant**. Chaque mot technique est expliqué juste à côté, avec ce symbole : 📖.

Les mots de base (modèle, epoch, loss, fold, OOF, CNN, NPR…) sont expliqués dans [../level1/EXPLICATION_SIMPLE.md](../level1/EXPLICATION_SIMPLE.md). Je les rappelle ici en une ligne quand ils reviennent.

---

## 1. Le jeu : 3 boîtes au lieu de 2

Au Level 1, il y avait 2 boîtes : vrai ou faux. Ici, il y en a **3** :

| Réponse | Nom | Ça veut dire |
|---|---|---|
| **0** | `real` (« réel ») | Une vraie photo, pas modifiée |
| **1** | `generated` (« générée ») | Une image **entièrement** fabriquée par une IA |
| **2** | `local_edit` (« retouche locale ») | Une vraie photo où **une seule partie** a été changée (un visage remplacé, un objet effacé…) |

📖 **Classe** = une boîte, une catégorie de réponse.

📖 **Classification multi-classe** = ranger chaque image dans **une boîte parmi plusieurs** (ici 3).

**Pourquoi c'est plus dur ?** Une image `local_edit` est presque entièrement une vraie photo. Seul un petit morceau est faux. Le programme doit trouver **ce petit détail**.

---

## 2. Les données

| Paquet | Nombre | Détail |
|---|---|---|
| train | 7 857 | 2 619 images de **chaque** classe |
| val | 978 | 326 de chaque classe |
| test | 993 | l'examen |

📖 **Équilibré** = il y a autant d'images dans chaque classe.

---

## 3. Comment je suis noté : le Macro F1

📖 **F1** (rappel du Level 1) = une note entre 0 et 1 qui mesure si je trouve bien une classe sans faire trop de fausses alertes.

📖 **Macro F1** = je calcule **un F1 pour chaque classe**, puis je fais **la moyenne des 3** :

```
Macro F1 = (F1 des vraies + F1 des générées + F1 des retouchées) / 3
```

**Pourquoi c'est important** : chaque classe compte **autant**. Si j'oublie complètement les retouches locales, je perds un tiers de ma note. Je ne peux donc négliger aucune classe.

---

## 4. Le grand plan

C'est **la même méthode que le Level 1**, adaptée à 3 boîtes :

1. **Enlever la triche** : les images jumelles dans la même famille (pHash).
2. **Couper en 3 groupes** (folds) et **vérifier** qu'aucune famille n'est coupée.
3. **Entraîner 3 experts**, qui donnent maintenant **3 probabilités** chacun.
4. **Réponses honnêtes** (OOF).
5. **Mélanger les experts** (stacking) et ajouter un **petit coup de pouce par classe** (les biais).
6. **Examen blanc** une seule fois (val).
7. **Réviser tout**, puis le vrai examen (test).

---

## 5. Les étapes, une par une

### Étape 1 — Enlever la triche (pHash)

📖 **pHash** = l'empreinte visuelle d'une image (64 cases de 0 ou 1).

Comme au Level 1 : si deux empreintes ont **4 cases différentes ou moins**, les images sont jumelles et vont dans la même famille.

**Amélioration** : pour comparer plus vite, je coupe chaque empreinte en morceaux. Au Level 1 j'en faisais 4, ici j'en fais **5**. Avec 5 morceaux, deux images jumelles ont **forcément** au moins un morceau identique, donc je n'en rate **aucune**.

📖 **Principe des tiroirs** = si on range 4 chaussettes dans 5 tiroirs, au moins un tiroir reste vide. Ici : 4 différences ne peuvent pas toucher 5 morceaux, donc un morceau est identique.

### Étape 2 — Couper en 3 groupes + vérification

📖 **Fold** = un groupe d'images. Je fais 3 folds.

**Nouveauté** : le code **vérifie tout seul** qu'aucune famille n'est coupée entre deux groupes. S'il y en a une, il s'arrête avec le message « Fuite de groupes ». 📖 **Fuite** = triche involontaire.

### Étape 3 — Les 3 experts, avec 3 réponses chacun

📖 **Expert** = un modèle. Les mêmes 3 qu'au Level 1 :

| Expert | Ce qu'il regarde |
|---|---|
| **DeepGuard** (`deep_rgb`) | Le **sens global** de l'image (basé sur SigLIP2, un modèle de Google) |
| **ConvNeXt-V2 RGB** (`cnn_rgb`) | Les **textures** de la photo normale (📖 **RGB** = rouge, vert, bleu = photo normale) |
| **ConvNeXt-V2 NPR** (`cnn_npr`) | Les **traces entre pixels voisins** (📖 **NPR** = *Neighboring Pixel Relationships*) |

**Ce qui change** : chaque expert donne maintenant **3 probabilités**, une par classe. Par exemple :
```
vraie : 0.05    générée : 0.10    retouchée : 0.85
```
La somme fait toujours 1 (100 %).

📖 **Softmax** = le calcul qui transforme les 3 scores du modèle en 3 probabilités qui font 1 au total.

**L'astuce pour DeepGuard**
DeepGuard ne savait répondre que « vrai » ou « faux » (2 réponses). Il m'en fallait 3.

📖 **Tête** (*head*) = la toute dernière partie du modèle, celle qui donne la réponse.

J'ai remplacé sa tête par une nouvelle tête à 3 sorties, mais **je n'ai pas jeté ce qu'il savait** :
- la sortie « vraie » reçoit ce qu'il savait sur « vrai » ;
- les sorties « générée » et « retouchée » reçoivent toutes les deux ce qu'il savait sur « faux » ;
- j'ajoute un **tout petit peu de hasard** à la sortie « retouchée ».

**Pourquoi ce petit hasard ?** Sinon les sorties « générée » et « retouchée » seraient des jumelles parfaites. Elles apprendraient exactement la même chose et ne deviendraient jamais différentes. Le hasard les « décolle ». 📖 On appelle ça **casser la symétrie**.

### Comment ils apprennent : ce qui change

📖 **Loss** (rappel) = le nombre qui dit à quel point le modèle se trompe.

📖 **Cross-entropy** (« entropie croisée ») = la loss utilisée quand il y a **plusieurs** boîtes. Elle punit fort le modèle s'il donne une petite probabilité à la bonne boîte.

📖 **Label smoothing** (« adoucissement des réponses ») = au lieu de dire au modèle « la bonne réponse, c'est 100 % retouchée », je lui dis « c'est environ 98 % retouchée ». Ça l'empêche de devenir **trop sûr de lui**, et il se trompe moins sur des images nouvelles.

**Le problème d'ordinateur que j'ai réglé**
Kaggle me prêtait **2 GPU** (📖 cartes graphiques qui calculent vite). Mais quand j'utilisais les deux en même temps, le programme **plantait**. La communication entre les deux cartes était instable.
J'ai donc utilisé **une seule carte**. Pour que l'apprentissage reste identique, j'ai réglé la taille des paquets d'images.

📖 **Batch** = un paquet d'images traitées ensemble.
📖 **Accumulation de gradients** = si la carte ne peut prendre que 4 images à la fois, je fais 6 petits paquets avant de corriger le modèle. C'est comme si j'avais un paquet de 24. 📖 Le **gradient** = la direction dans laquelle il faut corriger le modèle.

### Étape 4 — Les réponses honnêtes (OOF)

📖 **OOF** = *Out-Of-Fold*. Chaque image du train reçoit des réponses données par un modèle qui **ne l'avait jamais vue**.

Pour chaque image : 3 experts × 3 probabilités = **9 nombres**.

### Étape 5 — Mélanger et donner un coup de pouce

**a) Mélanger les experts : le stacking**

📖 **Stacking** (« empilement ») = un petit modèle qui apprend **comment combiner** les réponses des experts. Il reçoit les 9 nombres et donne 3 nouvelles probabilités.

📖 **Régression logistique** = le petit modèle simple que j'utilise pour ce mélange. Il est simple exprès : avec seulement 9 nombres en entrée, un modèle compliqué apprendrait par cœur (📖 **overfitting**).

**b) Le coup de pouce par classe : les biais**

Normalement, je choisis la boîte qui a la plus grande probabilité.
📖 **Argmax** = « prendre la plus grande ». Si (0.05, 0.10, 0.85) → je choisis la boîte 2.

Mais parfois, il vaut mieux **aider un peu** une boîte difficile.

📖 **Biais** = un petit bonus (ou malus) que j'ajoute à une boîte **avant** de choisir la plus grande.

Comme un arbitre qui donne un petit avantage à l'équipe qui joue avec un handicap.

J'ai cherché les meilleurs bonus en essayant beaucoup de valeurs, des grandes puis des petites. J'ai gardé celles qui donnaient le meilleur Macro F1.

**c) Le résultat**

J'ai comparé 5 possibilités (📖 **candidats**) :

| Candidat | Note (Macro F1) |
|---|---|
| DeepGuard seul | 0.893 |
| ConvNeXt RGB seul | 0.993 |
| ConvNeXt NPR seul | 0.995 |
| Moyenne simple des 3 | 0.997 |
| **Stacking** ✅ | **0.997** (le plus haut) |

→ Gagnant : **le stacking**, avec un **bonus pour la boîte « retouchée »**. C'est logique : c'est la boîte la plus difficile.

**Différence avec le Level 1** : ici, **mélanger les experts** marche mieux que n'importe quel expert seul.

### Étape 6 — L'examen blanc (val)

📖 **Holdout** = le paquet mis de côté, ouvert **une seule fois** à la fin.

Résultat : **Macro F1 = 0.996**. Seulement **4 erreurs sur 978 images**.

📖 **Matrice de confusion** = le tableau des bonnes réponses et des erreurs :

```
                  j'ai dit vraie   j'ai dit générée   j'ai dit retouchée
vraiment vraie         324                1                   1
vraiment générée         0              325                   1
vraiment retouchée       0                1                 325
```

Les bonnes réponses sont sur la diagonale (324, 325, 325).

### Étape 7 — Réviser tout, puis le vrai examen

- Je réentraîne les 3 experts sur **train + val**.
- Je garde **le même mélange et les mêmes bonus** (pas de triche).
- Ils répondent sur le test → fichier `submission.csv` (📖 le fichier de réponses envoyé à Kaggle).
- J'écris un **audit** (📖 le carnet de bord avec toutes les notes et tous les réglages).
- 📖 **Cache** = des sauvegardes intermédiaires. Si Kaggle coupe le programme, je peux reprendre sans tout refaire.

---

## 6. Ce qui change par rapport au Level 1

| | Level 1 | Level 2 |
|---|---|---|
| Nombre de boîtes | 2 | 3 |
| Le modèle donne | 1 probabilité | 3 probabilités (softmax) |
| Loss | BCE | Cross-entropy + label smoothing |
| Je décide avec | un seuil | la plus grande probabilité + un bonus par boîte |
| Note | F1 | Macro F1 |
| Gagnant | NPR seul | Mélange des 3 (stacking) |

---

## 7. À retenir en 5 phrases

1. Je dois ranger chaque image dans **3 boîtes** : vraie, générée ou retouchée.
2. J'ai gardé ma méthode anti-triche du Level 1, en l'**améliorant** (5 morceaux, vérification automatique).
3. J'ai adapté DeepGuard de 2 à 3 réponses **sans perdre ce qu'il savait**.
4. J'ai **mélangé les 3 experts** et donné un **petit bonus** à la boîte « retouchée », la plus difficile.
5. À l'examen blanc : **0.996**, soit 4 erreurs sur 978.
