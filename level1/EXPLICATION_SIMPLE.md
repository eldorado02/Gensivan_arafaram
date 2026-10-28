# Level 1 — Explication très simple

Ce fichier est fait pour un **grand débutant**. Chaque mot technique est expliqué juste à côté, avec ce symbole : 📖.

---

## 1. Le jeu : vrai ou faux ?

On me donne des images. Pour chaque image, je dois dire :

- **0** = c'est une **vraie photo** (prise avec un appareil photo, pas modifiée) ;
- **1** = c'est une **fausse image** (fabriquée par une IA, ou une vraie photo modifiée).

📖 **IA** = *Intelligence Artificielle*. Un programme qui fait des tâches « intelligentes ». Ici, certaines IA savent **fabriquer** de fausses images très réalistes.

📖 **Classification binaire** = ranger chaque chose dans **une boîte parmi deux** (ici : vrai ou faux). *Binaire* veut dire « deux possibilités ».

Je ne regarde pas les images moi-même : j'apprends à un **programme** à le faire.

📖 **Machine learning** (en français *apprentissage automatique*) = au lieu d'écrire les règles à la main, on montre au programme **beaucoup d'exemples avec la bonne réponse**, et il trouve les règles tout seul. Comme un enfant qui apprend à reconnaître un chat en voyant beaucoup de chats.

📖 **Deep learning** (*apprentissage profond*) = une forme de machine learning qui utilise des **réseaux de neurones** avec beaucoup de couches. C'est ce que j'utilise.

📖 **Réseau de neurones** = un programme fait de milliers de petits calculs reliés entre eux, un peu inspiré du cerveau. Il contient des millions de nombres qu'il ajuste pendant l'apprentissage.

📖 **Modèle** = le programme qui a appris. « Entraîner un modèle » = le faire apprendre.

---

## 2. Les données (les exemples)

📖 **Données** (*data* en anglais) = les exemples qu'on donne au programme. Ici, ce sont des images.

📖 **Dataset** = l'ensemble de toutes les données.

On me donne 3 paquets d'images :

| Paquet | Nombre | Bonne réponse connue ? | À quoi il sert |
|---|---|---|---|
| **train** | 7 857 | ✅ oui | Pour **apprendre** |
| **val** | 978 | ✅ oui | Pour **vérifier** à la fin |
| **test** | 993 | ❌ non | C'est **l'examen** : je dois deviner, puis Kaggle me note |

📖 **train** = « entraînement ». 📖 **val** = « validation » (vérification). 📖 **test** = l'examen final.

📖 **Label** = la bonne réponse d'une image (0 ou 1). On dit aussi **target** (« cible »).

📖 **Kaggle** = un site web de compétitions de data science. On y écrit son code dans des **notebooks**, et Kaggle prête des ordinateurs puissants.

📖 **Notebook** = un fichier où l'on écrit du code par petits blocs (des **cellules**) qu'on lance un par un, avec le résultat affiché juste en dessous.

Dans le train : **1 image sur 3 est vraie, 2 sur 3 sont fausses**.

---

## 3. Comment je suis noté : le F1

Quand je me trompe, il y a deux sortes d'erreurs :

- **Fausse alerte** : l'image est vraie, mais j'ai dit « fausse ». 📖 On l'appelle **FP** = *False Positive* (faux positif).
- **Fausse image ratée** : l'image est fausse, mais j'ai dit « vraie ». 📖 On l'appelle **FN** = *False Negative* (faux négatif).

Et les bonnes réponses :
- 📖 **TP** = *True Positive* (vrai positif) : fausse image bien trouvée.
- 📖 **TN** = *True Negative* (vrai négatif) : vraie photo bien reconnue.

📖 **Positif** = la classe qu'on cherche (ici « fausse », le 1). **Négatif** = l'autre (« vraie », le 0).

📖 **Précision** = « quand je dis fausse, est-ce que j'ai raison ? »
📖 **Rappel** = « est-ce que je trouve toutes les fausses ? »
📖 **F1** (ou *F1 score*) = une note entre 0 et 1 qui mélange précision et rappel. **1 = parfait**.

📖 **Métrique** = la façon de calculer la note.

---

## 4. Le grand plan, comme une histoire

Imagine que je prépare un élève pour un examen.

1. **Je trie les exercices en double** pour qu'il ne puisse pas tricher (étape pHash).
2. **Je fais 3 groupes d'exercices** : il s'entraîne sur 2 groupes et je l'interroge sur le 3ᵉ (étape des folds).
3. **J'entraîne 3 élèves différents**, chacun avec sa méthode (les 3 experts).
4. **Je regarde lequel est le meilleur** et quelle règle de décision marche le mieux.
5. **Je fais un examen blanc une seule fois**, avec des exercices qu'il n'a jamais vus (le paquet val).
6. **Je lui fais réviser tout**, puis il passe le vrai examen (le paquet test).

---

## 5. Les étapes, une par une

### Étape 1 — Enlever la triche : les images en double

**Problème** : certaines images sont presque identiques (la même photo un peu recadrée, par exemple). Si une copie sert à apprendre et l'autre sert à vérifier, le programme **reconnaît l'image** au lieu de vraiment comprendre. Sa note est trop belle et fausse.

📖 **Fuite de données** (*data leakage*) = quand le programme voit, sans le vouloir, des informations qu'il ne devrait pas voir. C'est comme un élève qui a vu les réponses de l'examen.

**Solution :**

📖 **pHash** = *perceptual hash* (« empreinte visuelle »). Je calcule pour chaque image une **petite signature** de 64 cases qui valent 0 ou 1. Deux images qui se ressemblent ont presque la même signature. C'est comme une empreinte digitale de l'image.

📖 **Bit** = une case qui vaut 0 ou 1. 64 bits = 64 cases.

📖 **Distance de Hamming** = le nombre de cases différentes entre deux signatures. Si **4 cases ou moins** sont différentes → les deux images sont « jumelles ».

📖 **Union-Find** = une méthode pour **former des familles** : si A ressemble à B, et B ressemble à C, alors A, B et C sont dans la même famille.

Résultat : 6 933 familles pour 7 857 images.

### Étape 2 — Couper en 3 groupes : les folds

📖 **Fold** = un groupe d'images. Je coupe le train en **3 folds**.

📖 **Validation croisée** (*cross-validation*, ou *K-Fold*) = je fais 3 tours. À chaque tour, j'apprends sur 2 folds et je vérifie sur le 3ᵉ. Chaque fold sert une fois de vérification.

📖 **StratifiedGroupKFold** = le nom de l'outil qui fait ce découpage, avec 2 règles :
- *Stratified* (« équilibré ») : chaque fold garde la même proportion (1/3 vraies, 2/3 fausses) ;
- *Group* (« par famille ») : une famille d'images jumelles reste **toujours dans le même fold**. Plus de triche possible.

### Étape 3 — Les 3 experts (3 « élèves »)

📖 **Expert** = ici, un modèle. J'en entraîne 3, et chacun regarde l'image d'une façon différente.

📖 **Pré-entraîné** = un modèle qui a **déjà appris** sur des millions d'images (par une grande entreprise). Je ne pars pas de zéro : je réutilise ce qu'il sait.

📖 **Fine-tuning** (« réglage fin ») = je prends ce modèle pré-entraîné et je le fais **encore un peu apprendre** sur mes images à moi.

**Expert A — DeepGuard (`deep_rgb`)**
- C'est un modèle trouvé sur **Hugging Face** (📖 un site où les gens partagent des modèles d'IA), nommé `king1oo1/deepfake-model`.
- Il est basé sur **SigLIP2**. 📖 **SigLIP2** = un modèle de Google qui a appris en regardant des milliards d'images avec leur description en texte. Il **comprend le sens** d'une image (« c'est un chien sur une plage »). C'est de la même famille que 📖 **CLIP** (*Contrastive Language-Image Pre-training*, un modèle d'OpenAI qui associe images et textes).
- Il a déjà appris à reconnaître les **deepfakes**. 📖 **Deepfake** = une fausse image ou vidéo créée par IA.
- Il regarde l'image **en entier**, de façon **globale**, comme un humain.
- 📖 C'est un **Transformer** : un type de réseau qui découpe l'image en petits carrés et regarde **comment ils sont liés entre eux**.

**Expert B — ConvNeXt-V2 en couleurs normales (`cnn_rgb`)**
- 📖 **ConvNeXt-V2** = un modèle de Meta (Facebook), de type **CNN**.
- 📖 **CNN** = *Convolutional Neural Network* (réseau de neurones convolutif). Il fait **glisser de petits filtres** (comme une loupe) sur l'image pour trouver des motifs : bords, puis textures, puis formes. Très bon pour les **détails**.
- 📖 **RGB** = *Red Green Blue* (rouge, vert, bleu) : les 3 couleurs qui forment une image normale. « En RGB » = la photo normale.
- 📖 **Tiny** = la plus petite version du modèle, pour qu'elle tienne dans l'ordinateur de Kaggle.
- 📖 **384 px** = l'image est redimensionnée en 384 × 384 **pixels** (📖 un pixel = un petit point de couleur de l'image).

**Expert C — ConvNeXt-V2 en NPR (`cnn_npr`)** ⭐ le meilleur
- Le même type de modèle, mais je lui montre l'image **transformée en NPR**.
- 📖 **NPR** = *Neighboring Pixel Relationships* (« relations entre pixels voisins »). J'enlève le contenu de l'image et je ne garde que **les petites différences entre chaque pixel et ses voisins**.
- **Pourquoi** : les IA fabriquent les images en les **agrandissant plusieurs fois**. Ça laisse des **traces régulières** entre pixels voisins, invisibles à l'œil. Comme les traces d'un rouleau de peinture sur un mur.
- Il regarde l'image entière **et ses 4 coins**, pour mieux attraper une petite retouche cachée dans un coin.

### Comment un expert apprend (le vocabulaire de l'entraînement)

📖 **Entraîner** = montrer les images au modèle, regarder ses erreurs, et le corriger, encore et encore.

📖 **Epoch** = un tour complet : le modèle a vu **toutes** les images une fois. J'en fais 5 au maximum.

📖 **Batch** = un petit paquet d'images traitées ensemble (par exemple 4). L'ordinateur ne peut pas tout faire d'un coup.

📖 **Loss** (« perte ») = un nombre qui dit **à quel point le modèle se trompe**. Le but est de le faire descendre. J'utilise la **BCE** (📖 *Binary Cross-Entropy*, la loss classique quand il y a 2 réponses possibles).

📖 **Poids** (ou **paramètres**) = les millions de nombres à l'intérieur du modèle. Apprendre = ajuster ces nombres.

📖 **Optimiseur** = la méthode qui corrige les poids. J'utilise **AdamW** (📖 un optimiseur très courant).

📖 **Learning rate** (« vitesse d'apprentissage ») = la taille des corrections. Trop grand : le modèle fait n'importe quoi. Trop petit : il apprend trop lentement. J'en utilise **deux** : petit pour la partie déjà pré-entraînée, plus grand pour la nouvelle partie.

📖 **Early stopping** (« arrêt anticipé ») = si le modèle ne s'améliore plus pendant 2 epochs, j'arrête et je garde sa meilleure version.

📖 **Checkpoint** = une sauvegarde du modèle à un moment donné.

📖 **Augmentation** = je modifie un peu les images au hasard (je les retourne, je change un peu les couleurs, j'ajoute un léger flou) pour que le modèle ne s'habitue pas à des détails inutiles.

📖 **Overfitting** (« sur-apprentissage ») = le modèle **apprend par cœur** les images d'entraînement mais se trompe sur des images nouvelles. Comme un élève qui récite sans comprendre.

📖 **GPU** = *Graphics Processing Unit*, une carte graphique. Elle calcule beaucoup plus vite qu'un processeur normal pour l'IA. Kaggle me prête des GPU **T4** (un modèle de carte NVIDIA).

📖 **Seed** (« graine ») = un nombre fixé au départ pour que le hasard soit **toujours le même**. Si je relance le code, j'obtiens les mêmes résultats. 📖 C'est la **reproductibilité**.

### Étape 4 — Les réponses « honnêtes » : l'OOF

📖 **OOF** = *Out-Of-Fold* (« hors du groupe »). À chaque tour, le modèle répond sur le fold qu'il **n'a pas vu**. À la fin, **chaque image du train a une réponse donnée par un modèle qui ne la connaissait pas**. Ces réponses sont honnêtes.

📖 **Prédiction** = la réponse du modèle.

📖 **Probabilité** = un nombre entre 0 et 1. Le modèle ne dit pas directement « 0 ou 1 », il dit par exemple « **0.91** » = « je suis sûr à 91 % que c'est faux ».

### Étape 5 — Choisir le meilleur et la règle de décision

📖 **Seuil** = la limite pour passer de la probabilité à la réponse. Si probabilité ≥ seuil → 1 (faux), sinon → 0 (vrai).

J'ai testé 4 possibilités (📖 **candidats**) :
- chaque expert seul (3 candidats) ;
- un **mélange des 3** : 📖 **stacking** = un petit modèle simple qui apprend **comment mélanger** les réponses des 3 experts. J'utilise une 📖 **régression logistique** (un modèle mathématique très simple qui donne une probabilité).

Le gagnant est **`cnn_npr` seul**, avec un **seuil de 0.4425**.

📖 **Médiane** = la valeur du milieu quand on range des nombres. J'ai trouvé un meilleur seuil sur 5 morceaux du train, et j'ai pris celui du milieu. C'est plus stable qu'un seul seuil qui pourrait être de la chance.

### Étape 6 — L'examen blanc : le paquet val

📖 **Holdout** (« mis de côté ») = des données que je **n'ouvre qu'une seule fois**, à la fin, pour avoir une note honnête. Ici, c'est le paquet val.

Résultat : **F1 = 0.998**. Seulement **2 erreurs sur 978 images** (2 vraies photos prises pour des fausses, et **aucune** fausse image ratée).

📖 **Matrice de confusion** = un tableau qui montre les bonnes réponses et les erreurs :

```
                    j'ai dit "vraie"   j'ai dit "fausse"
vraiment vraie            324                 2
vraiment fausse             0               652
```

### Étape 7 — Réviser tout, puis le vrai examen

- Je réentraîne le modèle sur **train + val** (plus d'exemples = un peu meilleur).
- Je garde **le même seuil** (je ne triche pas).
- Il répond sur le paquet test.
- J'écris le fichier 📖 **`submission.csv`** (« soumission ») : un tableau avec le nom de chaque image et ma réponse (0 ou 1). 📖 **CSV** = *Comma-Separated Values*, un tableau en texte où les colonnes sont séparées par des virgules.
- J'écris aussi un 📖 **audit** en **JSON** (📖 *JavaScript Object Notation*, un format de texte pour ranger des informations). Il garde toutes les notes et tous les réglages : la preuve de ce que j'ai fait.

---

## 6. À retenir en 5 phrases

1. Je dois dire si une image est **vraie (0) ou fausse (1)**.
2. J'ai **enlevé la triche** en mettant les images jumelles dans la même famille.
3. J'ai entraîné **3 experts** : un qui regarde le sens global, un qui regarde les textures, un qui regarde les **traces entre pixels (NPR)**.
4. J'ai choisi le meilleur (**NPR**) et le seuil avec des réponses **honnêtes (OOF)**.
5. À l'examen blanc, j'ai **0.998**, soit 2 erreurs sur 978.
