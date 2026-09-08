# iSeg — première segmentation 2.5D

Le premier objectif n'est pas d'obtenir tout de suite un bon score : c'est de
faire fonctionner et comprendre une boucle d'apprentissage complète sur les
vraies IRM iSeg.

Le fichier à lire est [train_simple.py](train_simple.py). Il contient, dans cet
ordre :

1. le petit U-Net 2.5D ;
2. le chargement des vraies coupes avec le Dataset ;
3. le forward ;
4. la Cross-Entropy ;
5. le calcul des gradients ;
6. la mise à jour des poids par Adam.

Le dossier `src/data/` est séparé car il est responsable uniquement de la
lecture fiable des fichiers iSeg (`.hdr` + `.img`) et de la construction des
six canaux 2.5D. Le script l'utilise directement.

## Installer et lancer le premier essai

```bash
.venv/bin/pip install -r requirements.txt
.venv/bin/python train_simple.py \
  --data /Users/foqker/Downloads/iSeg-2017-Training
```

Par défaut, le script utilise le sujet 1 et cinq batchs de deux vraies coupes.
C'est volontairement petit : cela valide le mécanisme sans attendre longtemps.

Pour parcourir toutes les coupes du sujet 1 :

```bash
.venv/bin/python train_simple.py \
  --data /Users/foqker/Downloads/iSeg-2017-Training \
  --max-batches 0
```

Plus tard, après avoir compris ce premier résultat, on pourra entraîner sur les
sujets 1 à 8 et réserver 9 et 10 pour la validation. Ne pas annoncer de score
scientifique avant cette séparation patient par patient.
