# iSeg — baseline 2.5D

Le projet segmente chaque pixel d'une coupe IRM en quatre classes : fond, LCR,
substance grise et substance blanche. L'entrée 2.5D contient six canaux : trois
coupes T1 voisines et trois coupes T2 voisines.

## Structure

- `src/data/` : lecture des paires Analyze `.hdr/.img`, normalisation, labels et
  construction des six canaux ;
- `src/models/` : U-Net 2.5D légère ;
- `src/training/` : Cross-Entropy et une étape d'apprentissage testable ;
- `scripts/smoke_train.py` : vérifie le mécanisme sur une tâche artificielle ;
- `scripts/train_baseline.py` : entraîne sur les vraies IRM ;
- `tests/` : vérifications de forme, données, loss et gradients.

## Installation

```bash
.venv/bin/pip install -r requirements.txt
```

## Vérifier avant d'entraîner

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/smoke_train.py --steps 40
```

Le smoke test vérifie seulement que les poids sont modifiés et que la loss
diminue. Il ne mesure aucune qualité médicale.

## Entraîner la baseline réelle

```bash
.venv/bin/python scripts/train_baseline.py \
  --data /Users/foqker/Downloads/iSeg-2017-Training \
  --epochs 10
```

Le script entraîne sur les sujets 1-8, mesure la Cross-Entropy sur les patients
9-10 et sauvegarde le meilleur checkpoint dans `outputs/baseline_2p5d.pt`.

Pour vérifier rapidement les vraies données avant un entraînement complet :

```bash
.venv/bin/python scripts/train_baseline.py \
  --data /Users/foqker/Downloads/iSeg-2017-Training \
  --epochs 1 --max-train-batches 1 --max-validation-batches 1
```

Le prochain script doit charger ce checkpoint, reconstruire les prédictions des
volumes 9 et 10 et calculer le Dice par tissu et par patient entier. Les sujets
11-23 n'ont pas de labels : ils ne peuvent pas servir à calculer un Dice.
