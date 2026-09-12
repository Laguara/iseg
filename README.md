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
- `scripts/evaluate_baseline.py` : calcule le Dice entier de chaque patient ;
- `scripts/visualize_prediction.py` : crée une figure IRM / label / prédiction ;
- `scripts/predict_test.py` : produit les volumes prédits des sujets 11-23 ;
- `tests/` : vérifications de forme, données, loss et gradients.

Le déroulé et les choix du projet sont expliqués dans [RAPPORT.md](RAPPORT.md).

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
volumes 9 et 10 et calculer le Dice par tissu et par patient entier.

## Evaluer le checkpoint

```bash
.venv/bin/python scripts/evaluate_baseline.py \
  --data /Users/foqker/Downloads/iSeg-2017-Training \
  --checkpoint outputs/baseline_2p5d.pt
```

Le script imprime et écrit le Dice fond/LCR/SG/SB de chaque patient 9 et 10,
puis leur moyenne au premier plan. Les sujets 11-23 n'ont pas de labels : ils
ne peuvent pas servir à calculer un Dice.

## Créer une figure pour le rapport

```bash
.venv/bin/python scripts/visualize_prediction.py \
  --data /Users/foqker/Downloads/iSeg-2017-Training \
  --checkpoint outputs/baseline_2p5d.pt \
  --subject 9 --slice 128
```

La figure est enregistrée dans `outputs/validation_example.png`.

## Prédire les sujets test sans labels

```bash
.venv/bin/python scripts/predict_test.py \
  --data /Users/foqker/Downloads/iSeg-2017-Testing \
  --checkpoint outputs/baseline_2p5d.pt
```

Le script écrit une prédiction `.npy` interne (classes 0 à 3) et une paire
Analyze `.hdr/.img` avec les valeurs iSeg 0/10/150/250. Aucun Dice n'est
possible sur ces sujets, car le jeu de test ne contient pas les labels.
