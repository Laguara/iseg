# Rapport — segmentation iSeg 2.5D

## Objectif

Segmenter chaque pixel d'une IRM cérébrale de nourrisson en quatre classes :
fond, LCR, substance grise (SG) ou substance blanche (SB). Le choix est une
baseline légère 2.5D : le réseau reste 2D, mais reçoit le contexte de trois
coupes voisines dans les deux modalités T1 et T2.

## Données et séparation

Chaque sujet d'entraînement possède trois volumes Analyze : T1, T2 et label.
Le label original est converti avant entraînement :

| Valeur iSeg | Classe interne | Tissu |
|---:|---:|---|
| 0 | 0 | Fond |
| 10 | 1 | LCR |
| 150 | 2 | SG |
| 250 | 3 | SB |

La séparation est faite par patient, jamais par coupe :

- entraînement : sujets 1 à 8 ;
- validation : sujets 9 et 10 ;
- test : sujets 11 à 23, sans labels.

Cette séparation évite de voir des coupes d'un même cerveau à la fois pendant
l'apprentissage et pendant l'évaluation.

## Entrée et modèle

Pour prédire la coupe `z`, le modèle reçoit six cartes :

```text
[T1(z-1), T1(z), T1(z+1), T2(z-1), T2(z), T2(z+1)]
```

Le modèle est un petit U-Net : l'encodeur réduit la résolution pour comprendre
le contexte global ; le décodeur la restaure ; les *skip connections* rendent
les contours précis. La sortie contient quatre logits par pixel. Le modèle
contient 68 980 paramètres entraînables.

## Apprentissage

Chaque batch suit toujours le même ordre :

```text
images -> modèle -> logits -> Cross-Entropy avec labels
       -> backward (gradients) -> Adam (modification des poids)
```

La Cross-Entropy est volontairement la loss initiale la plus simple. Le
checkpoint retenu est celui avec la plus faible loss sur les patients 9 et 10.

## Évaluation

Le Dice est calculé par tissu et sur le volume entier de chaque patient :

```text
Dice = 2 × intersection(prediction, label) / (taille prediction + taille label)
```

Il ne faut pas annoncer un Dice calculé sur seulement quelques patches ou
coupes comme une performance du projet entier.

## Résultats actuels

Le code a été vérifié : tests, smoke test et une batch réelle passent. Le
checkpoint produit après une seule batch technique donne un Dice très faible ;
ce résultat est attendu et ne doit pas être présenté. Les résultats finaux sont
à renseigner seulement après un entraînement complet avec la commande ci-dessous.

```bash
.venv/bin/python scripts/train_baseline.py \
  --data /Users/foqker/Downloads/iSeg-2017-Training \
  --epochs 10

.venv/bin/python scripts/evaluate_baseline.py \
  --data /Users/foqker/Downloads/iSeg-2017-Training \
  --checkpoint outputs/baseline_2p5d.pt
```

Copier ensuite dans ce tableau les valeurs de `outputs/validation_dice.json` :

| Patient | Dice LCR | Dice SG | Dice SB | Moyenne premier plan |
|---:|---:|---:|---:|---:|
| 9 | à remplir | à remplir | à remplir | à remplir |
| 10 | à remplir | à remplir | à remplir | à remplir |

## Limites et améliorations possibles

- La validation comporte seulement deux patients : les résultats sont donc
  fragiles.
- Le modèle est volontairement léger et privilégie la vitesse à la capacité.
- Il faudra comparer une seule amélioration à la fois : plus de canaux,
  augmentation cohérente image+label, ou autre loss. Ne pas tout changer en
  même temps.
- Les sujets 11 à 23 peuvent recevoir des prédictions, mais sans labels aucun
  Dice ne peut y être calculé.

Les prédictions test peuvent être créées avec :

```bash
.venv/bin/python scripts/predict_test.py \
  --data /Users/foqker/Downloads/iSeg-2017-Testing \
  --checkpoint outputs/baseline_2p5d.pt
```
