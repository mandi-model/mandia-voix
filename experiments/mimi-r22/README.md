# Mandia Voix — texte vers Mimi, pilote r22

**Résultat expérimental publié pour inspection. N reste la référence : cette version n'améliore pas suffisamment la prononciation et n'est pas validée comme voix ivoirienne aboutie.**

Cette publication clôt deux essais du 23 octobre 2026 sans nouvel entraînement. Elle contient notre prédicteur texte→codes Mimi8, deux exports d'inférence au format safetensors, une commande d'inférence autonome, la recette d'apprentissage et les résultats agrégés. Elle n'exige pas Colab. Aucun serveur de synthèse n'est connecté à la fiche Mandia.

## Taille et provenance

Le prédicteur compte **17 878 995 paramètres**, dont **14 421 891 entraînables** pendant les essais. Il réutilise l'encodeur lexical figé de N et initialise ses embeddings à partir des codebooks Mimi. Six couches temporelles de largeur320 et un décodeur récurrent de profondeur prédisent huit codebooks. Le codec **Mimi préentraîné et figé est nécessaire en plus** ; sa taille n'est pas comprise dans les17,9M. Son fichier stocke environ96,15M valeurs (paramètres et buffers). Aucun poids PersonaPlex7B n'est utilisé.

## Résultats réellement obtenus

R22 : quatre textes fixés avant continuation, 99 mots de référence, sans audio cible ni durée réelle fournis à la génération.

| Sortie | Écarts Whisper Turbo /99 mots | Écarts Whisper Medium /99 mots |
|---|---:|---:|
| N, référence conservée | 7 | 15 |
| Mimi français, glouton | 39 | 51 |
| Mimi adaptation ivoirienne, glouton | 74 | 83 |
| Mimi adaptation ivoirienne, échantillonnage | 81 | 93 |

Ces écarts comptent substitutions, suppressions et insertions. Les deux ASR peuvent partager des erreurs ; ils ne mesurent ni grains, ni écho, ni accent. Les sorties sont conservées sans sélectionner seulement les meilleures. Deux textes connus servent de contrôles de mémorisation et sont exclus du tableau. Le premier essai v21 utilise quatre autres textes : ses scores ne constituent pas une comparaison directe de progrès avec r22.

## Données et apprentissage

4 096 textes français distincts, environ4,55h, et49 extraits ivoiriens stricts, environ5,2min provenant d'une seule source. Validation séparée :32FR et9CI, avec limites d'une validation temporelle pour CI. Aucun entraînement sur l'enregistrement personnel ni sur le corpus de la chaîne du propriétaire. Les transcriptions et alignements sont automatiques ; pas de validation intégrale par locuteur humain revendiquée.

- V21 :3 000 pas français, puis2 000 mixtes ;12 checkpoints.
- R22 :10 000 pas français supplémentaires dans une nouvelle continuation depuis l'export français v21, puis1 000 mixtes à taux réduit avec initialisation de la voix CI depuis la voix FR apprise ;14 checkpoints.
- AdamW et autocastBF16. Lots français8, mixtes6FR+2CI. Sélection sur perte de validation uniquement ; aucun choix sur les quatre textes nouveaux ou leur ASR.
- Export retenu FR : étape9 000 de continuation ; CI : étape1 000. Les26 checkpoints complets (optimiseurs et états aléatoires inclus) restent dans le Drive de recherche. Les fichiers distribués ici sont uniquement les tenseurs d'inférence, sans chemins privés de Drive.

## Essayer

Python3.10+ recommandé, GPU conseillé. Installer eSpeak NG via le gestionnaire de paquets du système, puis :

```sh
python -m pip install -r requirements.txt
python inference.py --text "Bonjour chef, le colis est arrivé ce matin." --voice fr --output essai-fr.wav
python inference.py --text "Tantie, garde le panier près de la porte." --voice ci --output essai-ci.wav
```

Lancer ces commandes depuis ce dossier. Les exports sont récupérés depuis la préversion GitHub référencée dans `WEIGHTS.json` et vérifiés parSHA256. Mimi est chargé depuis Kyutai à une révision épinglée. `--cache` et `--output` permettent de placer modèles et résultats dans un Drive monté ou un volume externe. La génération refuse un texte dépassant sa durée maximale prédite, au lieu de tronquer silencieusement la parole.

`--voice ci` est une **adaptation de recherche moins intelligible que N** dans ce test, pas une promesse d'accent. Ni la naturalité, ni l'absence de grains, ni une généralisation à de nouveaux locuteurs ne sont validées. Une bonne reconstruction d'audio réel avec Mimi ne suffit pas à apprendre à lire du texte.

Voir `NOTICE.txt` pour les conditions et attributions ; les données sources ne sont pas redistribuées. La version9M précédente n'est pas remplacée.
