# Mandia Voix 9M — v0.1 expérimentale

Modèle de synthèse vocale française de **9 368 355 paramètres**, développé par Mandia à partir d'initialisations aléatoires. Cette version est notre référence de prononciation française. L'objectif est une voix compacte, naturelle, avec un accent ivoirien ; **le naturel et l'accent ivoirien de cette version restent à valider**. Du grain et un effet d'écho sont encore signalés à l'écoute.

- [Poids et sources de la version](https://github.com/mandi-model/mandia-voix/releases/tag/v0.1.0)
- [Fiche modèle Mandia](https://dataset.mandia.org/gnakalehacker/mandia-voix-9m-v0-1)
- [Corpus IvoireTTS](https://dataset.mandia.org/gnakalehacker/ivoiretts-v0-1)

## Essayer

Python 3.10+ conseillé. La vérification de publication utilise Python 3.9 et PyTorch 2.8 CPU ; l'expérience d'origine utilise PyTorch 2.11 et une L4. Installer **eSpeak NG 1.51** pour la transcription phonétique française (aucun audio eSpeak n'est employé).

```bash
git clone https://github.com/mandi-model/mandia-voix.git
cd mandia-voix
python -m pip install -r requirements.txt
# Debian / Ubuntu :
sudo apt-get install espeak-ng
python download_weights.py --output weights
python inference.py --weights weights --text "Bonjour madame, est-ce que cette place est encore libre ?" --output voice.wav
```

Le script vérifie les empreintes des poids avant de les charger avec `weights_only=True`. Il produit un WAV mono 24 kHz. L'inférence fonctionne sur CPU ; CUDA peut être choisi avec `--device cuda`. Les résultats peuvent varier légèrement entre matériels, versions et graines. Utiliser des phrases courtes ; les entrées trop longues sont refusées plutôt que tronquées. Cette version ne contient pas de service web d'inférence.

## Architecture et version sélectionnée

| Élément | Paramètres |
| --- | ---: |
| Modèle acoustique | 8 423 841 |
| Vocodeur iSTFT compact | 944 514 |
| Total de synthèse | 9 368 355 |

L'encodeur utilise des blocs convolutifs résiduels avec attention. L'entraînement acoustique emploie un alignement monotone et le flow matching conditionnel. Le vocodeur prédit magnitude et phase puis reconstruit la forme d'onde par iSTFT. Les discriminateurs employés uniquement à l'entraînement ne sont pas comptés dans l'inférence. Il s'agit d'une implémentation de principes publiés ; aucune supériorité ni nouveauté architecturale n'est revendiquée.

Le modèle acoustique retenu est le checkpoint français SIWIS à 18 000 mises à jour. Le vocodeur compact est celui sélectionné à 3 000 mises à jour de l'essai naturalness-width-v01. L'inférence utilise **Euler 32 étapes**, une amplitude initiale de bruit de **0,5** et **speaker 0**. L'adaptation conditionnelle ivoirienne rejetée à l'écoute est désactivée et ses poids ne font pas partie de cette version.

Les fichiers acoustic.pt et vocoder.pt sont des exports des seuls poids et configurations nécessaires à l'inférence, sans états des optimiseurs. Les SHA-256 des checkpoints d'origine et des exports sont consignés dans WEIGHTS.json. Les tenseurs sont vérifiés identiques lors de l'export ; les checkpoints complets de reprise restent dans Drive.

## Données et apprentissage

L'acoustique française sélectionnée a été entraînée sur 9 517 extraits SIWIS, environ 10,46 h. Le miroir 24 kHz utilisé est Rcarvalo/siwis-fr-24k, révision c77b462ea80fc874bec2dcde330036f28357683a ; c'est un dérivé non officiel, sans équivalence binaire revendiquée avec l'archive originale.

Le vocodeur provient d'expériences mixtes SIWIS et IvoireTTS ; son dernier affinage utilise 5 735 extraits SIWIS et 40 extraits ivoiriens longs, échantillonnage 90/10. Le modèle publié n'est donc pas présenté comme un entraînement acoustique ivoirien réussi sur les 1 000 extraits du catalogue. Les essais d'adaptation antérieurs ont parfois fortement dégradé la compréhension.

Les transcriptions ivoiriennes sont automatiques et ne sont pas intégralement relues. Les partitions de ce corpus sont distinctes par vidéo source ; l'indépendance des locuteurs n'est pas établie. Le dépôt ne redistribue pas les enregistrements d'entraînement.

## Évaluation et limites

L'amplitude 0,5 a été choisie sur 8 phrases de développement, avant de générer 8 nouvelles phrases avec deux graines chacune. Sur ces nouvelles phrases, le reconnaisseur automatique a compté **10 erreurs de mots sur 166 mots évalués (6,02 %)**, contre **15/166 (9,04 %)** pour l'amplitude 1, avec les mêmes poids. Les 166 mots comptent les deux graines : il y a bien 8 textes distincts.

Ce petit essai mesure une transcription automatique, avec Whisper large-v3-turbo utilisé comme outil d'évaluation. Il ne suffit pas à établir le naturel, la qualité de l'accent, une robustesse générale ou une comparaison avec des systèmes externes. Le lot n'a pas servi à sélectionner l'amplitude. D'autres séries plus anciennes ont servi au développement et ne sont pas un test indépendant.

Le retour d'écoute le plus récent signale une meilleure compréhension sur l'extrait écouté, avec du grain ou de l'écho encore présent. La phrase et la graine exactes de ce retour n'ont pas été précisées. Il ne s'agit pas d'une évaluation MOS ni d'une validation de toutes les phrases.

## Fichiers et reproduction

- src/ : modules figés utilisés pour l'inférence.
- inference.py et download_weights.py : exécution autonome et téléchargement vérifié.
- normalization.json : statistiques ajustées sur SIWIS entraînement.
- WEIGHTS.json et RELEASE-VERIFICATION.json : traçabilité et vérifications de l'export.
- experiments/ : sources des étapes acoustique, vocodeur, nettoyage spectral et calibration ; chemins de données à configurer pour reproduire l'entraînement.
- reports/ : résultats et protocoles des expériences sélectionnées.

Les scripts historiques peuvent contenir les chemins Drive de l'expérience. Leur exécution nécessite les corpus et les étapes préalables documentés ; ce dépôt n'annonce pas un entraînement complet reproductible depuis les seuls fichiers de poids. Aucun notebook Colab n'est imposé pour l'inférence.

## Crédits et usages

SIWIS : Junichi Yamagishi, Pierre-Edouard Honnet, Philip Garner et Alexandros Lazaridis (2017), *The SIWIS French Speech Synthesis Database, 2016*, Université d'Édimbourg. [DOI 10.7488/ds/1705](https://doi.org/10.7488/ds/1705). Licence de données CC BY 4.0 selon le README de la source. Le miroir utilisé et la préparation audio sont des modifications distinctes des données originales.

Principes scientifiques : [Matcha-TTS](https://arxiv.org/abs/2309.03199), [Vocos](https://arxiv.org/abs/2306.00814), [HiFi-GAN](https://arxiv.org/abs/2010.05646), [APNet](https://arxiv.org/abs/2305.07952). Ces références ne constituent pas une affirmation de reproduction exacte.

Les conditions du code, des poids et des données tierces sont distinctes. Cette publication est étiquetée **usage restreint** ; voir NOTICE.txt. La licence de SIWIS ne s'applique pas automatiquement aux autres enregistrements. Le dépôt public ne vaut pas autorisation d'usurper l'identité d'un locuteur.

