# Serveur Piiranha - Service de détection et masquage de PII

![Logo PII-Ranha](logo.png)

Un service basé sur FastAPI pour détecter et masquer les informations personnelles (PII) dans du texte en utilisant le modèle PII-Ranha.

## Fonctionnalités

- Détection de PII : Vérifie si un texte contient des informations personnelles
- Masquage de PII : Masque les PII détectées avec un marquage agrégé ou détaillé
- Vérification de santé : Vérifie l'état du service
- Découpage en tokens : Gère les longs textes en les divisant en segments de 256 tokens

## Exemples

```bash
% curl -X POST "http://localhost:8000/check-pii" -i  \
-H "Content-Type: application/json" \
-d '{"text":"Votre texte ici, je m'appelle Jean-Claude Dusse"}'
HTTP/1.1 400 Bad Request
date: Thu, 16 Jan 2025 11:40:49 GMT
server: uvicorn
content-length: 34
content-type: application/json

{"detail":"PII détecté dans l'entrée"
```
```bash
% curl -X POST "http://localhost:8001/check-pii" -i \
-H "Content-Type: application/json" \
-d '{"text":"Votre texte ici, Lorem ipsum dolor sit amet, consectetur adipiscing elit"}'
HTTP/1.1 200 OK
date: Thu, 16 Jan 2025 11:40:07 GMT
server: uvicorn
content-length: 43
content-type: application/json

{"status":"OK","message":"Aucun PII détecté"
```

## Points d'accès API

### Vérifier PII
`POST /check-pii`
- Corps de la requête : `{"text": "votre texte ici"}`
- Retourne : 200 si aucun PII détecté, 400 si PII détecté

### Masquer PII
`POST /mask-pii`
- Corps de la requête : `{"text": "votre texte ici"}`
- Paramètre optionnel : `aggregate_redaction=true|false` (par défaut : true)
- Retourne : Texte original et texte masqué

### Vérification de santé
`GET /health`
- Retourne : État du service

## Exécution du service

1. Installer les dépendances :
```bash
pip install fastapi uvicorn transformers torch
```

2. Lancer le serveur :
```bash
python app.py
```

3. Le service sera disponible à l'adresse `http://localhost:8000`

## Test avec curl

Vérifier la présence de PII :
```bash
curl -X POST "http://localhost:8000/check-pii" -H "Content-Type: application/json" -d '{"text":"Votre texte ici"}'
```

Masquer les PII :
```bash
curl -X POST "http://localhost:8000/mask-pii" -H "Content-Type: application/json" -d '{"text":"Votre texte ici"}'
```

## Développement

Le service utilise le modèle [PII-Ranha](https://huggingface.co/iiiorg/piiranha-v1-detect-personal-information) de Hugging Face.

## Auteur & Support Entreprise

**Auteur** : Sébastien Campion

Sébastien Campion - sebastien.campion@foss4.eu

**Note** : Ce projet est en développement actif. Merci de signaler tout problème ou demande de fonctionnalité via le système de suivi des issues.

## Licence

Ce projet est sous licence GNU Affero General Public License v3.0 - voir le fichier [LICENSE](LICENSE) pour plus de détails.
