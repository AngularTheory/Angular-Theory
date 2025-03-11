Voici la présentation révisée sans les étoiles et avec un formatage plus sobre :

---

# ∆ngular Theory 0.0 - Interface Interactive  
➤ Version 5.0 | Simulation avancée en physique théorique  

## Présentation  
∆ngular Theory 0.0 est un logiciel scientifique permettant :  
- L’analyse de l’équation pivot en temps réel  
- La comparaison avec des données expérimentales (neutrinos, JWST, Euclid)  
- Un mode Monte Carlo pour explorer la sensibilité des paramètres  
- Des calculs optimisés avec Numba et une exécution parallèle avec Dask  

## Installation  

### Prérequis  
Assurez-vous d’avoir Python 3.8+ installé. Ensuite, installez les dépendances avec :  
```bash
pip install -r requirements.txt
```

### Lancer l’interface  
Une fois installé, exécutez :  
```bash
python angular_theory.py
```

### Tester l’installation (optionnel, recommandé)  
Pour vérifier que tout fonctionne :  
```bash
python angular_theory.py --verify
```

## Structure du dépôt  
```
angular-theory-interface/
│── angular_theory.py          # Interface principale
│── tests/                     # Tests unitaires
│   ├── test_pivot.py          # Vérification de l’équation pivot
│   ├── test_data.py           # Vérification du chargement des données
│── docs/                      # Documentation
│   ├── README.md              # Présentation du projet
│   ├── INSTALL.md             # Guide d'installation détaillé
│   ├── API_REFERENCE.md       # Explication des classes et modules
│── requirements.txt           # Liste des dépendances
│── LICENSE                    # Licence d'utilisation
│── .gitignore                 # Fichiers à exclure de Git
```

## Dépannage et Problèmes courants  

### Python introuvable  
Si `python` ou `python3` ne fonctionne pas, vérifiez :  
```bash
python --version
python3 --version
```
Si Python n'est pas installé, téléchargez-le depuis [python.org](https://www.python.org).

### Erreur d’installation des dépendances  
Si `pip install -r requirements.txt` échoue :  
```bash
python -m ensurepip --default-pip
pip install --upgrade pip
pip install nom_du_module
```

### Erreur à l’exécution de angular_theory.py  
Si le programme plante, assurez-vous d’être dans le bon dossier :  
```bash
cd chemin/vers/le/dossier
python angular_theory.py
```

### Interface graphique ne s’affiche pas  
Si PyQt6 ne fonctionne pas, essayez :  
```bash
pip install PyQt6
```
Puis relancez le programme.

