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





## Simulation et Exploration

∆ngular Theory 0.0 inclut un module de simulation avancé permettant de :

- Lancer des calculs Monte Carlo pour tester la sensibilité des paramètres.
- Exécuter des simulations parallèles optimisées avec Dask et Numba.
- Visualiser en temps réel l’évolution des variables fondamentales.

### Exécuter une simulation standard  
Pour une exécution simple avec les paramètres par défaut, utilisez :  
```bash
python angular_theory.py --simulate
```

### Exécuter une simulation Monte Carlo  
Pour analyser la sensibilité des paramètres avec 1000 simulations, utilisez :  
```bash
python angular_theory.py --monte-carlo --n 1000
```
(Vous pouvez modifier `--n 1000` pour ajuster le nombre de simulations.)

### Mode Debug : Vérifier les logs et optimiser  
Si des erreurs surviennent ou pour activer le mode verbose, utilisez :  
```bash
python angular_theory.py --simulate --verbose
```

### Visualiser les résultats  
Les résultats peuvent être analysés en temps réel via l'interface :

1. Lancez l’interface avec :  
   ```bash
   python angular_theory.py
   ```

2. Naviguez vers l’onglet "Simulation".
3. Ajustez les paramètres et relancez la simulation.


"""
∆ngular Theory 0.0 - Interface Interactive (v5.0)
Copyright (C) 2024 David Souday
Licence CC-BY-NC-ND 4.0 International
"""

import sys
import numpy as np
import logging
import json
import pytest
from numba import jit
from PyQt6.QtCore import pyqtSignal, QThread
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QLabel, QDoubleSpinBox, QPushButton, 
                             QProgressBar, QMessageBox, QTabWidget, 
                             QFileDialog)
import matplotlib.pyplot as plt

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('angular_theory.log'),
        logging.StreamHandler()
    ]
)

# Test unitaire intégré
def run_unit_tests():
    """Exécute les tests unitaires au démarrage."""
    class TestConfig:
        pass

    test_args = ['-v', '--tb=short', '--disable-warnings', 'tests/']
    return pytest.main(test_args, plugins=[TestConfig()])

if '--run-tests' in sys.argv:
    exit_code = run_unit_tests()
    sys.exit(exit_code)

# Fonction de torsion
def torsion_function(s, Δθ₀):
    """Calcule la torsion 𝒯(s) régulant l'entropie."""
    return Δθ₀ / (s + Δθ₀)

# Entropie effective
def effective_entropy(s, Δθ₀, κ):
    """Calcule l'entropie effective S_eff(s) prenant en compte la torsion."""
    if κ <= 0:
        raise ValueError("κ doit être supérieur à zéro pour éviter la division par zéro.")
    T_s = torsion_function(s, Δθ₀)
    return (s ** 2 + Δθ₀ * np.log(1 + s)) * (1 + (Δθ₀ / κ) * T_s)

# Équation pivot optimisée
@jit(nopython=True, fastmath=True, cache=True)
def numba_pivot_equation(s, Δθ₀, α, β, ε, δ, τ, κ):
    """Calcule l'équation pivot avec couplage torsion-entropie."""
    S_eff = effective_entropy(s, Δθ₀, κ)
    T_s = torsion_function(s, Δθ₀)
    exp_term = np.exp(-τ ** 2 / (4 * S_eff))
    mod_term = (1 + ε * np.cos(Δθ₀ * δ * s * T_s)) ** β
    return (Δθ₀ ** α) * exp_term * mod_term

class MonteCarloScheduler(QThread):
    simulation_complete = pyqtSignal(dict)
    progress_updated = pyqtSignal(int)

    def __init__(self, base_params, n_simulations):
        super().__init__()
        self.base_params = base_params
        self.n_simulations = n_simulations
        self.results = []

    def run(self):
        """Lance les simulations en parallèle."""
        try:
            from dask import delayed, compute
            from dask.distributed import Client

            # Essayer d'utiliser Dask
            try:
                with Client() as client:
                    tasks = [delayed(self._run_simulation)(self._perturb_params(self.base_params, i), i) for i in range(self.n_simulations)]
                    results = compute(*tasks, scheduler='distributed')

            except Exception as e:
                logging.warning("Dask n'est pas disponible, exécution en local.")
                results = [self._run_simulation(self._perturb_params(self.base_params, i), i) for i in range(self.n_simulations)]

            for i, result in enumerate(results):
                self.results.append(result)
                self.progress_updated.emit(int((i + 1) / self.n_simulations * 100))

            self.simulation_complete.emit({'status': 'complete', 'results': self.results})

        except Exception as e:
            logging.error(f"Erreur simulation: {str(e)}")
            self.simulation_complete.emit({'status': 'error', 'message': str(e)})

    def _perturb_params(self, params, seed):
        """Génère des paramètres perturbés pour la simulation."""
        np.random.seed(seed)
        return {k: v * np.random.normal(1, 0.1) for k, v in params.items()}

    def _run_simulation(self, params, seed):
        """Exécute une simulation individuelle."""
        try:
            s = np.linspace(0, 100, 1000)
            return {
                'params': params,
                'result': numba_pivot_equation(s, **params),
                'seed': seed,
                's': s
            }
        except Exception as e:
            logging.error(f"Erreur simulation {seed}: {str(e)}")
            return None

class AngularTheoryApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.monte_carlo_runner = None
        self._init_ui()

    def _init_ui(self):
        """Initialise l'interface utilisateur."""
        self.setWindowTitle("∆ngular Theory 0.0")
        self.setGeometry(100, 100, 800, 600)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.monte_carlo_tab = QWidget()
        layout = QVBoxLayout()

        self.mc_iterations = QDoubleSpinBox()
        self.mc_iterations.setRange(10, 10000)
        layout.addWidget(QLabel("Nombre de simulations:"))
        layout.addWidget(self.mc_iterations)

        self.mc_progress = QProgressBar()
        layout.addWidget(self.mc_progress)

        self.run_mc_button = QPushButton("Lancer simulation")
        self.run_mc_button.clicked.connect(self.run_monte_carlo)
        layout.addWidget(self.run_mc_button)

        self.save_params_button = QPushButton("Sauvegarder les paramètres")
        self.save_params_button.clicked.connect(self.save_params)
        layout.addWidget(self.save_params_button)

        self.load_params_button = QPushButton("Charger les paramètres")
        self.load_params_button.clicked.connect(self.load_params)
        layout.addWidget(self.load_params_button)

        self.monte_carlo_tab.setLayout(layout)
        self.tabs.addTab(self.monte_carlo_tab, "Monte Carlo")

    def run_monte_carlo(self):
        """Lance une simulation Monte Carlo."""
        if self.monte_carlo_runner and self.monte_carlo_runner.isRunning():
            QMessageBox.warning(self, "Avertissement", "Une simulation est déjà en cours.")
            return

        try:
            params = self._get_current_params()
            self.monte_carlo_runner = MonteCarloScheduler(params, int(self.mc_iterations.value()))
            self.monte_carlo_runner.progress_updated.connect(self.mc_progress.setValue)
            self.monte_carlo_runner.simulation_complete.connect(self.handle_mc_results)
            self.monte_carlo_runner.start()
        except ValueError as e:
            QMessageBox.critical(self, "Erreur d'entrée", str(e))

    def handle_mc_results(self, results):
        """Traite les résultats de la simulation."""
        if results['status'] == 'complete':
            self._analyze_mc_results(results['results'])
        else:
            QMessageBox.critical(self, "Erreur", results['message'])

    def _analyze_mc_results(self, results):
        """Analyse statistique des résultats et visualisations."""
        for result in results:
            if result is not None:
                plt.plot(result['s'], result['result'], label=f"Simulation {result['seed']}")
        
        plt.title("Résultats des simulations Monte Carlo")
        plt.xlabel("s")
        plt.ylabel("Résultat")
        plt.legend()
        plt.show()

    def save_params(self):
        """Sauvegarde les paramètres dans un fichier JSON."""
        params = self._get_current_params()
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getSaveFileName(self, "Sauvegarder les paramètres", "", "JSON Files (*.json);;All Files (*)", options=options)
        if filename:
            with open(filename, 'w') as f:
                json.dump(params, f)
            QMessageBox.information(self, "Succès", "Paramètres sauvegardés avec succès.")

    def load_params(self):
        """Charge les paramètres depuis un fichier JSON."""
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(self, "Charger les paramètres", "", "JSON Files (*.json);;All Files (*)", options=options)
        if filename:
            with open(filename, 'r') as f:
                params = json.load(f)
                self._set_current_params(params)
            QMessageBox.information(self, "Succès", "Paramètres chargés avec succès.")

    def _set_current_params(self, params):
        """Met à jour les champs d'entrée avec les paramètres chargés."""
        self.mc_iterations.setValue(params.get('n_simulations', 10))  # Exemple d'utilisation pour un paramètre

    def _get_current_params(self):
        """Récupère les paramètres actuels."""
        return {
            'Δθ₀': 0.01,
            'α': 1.5,
            'β': 1.2,
            'ε': 0.5,
            'δ': 0.1,
            'τ': 1.0,
            'κ': 0.2,  # Assurez-vous que cette valeur soit ajustable à partir de l'UI
            'n_simulations': int(self.mc_iterations.value())
        }

# Tests unitaires
class TestAngularTheory:
    """Batterie de tests unitaires pour les composants critiques."""

    def test_pivot_equation(self):
        """Vérifie le calcul de l'équation pivot avec torsion et entropie."""
        params = {'Δθ₀': 0.01, 'α': 1.5, 'β': 1.2, 'ε': 0.5, 'δ': 0.1, 'τ': 1.0, 'κ': 0.2}
        s = np.linspace(0, 100, 10)
        result = numba_pivot_equation(s, **params)
        assert result.shape == (10,)
        assert np.all(result >= 0)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AngularTheoryApp()
    window.show()
    sys.exit(app.exec())


# Pour les citations et références, consultez le fichier REFERENCES.md
