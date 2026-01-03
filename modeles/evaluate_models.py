"""
Script pour évaluer et comparer les modèles Prophet et LSTM
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
import os
from prophet import Prophet
from prophet.serialize import model_from_json
import tensorflow as tf
import joblib

class ModelEvaluator:
    def __init__(self):
        """Initialise l'évaluateur de modèles"""
        self.results = {}
        
    def load_models(self):
        """Charge les modèles entraînés"""
        print("📂 Chargement des modèles...")
        
        models = {}
        
        # Chargement Prophet
        try:
            with open('models/prophet_model.json', 'r') as fin:
                models['prophet'] = model_from_json(fin.read())
            
            with open('models/prophet_metrics.json', 'r') as f:
                self.results['prophet_metrics'] = json.load(f)
            
            print("✅ Modèle Prophet chargé")
        except FileNotFoundError:
            print("⚠️  Modèle Prophet non trouvé")
            models['prophet'] = None
        
        # Chargement LSTM
        try:
            models['lstm'] = tf.keras.models.load_model('models/lstm_model.h5')
            
            with open('models/lstm_metrics.json', 'r') as f:
                self.results['lstm_metrics'] = json.load(f)
            
            print("✅ Modèle LSTM chargé")
        except FileNotFoundError:
            print("⚠️  Modèle LSTM non trouvé")
            models['lstm'] = None
        
        # Chargement du scaler
        try:
            self.scaler = joblib.load('models/scaler.pkl')
            print("✅ Scaler chargé")
        except FileNotFoundError:
            print("⚠️  Scaler non trouvé")
            self.scaler = None
        
        return models
    
    def load_test_data(self):
        """Charge les données de test"""
        print("\n📊 Chargement des données de test...")
        
        try:
            # Données originales
            X_test_raw = np.load('data/processed/X_test_raw.npy')
            y_test = np.load('data/processed/y_test.npy')
            
            # Données normalisées pour LSTM
            X_test_scaled = np.load('data/processed/X_test_scaled.npy')
            X_test_seq = np.load('data/processed/X_test_seq.npy')
            y_test_seq = np.load('data/processed/y_test_seq.npy')
            
            print(f"  Données brutes: X={X_test_raw.shape}, y={y_test.shape}")
            print(f"  Séquences LSTM: X={X_test_seq.shape}, y={y_test_seq.shape}")
            
            return {
                'X_test_raw': X_test_raw,
                'y_test': y_test,
                'X_test_scaled': X_test_scaled,
                'X_test_seq': X_test_seq,
                'y_test_seq': y_test_seq
            }
        except FileNotFoundError as e:
            print(f"❌ Erreur de chargement: {e}")
            return None
    
    def evaluate_prophet_on_test(self, model, test_data):
        """Évalue Prophet sur l'ensemble de test complet"""
        if model is None or test_data is None:
            return None
        
        print("\n📈 Évaluation de Prophet sur l'ensemble de test...")
        
        # Prédiction avec Prophet
        future = model.make_future_dataframe(periods=len(test_data['y_test']))
        forecast = model.predict(future)
        
        # Extraction des prédictions pour la période de test
        prophet_test_preds = forecast['yhat'].values[-len(test_data['y_test']):]
        
        # Métriques
        mae = mean_absolute_error(test_data['y_test'], prophet_test_preds)
        rmse = np.sqrt(mean_squared_error(test_data['y_test'], prophet_test_preds))
        mape = np.mean(np.abs((test_data['y_test'] - prophet_test_preds) / test_data['y_test'])) * 100
        r2 = r2_score(test_data['y_test'], prophet_test_preds)
        
        results = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2_score': r2,
            'predictions': prophet_test_preds.tolist(),
            'actual': test_data['y_test'].tolist()
        }
        
        print(f"  MAE: {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  R² Score: {r2:.3f}")
        
        return results
    
    def evaluate_lstm_on_test(self, model, test_data):
        """Évalue LSTM sur l'ensemble de test"""
        if model is None or test_data is None or self.scaler is None:
            return None
        
        print("\n📈 Évaluation de LSTM sur l'ensemble de test...")
        
        # Prédictions
        y_pred_scaled = model.predict(test_data['X_test_seq'], verbose=0)
        
        # Inverse scaling
        n_features = self.scaler.n_features_in_
        dummy_array = np.zeros((len(y_pred_scaled), n_features))
        dummy_array[:, 0] = y_pred_scaled[:, 0]
        y_pred = self.scaler.inverse_transform(dummy_array)[:, 0]
        
        # Pour y_true, on prend les dernières valeurs (après séquence)
        y_true = test_data['y_test_seq']
        
        # Métriques
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        r2 = r2_score(y_true, y_pred)
        
        results = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2_score': r2,
            'predictions': y_pred.tolist(),
            'actual': y_true.tolist()
        }
        
        print(f"  MAE: {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  R² Score: {r2:.3f}")
        
        return results
    
    def compare_models(self, prophet_results, lstm_results):
        """Compare les performances des deux modèles"""
        print("\n" + "="*50)
        print("🔄 COMPARAISON DES MODÈLES")
        print("="*50)
        
        comparison = {}
        
        if prophet_results:
            print(f"\n📊 PROPHET:")
            for metric, value in prophet_results.items():
                if metric not in ['predictions', 'actual']:
                    print(f"  {metric.upper()}: {value:.4f}")
        
        if lstm_results:
            print(f"\n📊 LSTM:")
            for metric, value in lstm_results.items():
                if metric not in ['predictions', 'actual']:
                    print(f"  {metric.upper()}: {value:.4f}")
        
        # Détermination du meilleur modèle pour chaque métrique
        if prophet_results and lstm_results:
            print(f"\n🏆 MEILLEUR MODÈLE PAR MÉTRIQUE:")
            
            for metric in ['mae', 'rmse', 'mape']:
                if metric in prophet_results and metric in lstm_results:
                    prophet_val = prophet_results[metric]
                    lstm_val = lstm_results[metric]
                    
                    # Pour MAE et RMSE, plus bas = mieux
                    if metric in ['mae', 'rmse']:
                        if prophet_val < lstm_val:
                            winner = "Prophet"
                            diff = lstm_val - prophet_val
                            diff_pct = (diff / lstm_val) * 100
                        else:
                            winner = "LSTM"
                            diff = prophet_val - lstm_val
                            diff_pct = (diff / prophet_val) * 100
                    
                    # Pour MAPE, plus bas = mieux
                    elif metric == 'mape':
                        if prophet_val < lstm_val:
                            winner = "Prophet"
                            diff = lstm_val - prophet_val
                            diff_pct = (diff / lstm_val) * 100
                        else:
                            winner = "LSTM"
                            diff = prophet_val - lstm_val
                            diff_pct = (diff / prophet_val) * 100
                    
                    # Pour R², plus haut = mieux
                    elif metric == 'r2_score':
                        if prophet_val > lstm_val:
                            winner = "Prophet"
                            diff = prophet_val - lstm_val
                            diff_pct = (diff / lstm_val) * 100
                        else:
                            winner = "LSTM"
                            diff = lstm_val - prophet_val
                            diff_pct = (diff / prophet_val) * 100
                    
                    comparison[metric] = {
                        'winner': winner,
                        'difference': round(diff, 4),
                        'difference_percent': round(diff_pct, 2),
                        'prophet_value': round(prophet_val, 4),
                        'lstm_value': round(lstm_val, 4)
                    }
                    
                    print(f"  {metric.upper()}: {winner} "
                          f"(différence: {diff:.4f}, {diff_pct:.2f}%)")
        
        return comparison
    
    def plot_comparison(self, prophet_results, lstm_results, comparison):
        """Génère des graphiques de comparaison"""
        print("\n🎨 Génération des graphiques de comparaison...")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Prédictions vs réalité (Prophet)
        if prophet_results:
            ax1 = axes[0, 0]
            ax1.plot(prophet_results['actual'][:100], 'b-', label='Réel', alpha=0.7)
            ax1.plot(prophet_results['predictions'][:100], 'r--', label='Prédictions', alpha=0.8)
            ax1.set_xlabel('Échantillons')
            ax1.set_ylabel('Prix')
            ax1.set_title('Prédictions Prophet vs Réalité')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Prédictions vs réalité (LSTM)
        if lstm_results:
            ax2 = axes[0, 1]
            ax2.plot(lstm_results['actual'][:100], 'b-', label='Réel', alpha=0.7)
            ax2.plot(lstm_results['predictions'][:100], 'g--', label='Prédictions', alpha=0.8)
            ax2.set_xlabel('Échantillons')
            ax2.set_ylabel('Prix')
            ax2.set_title('Prédictions LSTM vs Réalité')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Comparaison des erreurs
        if prophet_results and lstm_results:
            ax3 = axes[0, 2]
            
            # Calcul des erreurs
            prophet_errors = np.array(prophet_results['actual']) - np.array(prophet_results['predictions'])
            lstm_errors = np.array(lstm_results['actual']) - np.array(lstm_results['predictions'])
            
            ax3.hist(prophet_errors, bins=30, alpha=0.5, label='Prophet', edgecolor='black')
            ax3.hist(lstm_errors, bins=30, alpha=0.5, label='LSTM', edgecolor='black')
            ax3.axvline(x=0, color='r', linestyle='--', alpha=0.5)
            ax3.set_xlabel('Erreur de prédiction')
            ax3.set_ylabel('Fréquence')
            ax3.set_title('Distribution des erreurs')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Métriques de comparaison (bar chart)
        if comparison:
            ax4 = axes[1, 0]
            metrics = list(comparison.keys())
            prophet_values = [comparison[m]['prophet_value'] for m in metrics]
            lstm_values = [comparison[m]['lstm_value'] for m in metrics]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            bars1 = ax4.bar(x - width/2, prophet_values, width, label='Prophet', alpha=0.8)
            bars2 = ax4.bar(x + width/2, lstm_values, width, label='LSTM', alpha=0.8)
            
            ax4.set_xlabel('Métriques')
            ax4.set_ylabel('Valeur')
            ax4.set_title('Comparaison des métriques')
            ax4.set_xticks(x)
            ax4.set_xticklabels([m.upper() for m in metrics])
            ax4.legend()
            ax4.grid(True, alpha=0.3, axis='y')
            
            # Ajout des valeurs sur les barres
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 5. Scatter plot des prédictions
        if prophet_results and lstm_results:
            ax5 = axes[1, 1]
            ax5.scatter(prophet_results['predictions'], lstm_results['predictions'], 
                       alpha=0.6, edgecolors='k')
            
            min_val = min(min(prophet_results['predictions']), min(lstm_results['predictions']))
            max_val = max(max(prophet_results['predictions']), max(lstm_results['predictions']))
            ax5.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='y=x')
            
            ax5.set_xlabel('Prédictions Prophet')
            ax5.set_ylabel('Prédictions LSTM')
            ax5.set_title('Corrélation entre les prédictions')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 6. Recommandation finale
        ax6 = axes[1, 2]
        
        if comparison:
            # Comptage des victoires
            prophet_wins = sum(1 for m in comparison.values() if m['winner'] == 'Prophet')
            lstm_wins = sum(1 for m in comparison.values() if m['winner'] == 'LSTM')
            
            labels = ['Prophet', 'LSTM']
            sizes = [prophet_wins, lstm_wins]
            colors = ['#ff9999', '#66b3ff']
            
            wedges, texts, autotexts = ax6.pie(sizes, labels=labels, colors=colors,
                                             autopct='%1.1f%%', startangle=90)
            
            ax6.set_title(f'Modèle recommandé\n'
                         f'Prophet: {prophet_wins}/4, LSTM: {lstm_wins}/4')
            
            # Détermination du modèle recommandé
            if prophet_wins > lstm_wins:
                recommendation = "Prophet"
                reason = "Meilleur sur plus de métriques"
            elif lstm_wins > prophet_wins:
                recommendation = "LSTM"
                reason = "Meilleur sur plus de métriques"
            else:
                recommendation = "LSTM"
                reason = "Égalité - LSTM recommandé pour sa capacité à capturer les patterns complexes"
            
            ax6.text(0, -1.5, f"Recommandation: {recommendation}\n{reason}", 
                    ha='center', fontsize=10, bbox=dict(boxstyle="round", alpha=0.1))
        else:
            ax6.text(0.5, 0.5, 'Comparaison non disponible',
                    horizontalalignment='center',
                    verticalalignment='center')
            ax6.set_title('Recommandation')
        
        plt.tight_layout()
        plt.savefig('outputs/model_comparison.png', dpi=300, bbox_inches='tight')
        print("✅ Graphique de comparaison sauvegardé: outputs/model_comparison.png")
        
        return fig
    
    def generate_report(self, prophet_results, lstm_results, comparison):
        """Génère un rapport détaillé d'évaluation"""
        print("\n📝 Génération du rapport d'évaluation...")
        
        report = {
            "evaluation_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "models_evaluated": {
                "prophet": prophet_results is not None,
                "lstm": lstm_results is not None
            },
            "test_set_size": len(prophet_results['actual']) if prophet_results else 
                            len(lstm_results['actual']) if lstm_results else 0,
            "metrics_comparison": comparison,
            "detailed_results": {
                "prophet": prophet_results,
                "lstm": lstm_results
            } if prophet_results and lstm_results else None,
            "recommendation": self._generate_recommendation(comparison)
        }
        
        # Sauvegarde du rapport
        with open('outputs/model_evaluation_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=self._json_serializer)
        
        # Génération d'un rapport Markdown
        self._generate_markdown_report(report)
        
        print("✅ Rapport d'évaluation sauvegardé: outputs/model_evaluation_report.json")
        print("✅ Rapport Markdown généré: outputs/model_evaluation_report.md")
        
        return report
    
    def _generate_recommendation(self, comparison):
        """Génère une recommandation basée sur la comparaison"""
        if not comparison:
            return {
                "recommended_model": "Aucun",
                "reason": "Données insuffisantes pour la comparaison",
                "confidence": 0
            }
        
        # Comptage des victoires
        prophet_wins = sum(1 for m in comparison.values() if m['winner'] == 'Prophet')
        lstm_wins = sum(1 for m in comparison.values() if m['winner'] == 'LSTM')
        
        if prophet_wins > lstm_wins:
            return {
                "recommended_model": "Prophet",
                "reason": "Meilleures performances sur la majorité des métriques",
                "confidence": round((prophet_wins / len(comparison)) * 100, 1),
                "advantages": [
                    "Meilleure précision générale",
                    "Intervalles de confiance intégrés",
                    "Interprétabilité des composantes saisonnières"
                ],
                "use_case": "Prédictions à moyen terme avec besoin d'interprétabilité"
            }
        elif lstm_wins > prophet_wins:
            return {
                "recommended_model": "LSTM",
                "reason": "Meilleures performances sur la majorité des métriques",
                "confidence": round((lstm_wins / len(comparison)) * 100, 1),
                "advantages": [
                    "Meilleure capture des patterns non-linéaires",
                    "Capacité à apprendre des dépendances temporelles complexes",
                    "Adaptabilité aux changements de patterns"
                ],
                "use_case": "Prédictions à court terme nécessitant une haute précision"
            }
        else:
            return {
                "recommended_model": "LSTM",
                "reason": "Égalité des performances - LSTM recommandé pour sa flexibilité",
                "confidence": 50.0,
                "advantages": [
                    "Flexibilité architecturale",
                    "Capacité à intégrer des features supplémentaires",
                    "Potentiel d'amélioration avec plus de données"
                ],
                "use_case": "Scénarios où les deux modèles sont équivalents"
            }
    
    def _generate_markdown_report(self, report):
        """Génère un rapport au format Markdown"""
        md_content = [
            "# Rapport d'Évaluation des Modèles",
            f"**Date d'évaluation**: {report['evaluation_date']}",
            f"**Taille de l'ensemble de test**: {report['test_set_size']} échantillons",
            "",
            "## 📊 Modèles Évalués",
            ""
        ]
        
        if report['models_evaluated']['prophet']:
            md_content.append("- ✅ **Prophet**: Modèle de séries temporelles classique")
        
        if report['models_evaluated']['lstm']:
            md_content.append("- ✅ **LSTM**: Réseau de neurones récurrent")
        
        md_content.extend([
            "",
            "## 📈 Résultats Détaillés",
            ""
        ])
        
        if report['detailed_results']:
            # Tableau des métriques
            md_content.append("| Métrique | Prophet | LSTM | Différence | Meilleur |")
            md_content.append("|----------|---------|------|------------|----------|")
            
            for metric, comp in report['metrics_comparison'].items():
                md_content.append(
                    f"| {metric.upper()} | {comp['prophet_value']:.4f} | {comp['lstm_value']:.4f} | "
                    f"{comp['difference']:.4f} ({comp['difference_percent']}%) | **{comp['winner']}** |"
                )
        
        md_content.extend([
            "",
            "## 🏆 Recommandation",
            "",
            f"**Modèle recommandé**: **{report['recommendation']['recommended_model']}**",
            f"**Confiance**: {report['recommendation']['confidence']}%",
            f"**Raison**: {report['recommendation']['reason']}",
            "",
            "### Avantages du modèle recommandé:",
            ""
        ])
        
        for advantage in report['recommendation']['advantages']:
            md_content.append(f"- {advantage}")
        
        md_content.extend([
            "",
            f"### Cas d'utilisation recommandé:",
            f"{report['recommendation']['use_case']}",
            "",
            "## 📋 Conclusion",
            "",
            "Cette évaluation compare les performances de deux approches différentes "
            "pour la prédiction des prix Bitcoin. Le modèle recommandé devrait être utilisé "
            "comme base pour le système de recommandation de trading.",
            "",
            "> **Note**: Ces résultats sont basés sur des données historiques et ne "
            "garantissent pas les performances futures. Toujours effectuer une analyse "
            "de risque supplémentaire avant de prendre des décisions de trading."
        ])
        
        with open('outputs/model_evaluation_report.md', 'w') as f:
            f.write('\n'.join(md_content))
    
    def _json_serializer(self, obj):
        """Helper pour sérialiser des objets non-standard en JSON"""
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        else:
            return str(obj)

def main():
    """Pipeline principal d'évaluation"""
    
    # Création des dossiers
    os.makedirs('outputs', exist_ok=True)
    
    # Initialisation
    evaluator = ModelEvaluator()
    
    print("="*60)
    print("📊 ÉVALUATION ET COMPARAISON DES MODÈLES")
    print("="*60)
    
    # 1. Chargement des modèles
    models = evaluator.load_models()
    
    # 2. Chargement des données de test
    test_data = evaluator.load_test_data()
    
    if test_data is None:
        print("❌ Impossible de charger les données de test")
        return
    
    # 3. Évaluation des modèles
    prophet_results = evaluator.evaluate_prophet_on_test(models.get('prophet'), test_data)
    lstm_results = evaluator.evaluate_lstm_on_test(models.get('lstm'), test_data)
    
    # 4. Comparaison
    comparison = evaluator.compare_models(prophet_results, lstm_results)
    
    # 5. Visualisation
    evaluator.plot_comparison(prophet_results, lstm_results, comparison)
    
    # 6. Génération du rapport
    report = evaluator.generate_report(prophet_results, lstm_results, comparison)
    
    # Affichage de la recommandation
    print("\n" + "="*60)
    print("🎯 RECOMMANDATION FINALE")
    print("="*60)
    
    rec = report['recommendation']
    print(f"\n📌 Modèle recommandé: {rec['recommended_model']}")
    print(f"📊 Confiance: {rec['confidence']}%")
    print(f"📝 Raison: {rec['reason']}")
    print(f"\n🏆 Avantages:")
    for advantage in rec['advantages']:
        print(f"  • {advantage}")
    print(f"\n🎯 Cas d'utilisation: {rec['use_case']}")
    
    print("\n" + "="*60)
    print("✅ ÉVALUATION TERMINÉE")
    print("="*60)
    
    return evaluator, report

if __name__ == "__main__":
    evaluator, report = main()