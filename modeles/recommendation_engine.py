"""
Moteur de recommandation pour transformer les prédictions en signaux d'achat/vente
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import os

class RecommendationEngine:
    def __init__(self, threshold=0.02, hold_threshold=0.01):
        """
        Initialise le moteur de recommandation
        
        Args:
            threshold: Seuil de changement pour déclencher BUY/SELL (2% par défaut)
            hold_threshold: Seuil pour HOLD (entre -1% et +1%)
        """
        self.threshold = threshold
        self.hold_threshold = hold_threshold
        self.history = []
        
    def calculate_price_change(self, current_price, predicted_price):
        """Calcule le pourcentage de changement"""
        return (predicted_price - current_price) / current_price
    
    def generate_signal(self, price_change, confidence):
        """
        Génère un signal basé sur le changement de prix
        
        Returns:
            dict: Signal avec action, score de confiance et détails
        """
        # Détermination de l'action
        if price_change > self.threshold:
            action = "BUY"
            action_score = min(90 + (price_change * 1000), 99)
        elif price_change < -self.threshold:
            action = "SELL"
            action_score = min(90 + (abs(price_change) * 1000), 99)
        else:
            action = "HOLD"
            action_score = 70  # Score fixe pour HOLD
        
        # Ajustement avec la confiance du modèle
        adjusted_confidence = min(action_score * (confidence / 100), 99)
        
        # Génération du message
        if action == "BUY":
            message = f"Prévision de hausse de {price_change*100:.2f}%"
        elif action == "SELL":
            message = f"Prévision de baisse de {abs(price_change)*100:.2f}%"
        else:
            message = f"Marché stable (changement de {price_change*100:.2f}%)"
        
        return {
            "action": action,
            "confidence": round(adjusted_confidence, 1),
            "predicted_change_percent": round(price_change * 100, 2),
            "message": message,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def generate_portfolio_recommendation(self, signals, portfolio_value=10000):
        """
        Génère une recommandation pour un portefeuille
        
        Args:
            signals: Liste des signaux pour différents actifs
            portfolio_value: Valeur totale du portefeuille
        
        Returns:
            dict: Recommandations d'allocation
        """
        recommendations = {
            "portfolio_value": portfolio_value,
            "total_confidence": 0,
            "allocations": [],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Filtrage des signaux valides
        valid_signals = [s for s in signals if s["confidence"] > 60]
        
        if not valid_signals:
            return {
                **recommendations,
                "message": "Aucun signal assez fort pour recommander une action",
                "action": "HOLD_ALL"
            }
        
        # Calcul des poids basés sur la confiance
        total_confidence = sum(s["confidence"] for s in valid_signals)
        recommendations["total_confidence"] = round(total_confidence / len(valid_signals), 1)
        
        # Allocation pour chaque actif
        for signal in valid_signals:
            weight = signal["confidence"] / total_confidence if total_confidence > 0 else 0
            
            allocation = {
                "action": signal["action"],
                "confidence": signal["confidence"],
                "allocation_percent": round(weight * 100, 1),
                "allocation_amount": round(portfolio_value * weight, 2),
                "predicted_change": signal["predicted_change_percent"],
                "message": signal["message"]
            }
            
            recommendations["allocations"].append(allocation)
        
        # Détermination de l'action globale
        buy_signals = [s for s in valid_signals if s["action"] == "BUY"]
        sell_signals = [s for s in valid_signals if s["action"] == "SELL"]
        
        if len(buy_signals) > len(sell_signals):
            recommendations["overall_action"] = "BULLISH"
        elif len(sell_signals) > len(buy_signals):
            recommendations["overall_action"] = "BEARISH"
        else:
            recommendations["overall_action"] = "NEUTRAL"
        
        return recommendations
    
    def backtest_strategy(self, historical_data, predictions, initial_balance=10000):
        """
        Backtest la stratégie de trading
        
        Args:
            historical_data: Données historiques des prix
            predictions: Prédictions générées
            initial_balance: Balance initiale
        
        Returns:
            dict: Résultats du backtest
        """
        print("🔍 Backtesting de la stratégie...")
        
        balance = initial_balance
        position = 0  # Nombre d'unités détenues
        trades = []
        portfolio_values = []
        
        for i in range(len(predictions)):
            current_price = historical_data.iloc[i]['Close']
            predicted_price = predictions[i]
            
            # Calcul du changement
            price_change = self.calculate_price_change(current_price, predicted_price)
            
            # Génération du signal
            signal = self.generate_signal(price_change, confidence=80)
            
            # Exécution du trade
            if signal["action"] == "BUY" and position == 0:
                # Achat
                position = balance / current_price
                balance = 0
                trades.append({
                    "date": historical_data.iloc[i]['Date'],
                    "action": "BUY",
                    "price": current_price,
                    "units": position,
                    "signal_confidence": signal["confidence"]
                })
                
            elif signal["action"] == "SELL" and position > 0:
                # Vente
                balance = position * current_price
                position = 0
                trades.append({
                    "date": historical_data.iloc[i]['Date'],
                    "action": "SELL",
                    "price": current_price,
                    "profit": balance - initial_balance,
                    "signal_confidence": signal["confidence"]
                })
            
            # Calcul de la valeur du portefeuille
            portfolio_value = balance + (position * current_price)
            portfolio_values.append(portfolio_value)
        
        # Calcul des métriques de performance
        final_value = portfolio_values[-1] if portfolio_values else initial_balance
        total_return = ((final_value - initial_balance) / initial_balance) * 100
        
        # Calcul de la volatilité
        returns = np.diff(portfolio_values) / portfolio_values[:-1] * 100
        volatility = np.std(returns) if len(returns) > 0 else 0
        
        # Sharpe Ratio (simplifié)
        sharpe_ratio = (total_return / volatility) if volatility > 0 else 0
        
        results = {
            "initial_balance": initial_balance,
            "final_balance": round(final_value, 2),
            "total_return_percent": round(total_return, 2),
            "total_trades": len(trades),
            "volatility_percent": round(volatility, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "win_rate": self.calculate_win_rate(trades),
            "trades": trades[-10:] if trades else [],  # 10 derniers trades seulement
            "portfolio_values": portfolio_values
        }
        
        return results
    
    def calculate_win_rate(self, trades):
        """Calcule le taux de réussite des trades"""
        if len(trades) < 2:
            return 0
        
        profitable_trades = 0
        for i in range(1, len(trades)):
            if trades[i]['action'] == 'SELL':
                profit = trades[i]['profit']
                if profit > 0:
                    profitable_trades += 1
        
        return round((profitable_trades / (len(trades) // 2)) * 100, 1) if trades else 0
    
    def evaluate_signals(self, true_prices, predicted_prices, signals):
        """
        Évalue la précision des signaux
        
        Args:
            true_prices: Prix réels
            predicted_prices: Prix prédits
            signals: Signaux générés
        
        Returns:
            dict: Métriques d'évaluation
        """
        # Conversion des signaux en labels
        true_labels = []
        predicted_labels = []
        
        for i in range(1, len(true_prices)):
            actual_change = (true_prices[i] - true_prices[i-1]) / true_prices[i-1]
            predicted_change = (predicted_prices[i] - true_prices[i-1]) / true_prices[i-1]
            
            # Labels réels
            if actual_change > self.threshold:
                true_labels.append("BUY")
            elif actual_change < -self.threshold:
                true_labels.append("SELL")
            else:
                true_labels.append("HOLD")
            
            # Labels prédits
            if predicted_change > self.threshold:
                predicted_labels.append("BUY")
            elif predicted_change < -self.threshold:
                predicted_labels.append("SELL")
            else:
                predicted_labels.append("HOLD")
        
        # Calcul des métriques
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
        recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
        f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)
        
        return {
            "accuracy": round(accuracy * 100, 2),
            "precision": round(precision * 100, 2),
            "recall": round(recall * 100, 2),
            "f1_score": round(f1 * 100, 2),
            "n_samples": len(true_labels)
        }
    
    def plot_performance(self, backtest_results, save_path="outputs/recommendation_performance.png"):
        """Visualise la performance de la stratégie"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 1. Valeur du portefeuille
        ax1 = axes[0, 0]
        portfolio_values = backtest_results["portfolio_values"]
        ax1.plot(portfolio_values, 'b-', linewidth=2)
        ax1.axhline(y=backtest_results["initial_balance"], color='r', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Jours')
        ax1.set_ylabel('Valeur du portefeuille ($)')
        ax1.set_title(f'Performance du portefeuille\nRendement total: {backtest_results["total_return_percent"]}%')
        ax1.grid(True, alpha=0.3)
        ax1.fill_between(range(len(portfolio_values)), 
                        backtest_results["initial_balance"], 
                        portfolio_values,
                        alpha=0.3, color='green' if backtest_results["total_return_percent"] > 0 else 'red')
        
        # 2. Distribution des rendements
        ax2 = axes[0, 1]
        returns = np.diff(portfolio_values) / portfolio_values[:-1] * 100
        ax2.hist(returns, bins=30, edgecolor='black', alpha=0.7)
        ax2.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Rendement quotidien (%)')
        ax2.set_ylabel('Fréquence')
        ax2.set_title(f'Distribution des rendements\nVolatilité: {backtest_results["volatility_percent"]}%')
        ax2.grid(True, alpha=0.3)
        
        # 3. Métriques de performance
        ax3 = axes[1, 0]
        metrics = ['Rendement', 'Volatilité', 'Sharpe', 'Win Rate']
        values = [
            backtest_results["total_return_percent"],
            backtest_results["volatility_percent"],
            backtest_results["sharpe_ratio"],
            backtest_results["win_rate"]
        ]
        colors = ['green' if v > 0 else 'red' if v < 0 else 'gray' for v in values]
        bars = ax3.bar(metrics, values, color=colors, alpha=0.7)
        ax3.set_ylabel('Valeur')
        ax3.set_title('Métriques de performance')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Ajout des valeurs sur les barres
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.1f}', ha='center', va='bottom')
        
        # 4. Trades
        ax4 = axes[1, 1]
        trades = backtest_results["trades"]
        if trades:
            buy_dates = [i for i, t in enumerate(trades) if t["action"] == "BUY"]
            sell_dates = [i for i, t in enumerate(trades) if t["action"] == "SELL"]
            
            ax4.scatter(buy_dates, [1] * len(buy_dates), color='green', 
                       s=100, marker='^', label='BUY', alpha=0.7)
            ax4.scatter(sell_dates, [-1] * len(sell_dates), color='red',
                       s=100, marker='v', label='SELL', alpha=0.7)
            
            ax4.set_xlabel('Trade #')
            ax4.set_yticks([-1, 1])
            ax4.set_yticklabels(['SELL', 'BUY'])
            ax4.set_title(f'Signaux de trading\nTotal trades: {backtest_results["total_trades"]}')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Aucun trade effectué',
                    horizontalalignment='center',
                    verticalalignment='center')
            ax4.set_title('Signaux de trading')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Graphique de performance sauvegardé: {save_path}")
        
        return fig
    
    def save_recommendations(self, recommendations, filename="outputs/recommendations.json"):
        """Sauvegarde les recommandations"""
        with open(filename, 'w') as f:
            json.dump(recommendations, f, indent=2)
        print(f"✅ Recommandations sauvegardées: {filename}")

def main():
    """Pipeline principal du moteur de recommandation"""
    
    # Création des dossiers
    os.makedirs('outputs', exist_ok=True)
    
    # Initialisation
    engine = RecommendationEngine(threshold=0.02, hold_threshold=0.01)
    
    print("🎯 MOTEUR DE RECOMMANDATION")
    print("="*50)
    
    # Exemple: Génération de signaux
    print("\n1. Génération de signaux d'exemple...")
    
    # Données d'exemple
    current_prices = [50000, 51000, 49500, 50500, 52000]
    predicted_prices = [51500, 52500, 48500, 51500, 53500]
    confidences = [85, 80, 75, 82, 88]
    
    signals = []
    for current, predicted, confidence in zip(current_prices, predicted_prices, confidences):
        price_change = engine.calculate_price_change(current, predicted)
        signal = engine.generate_signal(price_change, confidence)
        signals.append(signal)
        
        print(f"  Prix actuel: ${current:,.0f}, Prédit: ${predicted:,.0f}")
        print(f"  → Action: {signal['action']}, Confiance: {signal['confidence']}%")
        print(f"  Message: {signal['message']}")
        print()
    
    # 2. Recommandation de portefeuille
    print("2. Recommandation de portefeuille...")
    portfolio_recommendation = engine.generate_portfolio_recommendation(
        signals, portfolio_value=10000
    )
    
    print(f"  Valeur du portefeuille: ${portfolio_recommendation['portfolio_value']:,.2f}")
    print(f"  Confiance totale: {portfolio_recommendation['total_confidence']}%")
    print(f"  Tendance générale: {portfolio_recommendation['overall_action']}")
    
    if portfolio_recommendation['allocations']:
        print("\n  Allocations recommandées:")
        for alloc in portfolio_recommendation['allocations']:
            print(f"    • {alloc['action']}: {alloc['allocation_percent']}% "
                  f"(${alloc['allocation_amount']:.2f}) - "
                  f"Confiance: {alloc['confidence']}%")
    
    # 3. Backtesting (avec données simulées)
    print("\n3. Backtesting de la stratégie...")
    
    # Simulation de données historiques
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    historical_data = pd.DataFrame({
        'Date': dates,
        'Close': np.random.normal(50000, 2000, 100).cumsum()
    })
    
    # Simulation de prédictions
    predictions = historical_data['Close'].values + np.random.normal(0, 500, 100)
    
    # Backtest
    backtest_results = engine.backtest_strategy(
        historical_data, predictions, initial_balance=10000
    )
    
    print(f"  Balance initiale: ${backtest_results['initial_balance']:,.2f}")
    print(f"  Balance finale: ${backtest_results['final_balance']:,.2f}")
    print(f"  Rendement total: {backtest_results['total_return_percent']}%")
    print(f"  Volatilité: {backtest_results['volatility_percent']}%")
    print(f"  Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
    print(f"  Win Rate: {backtest_results['win_rate']}%")
    print(f"  Total trades: {backtest_results['total_trades']}")
    
    # 4. Évaluation des signaux
    print("\n4. Évaluation de la précision des signaux...")
    evaluation = engine.evaluate_signals(
        historical_data['Close'].values,
        predictions,
        signals * 20  # Répétition pour avoir assez de signaux
    )
    
    print(f"  Précision: {evaluation['accuracy']}%")
    print(f"  F1-Score: {evaluation['f1_score']}%")
    print(f"  Échantillons: {evaluation['n_samples']}")
    
    # 5. Visualisation
    print("\n5. Génération des visualisations...")
    engine.plot_performance(backtest_results)
    
    # 6. Sauvegarde
    print("\n6. Sauvegarde des résultats...")
    
    # Sauvegarde des signaux
    engine.save_recommendations({
        "signals": signals,
        "portfolio_recommendation": portfolio_recommendation,
        "backtest_results": backtest_results,
        "evaluation_metrics": evaluation,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    
    print("\n" + "="*50)
    print("✅ MOTEUR DE RECOMMANDATION TERMINÉ")
    print("="*50)
    
    return engine, signals, portfolio_recommendation, backtest_results

if __name__ == "__main__":
    engine, signals, portfolio_rec, backtest = main()