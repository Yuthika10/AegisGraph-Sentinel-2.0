import numpy as np
from collections import deque, defaultdict
import networkx as nx

class LateralMovementDetector:
    """
    Detects MITRE ATTACK TA0008 (Lateral Movement) by tracking 
    spikes in Betweenness Centrality across a temporal graph.
    """
    def __init__(self, history_size=10, std_multiplier=2.0, spike_multiplier=3.0, risk_penalty=0.25):
        self.history_size = history_size
        self.std_multiplier = std_multiplier
        self.spike_multiplier = spike_multiplier
        self.risk_penalty = risk_penalty
        
        # O(1) append/pop queue to track the last N scores per account safely
        self.centrality_history = defaultdict(lambda: deque(maxlen=self.history_size))
        
        # Active in-memory graph for real-time edge updates
        self.active_graph = nx.DiGraph()

    def update_graph(self, src_account, dst_account):
        """Updates the network topology dynamically."""
        if self.active_graph.has_edge(src_account, dst_account):
            self.active_graph[src_account][dst_account]['weight'] += 1
        else:
            self.active_graph.add_edge(src_account, dst_account, weight=1)

    def _calculate_approx_centrality(self, account_id):
        """
        Calculates localized betweenness centrality. 
        Uses approximation (k-nodes) to avoid O(V*E) bottlenecks in real-time.
        """
        num_nodes = self.active_graph.number_of_nodes()
        if num_nodes < 3:
            return 0.0 
            
        # Limit the sampling to max 50 nodes for extreme performance
        k_approx = min(50, num_nodes)
        
        centralities = nx.betweenness_centrality(
            self.active_graph, 
            k=k_approx, 
            normalized=True,
            weight='weight'
        )
        return centralities.get(account_id, 0.0)

    def analyze_account(self, account_id):
        """
        Evaluates the account against its historical baseline.
        Returns: (risk_score_added: float, is_pivoting: bool)
        """
        current_score = self._calculate_approx_centrality(account_id)
        history = self.centrality_history[account_id]
        
        # If we don't have enough history to form a baseline, just record and pass
        if len(history) < 3:
            history.append(current_score)
            return 0.0, False
            
        baseline_mean = np.mean(history)
        baseline_std = np.std(history)
        
        # Trigger 1: Current score is 2 standard deviations above mean
        std_trigger = current_score > (baseline_mean + self.std_multiplier * baseline_std)
        
        # Trigger 2: Current score is 3x the baseline average
        mult_trigger = current_score > (self.spike_multiplier * baseline_mean)
        
        is_pivoting = False
        risk_added = 0.0
        
        # Ignore micro-fluctuations near zero (noise)
        if (std_trigger or mult_trigger) and current_score > 0.05:
            is_pivoting = True
            risk_added = self.risk_penalty
            
        # Update history queue
        history.append(current_score)
        
        return risk_added, is_pivoting

if __name__ == "__main__":
    # Quick Local Test
    detector = LateralMovementDetector()
    print("Initializing Lateral Movement Engine...")
    
    # Simulate an attacker pivoting through an account (ACC_001)
    detector.update_graph("ACC_002", "ACC_001")
    detector.update_graph("ACC_001", "ACC_003")
    
    # Manually seed normal historical data
    detector.centrality_history["ACC_001"].extend([0.01, 0.015, 0.012, 0.01])
    
    # Trigger an analysis on a massive spike (0.15 is huge for centrality)
    detector.active_graph.add_edges_from([("X", "ACC_001"), ("Y", "ACC_001"), ("ACC_001", "Z")])
    risk, flagged = detector.analyze_account("ACC_001")
    
    print(f"Risk Penalty Added: {risk} | Flagged as Mule/Pivot: {flagged}")