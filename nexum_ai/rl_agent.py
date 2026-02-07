"""
Reinforcement Learning Query Optimizer using Q-Learning
Learns to optimize query execution strategies based on performance metrics
"""

import numpy as np
from typing import Dict, Optional, Any


class QLearningAgent:
    """
    Q-Learning agent for query optimization
    
    State space: [query_length_bucket, cache_hit, complexity_score]
    Action space: [force_cache, bypass_cache, index_scan(future)]
    Reward: inverse of execution latency
    """
    
    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        epsilon: float = 0.2,
        epsilon_decay: float = 0.995,
        state_file: str = "q_table.pkl"
    ) -> None:
        """
        Initialize Q-learning agent
        
        Args:
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon: Exploration rate
            epsilon_decay: Epsilon decay rate per episode
            state_file: Path to save/load Q-table
        """
        self.q_table: Dict[str, Dict[str, float]] = {}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = epsilon_decay
        self.state_file = state_file
        
        self.actions = ["force_cache", "bypass_cache", "normal"]
        
        self.training_history = []
        self.episode_count = 0
        
        self.load_state()
        
        print(f"RL Agent initialized: lr={learning_rate}, gamma={discount_factor}, epsilon={epsilon}")
    
    def _get_state_key(self, query_length: int, cache_hit: bool, complexity: int) -> str:
        """
        Convert state to string key for Q-table
        
        Args:
            query_length: Length of SQL query
            cache_hit: Whether query hit cache
            complexity: Complexity score (0-10)
        
        Returns:
            State key string
        """
        query_bucket = min(query_length // 10, 10)
        state = f"state_{query_bucket}_{int(cache_hit)}_{complexity}"
        return state
    
    def get_action(self, query_length: int, cache_hit: bool, complexity: int) -> str:
        """
        Select action using epsilon-greedy policy
        
        Args:
            query_length: Length of SQL query
            cache_hit: Whether query hit cache
            complexity: Complexity score
        
        Returns:
            Selected action
        """
        state = self._get_state_key(query_length, cache_hit, complexity)
        
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.actions)
            print(f"Exploring: chose {action}")
            return action
        
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in self.actions}
        
        q_values = self.q_table[state]
        best_action = max(q_values, key=q_values.get)
        
        print(f"Exploiting: chose {best_action} (Q={q_values[best_action]:.3f})")
        return best_action
    
    def update(
        self,
        query_length: int,
        cache_hit: bool,
        complexity: int,
        action: str,
        latency_ms: float
    ) -> None:
        """
        Update Q-value based on observed reward
        
        Args:
            query_length: Length of SQL query
            cache_hit: Whether query hit cache
            complexity: Complexity score
            action: Action taken
            latency_ms: Execution latency in ms (lower is better)
        """
        state = self._get_state_key(query_length, cache_hit, complexity)
        
        reward = self._calculate_reward(latency_ms, cache_hit)
        
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in self.actions}
        
        current_q = self.q_table[state].get(action, 0.0)
        
        max_next_q = max(self.q_table[state].values()) if self.q_table[state] else 0.0
        
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        
        self.q_table[state][action] = new_q
        
        self.training_history.append({
            'state': state,
            'action': action,
            'reward': reward,
            'q_value': new_q,
            'latency_ms': latency_ms
        })
        
        print(f"Updated Q({state}, {action}) = {new_q:.4f} (reward={reward:.4f}, latency={latency_ms:.2f}ms)")
    
    def _calculate_reward(self, latency_ms: float, cache_hit: bool) -> float:
        """
        Calculate reward based on latency and cache performance
        
        Args:
            latency_ms: Execution latency in milliseconds
            cache_hit: Whether query hit cache
        
        Returns:
            Reward value
        """
        base_reward = -latency_ms / 100.0
        
        if cache_hit and latency_ms < 1.0:
            base_reward += 5.0
        
        return base_reward
    
    def decay_epsilon(self) -> None:
        """Decay exploration rate after each episode"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.episode_count += 1
            print(f"Episode {self.episode_count}: epsilon decayed to {self.epsilon:.4f}")
    
    def get_stats(self) -> Dict[str, float]:
        """Get agent statistics"""
        return {
            'q_table_size': len(self.q_table),
            'total_updates': len(self.training_history),
            'epsilon': self.epsilon,
            'episodes': self.episode_count,
            'avg_reward': np.mean([h['reward'] for h in self.training_history[-100:]]) if self.training_history else 0.0
        }
    
    def explain_action(self, query_length: int, cache_hit: bool, complexity: int) -> Dict[str, Any]:
        """
        Explain what action would be taken without executing.
        
        Returns Q-values, state analysis, and predicted action for EXPLAIN command.
        This method provides a read-only analysis of the RL agent's decision-making
        process without actually executing any action or updating the Q-table.
        
        Args:
            query_length: Length of SQL query
            cache_hit: Whether query hit cache
            complexity: Complexity score (0-10)
        
        Returns:
            Dict containing:
                - state: state key string
                - state_breakdown: dict with query_length_bucket, cache_hit, complexity
                - q_values: Q-values for all actions
                - best_action: action with highest Q-value
                - epsilon: current exploration rate
                - would_explore: whether exploration is possible
                - predicted_action: deterministic best action (ignores epsilon-greedy)
                - explanation: human-readable explanation of agent behavior
                - agent_stats: total_states_learned, total_updates, episodes
        """
        state = self._get_state_key(query_length, cache_hit, complexity)
        
        # Get Q-values for this state
        if state in self.q_table:
            q_values = {a: round(v, 4) for a, v in self.q_table[state].items()}
        else:
            q_values = {a: 0.0 for a in self.actions}
        
        # Determine best action
        best_action = max(self.actions, key=lambda a: q_values.get(a, 0.0))
        
        # Truncate best_action for display if needed (defensive limit)
        best_action_display = best_action[:20] if len(best_action) > 20 else best_action
        
        return {
            'state': state,
            'state_breakdown': {
                'query_length_bucket': min(query_length // 10, 10),
                'cache_hit': cache_hit,
                'complexity': complexity
            },
            'q_values': q_values,
            'best_action': best_action_display,
            'epsilon': round(self.epsilon, 4),
            'would_explore': self.epsilon > 0,
            'predicted_action': best_action_display,  # Deterministic for explain
            'explanation': f'With ε={self.epsilon:.4f}, agent would explore {self.epsilon*100:.1f}% of the time',
            'agent_stats': {
                'total_states_learned': len(self.q_table),
                'total_updates': len(self.training_history),
                'episodes': self.episode_count
            }
        }
    
    def save_state(self, filepath: Optional[str] = None) -> None:
        """Save Q-table and agent state to file using joblib"""
        try:
            import joblib
        except ImportError:
            print("Warning: joblib not installed, cannot save state")
            return
        
        if filepath is None:
            filepath = self.state_file
            
        data = {
            'q_table': self.q_table,
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'history_size': len(self.training_history)
        }
        
        try:
            joblib.dump(data, filepath)
            print(f"Agent state saved to {filepath}")
        except Exception as e:
            print(f"Error saving agent state: {e}")
    
    def load_state(self, filepath: Optional[str] = None) -> None:
        """Load Q-table and agent state from file using joblib"""
        try:
            import joblib
        except ImportError:
            print("Warning: joblib not installed, cannot load state")
            return
        
        import os
        
        if filepath is None:
            filepath = self.state_file
            
        if os.path.exists(filepath):
            try:
                data = joblib.load(filepath)
                self.q_table = data.get('q_table', {})
                self.epsilon = data.get('epsilon', self.epsilon)
                self.episode_count = data.get('episode_count', 0)
                print(f"Agent state loaded from {filepath}: {len(self.q_table)} states, epsilon={self.epsilon:.4f}")
            except Exception as e:
                print(f"Error loading agent state: {e}")
        else:
            print(f"No saved state found at {filepath}, starting fresh")


def test_rl_agent() -> None:
    """Test the RL agent with simulated queries"""
    agent = QLearningAgent(learning_rate=0.1, epsilon=0.3)
    
    print("\n" + "="*60)
    print("RL Agent Training Simulation")
    print("="*60 + "\n")
    
    test_scenarios = [
        {"query_len": 25, "cache_hit": False, "complexity": 3, "latency": 5.2},
        {"query_len": 25, "cache_hit": True, "complexity": 3, "latency": 0.05},
        {"query_len": 42, "cache_hit": False, "complexity": 7, "latency": 12.5},
        {"query_len": 42, "cache_hit": True, "complexity": 7, "latency": 0.08},
        {"query_len": 18, "cache_hit": False, "complexity": 2, "latency": 2.3},
        {"query_len": 18, "cache_hit": True, "complexity": 2, "latency": 0.04},
    ]
    
    for i, scenario in enumerate(test_scenarios):
        print(f"\nIteration {i+1}:")
        
        action = agent.get_action(
            scenario["query_len"],
            scenario["cache_hit"],
            scenario["complexity"]
        )
        
        agent.update(
            scenario["query_len"],
            scenario["cache_hit"],
            scenario["complexity"],
            action,
            scenario["latency"]
        )
        
        print()
    
    agent.decay_epsilon()
    
    print("\n" + "="*60)
    print("Agent Statistics:")
    print("="*60)
    stats = agent.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n" + "="*60)
    print("Q-Table Sample:")
    print("="*60)
    for state, actions in list(agent.q_table.items())[:3]:
        print(f"{state}:")
        for action, q_val in actions.items():
            print(f"  {action}: {q_val:.4f}")


if __name__ == "__main__":
    test_rl_agent()
