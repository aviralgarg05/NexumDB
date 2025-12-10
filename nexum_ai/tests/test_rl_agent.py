"""
Unit tests for rl_agent.py - RL training loop, state/action handling
"""

import os
import tempfile
from nexum_ai.rl_agent import QLearningAgent


class TestQLearningAgent:
    """Test suite for QLearningAgent class"""
    
    def test_initialization(self):
        """Test QLearningAgent initialization"""
        agent = QLearningAgent(
            learning_rate=0.2,
            discount_factor=0.8,
            epsilon=0.3
        )
        assert agent.learning_rate == 0.2
        assert agent.discount_factor == 0.8
        assert agent.epsilon == 0.3
        assert agent.q_table == {}
        assert agent.actions == ["force_cache", "bypass_cache", "normal"]
    
    def test_get_state_key(self):
        """Test state key generation"""
        agent = QLearningAgent()
        
        state_key = agent._get_state_key(
            query_length=25,
            cache_hit=True,
            complexity=5
        )
        
        assert isinstance(state_key, str)
        assert "state_" in state_key
        assert "_1_" in state_key  # cache_hit=True
    
    def test_get_state_key_bucketing(self):
        """Test query length bucketing in state key"""
        agent = QLearningAgent()
        
        # Queries with similar lengths should map to same bucket
        state1 = agent._get_state_key(25, False, 3)
        state2 = agent._get_state_key(28, False, 3)
        assert state1 == state2
        
        # Very different lengths should map to different buckets
        state3 = agent._get_state_key(5, False, 3)
        assert state1 != state3
    
    def test_get_action_exploration(self):
        """Test action selection during exploration"""
        agent = QLearningAgent(epsilon=1.0)  # Always explore
        
        action = agent.get_action(
            query_length=25,
            cache_hit=False,
            complexity=3
        )
        
        assert action in agent.actions
    
    def test_get_action_exploitation(self):
        """Test action selection during exploitation"""
        agent = QLearningAgent(epsilon=0.0)  # Never explore
        
        # Set up Q-values
        state = agent._get_state_key(25, False, 3)
        agent.q_table[state] = {
            "force_cache": 0.5,
            "bypass_cache": 1.5,
            "normal": 0.3
        }
        
        action = agent.get_action(25, False, 3)
        assert action == "bypass_cache"  # Highest Q-value
    
    def test_get_action_new_state(self):
        """Test action selection for unseen state"""
        agent = QLearningAgent(epsilon=0.0)
        
        action = agent.get_action(
            query_length=50,
            cache_hit=True,
            complexity=7
        )
        
        assert action in agent.actions
        state = agent._get_state_key(50, True, 7)
        assert state in agent.q_table
    
    def test_update_q_value(self):
        """Test Q-value update"""
        agent = QLearningAgent(learning_rate=0.5, discount_factor=0.9)
        
        agent.update(
            query_length=25,
            cache_hit=False,
            complexity=3,
            action="normal",
            latency_ms=10.0
        )
        
        state = agent._get_state_key(25, False, 3)
        assert state in agent.q_table
        assert "normal" in agent.q_table[state]
        assert len(agent.training_history) == 1
    
    def test_update_multiple_times(self):
        """Test multiple Q-value updates"""
        agent = QLearningAgent(learning_rate=0.5)
        
        # First update
        agent.update(25, False, 3, "normal", 10.0)
        state = agent._get_state_key(25, False, 3)
        q_value_1 = agent.q_table[state]["normal"]
        
        # Second update with better latency
        agent.update(25, False, 3, "normal", 5.0)
        q_value_2 = agent.q_table[state]["normal"]
        
        # Q-value should improve with better latency (or stay same if converged)
        assert q_value_2 >= q_value_1
    
    def test_calculate_reward_low_latency(self):
        """Test reward calculation for low latency"""
        agent = QLearningAgent()
        
        reward = agent._calculate_reward(latency_ms=1.0, cache_hit=False)
        assert reward < 0  # Negative but small penalty
        
        reward_cached = agent._calculate_reward(latency_ms=0.5, cache_hit=True)
        assert reward_cached > reward  # Cache hit bonus
    
    def test_calculate_reward_high_latency(self):
        """Test reward calculation for high latency"""
        agent = QLearningAgent()
        
        reward = agent._calculate_reward(latency_ms=100.0, cache_hit=False)
        assert reward <= -1.0  # Large penalty for slow queries
    
    def test_decay_epsilon(self):
        """Test epsilon decay"""
        agent = QLearningAgent(epsilon=0.5, epsilon_decay=0.9)
        initial_epsilon = agent.epsilon
        
        agent.decay_epsilon()
        
        assert agent.epsilon < initial_epsilon
        assert agent.epsilon == initial_epsilon * 0.9
        assert agent.episode_count == 1
    
    def test_decay_epsilon_minimum(self):
        """Test epsilon doesn't decay below minimum"""
        agent = QLearningAgent(epsilon=0.02, epsilon_decay=0.5)
        agent.epsilon_min = 0.01
        
        agent.decay_epsilon()
        
        assert agent.epsilon >= agent.epsilon_min
    
    def test_get_stats(self):
        """Test agent statistics"""
        agent = QLearningAgent()
        
        # Perform some updates
        agent.update(25, False, 3, "normal", 10.0)
        agent.update(30, True, 5, "force_cache", 0.5)
        
        stats = agent.get_stats()
        
        assert 'q_table_size' in stats
        assert 'total_updates' in stats
        assert 'epsilon' in stats
        assert 'episodes' in stats
        assert 'avg_reward' in stats
        assert stats['total_updates'] == 2
    
    def test_save_state(self):
        """Test saving agent state"""
        agent = QLearningAgent()
        agent.update(25, False, 3, "normal", 10.0)
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Without joblib installed, should print warning
            agent.save_state(tmp_path)
            # Test passes if no exception raised
            assert True
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_load_state(self):
        """Test loading agent state"""
        agent = QLearningAgent()
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
            # Create the file so it exists
            with open(tmp_path, 'w') as f:
                f.write("dummy")
        
        try:
            # Without joblib installed, should print warning
            agent.load_state(tmp_path)
            # Test passes if no exception raised
            assert True
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_load_state_nonexistent_file(self):
        """Test loading state when file doesn't exist"""
        agent = QLearningAgent()
        initial_epsilon = agent.epsilon
        
        agent.load_state("/nonexistent/path.pkl")
        
        # Should not crash, should keep initial values
        assert agent.epsilon == initial_epsilon
        assert agent.q_table == {}


class TestQLearningAgentIntegration:
    """Integration tests for QLearningAgent"""
    
    def test_training_loop(self):
        """Test a complete training loop"""
        agent = QLearningAgent(learning_rate=0.1, epsilon=0.3)
        
        # Simulate training episodes
        scenarios = [
            (25, False, 3, 10.0),  # Short query, no cache, medium complexity
            (25, True, 3, 0.5),    # Same query, cache hit
            (50, False, 7, 25.0),  # Long query, no cache, high complexity
            (50, True, 7, 1.0),    # Same query, cache hit
        ]
        
        for query_len, cache_hit, complexity, latency in scenarios:
            action = agent.get_action(query_len, cache_hit, complexity)
            agent.update(query_len, cache_hit, complexity, action, latency)
        
        assert len(agent.training_history) == 4
        assert len(agent.q_table) > 0
    
    def test_learning_from_experience(self):
        """Test that agent learns from repeated experiences"""
        agent = QLearningAgent(learning_rate=0.5, epsilon=0.0)
        
        # Repeatedly show that cache hits are faster
        for _ in range(10):
            agent.update(25, True, 3, "force_cache", 0.5)  # Fast
            agent.update(25, False, 3, "bypass_cache", 10.0)  # Slow
        
        # Agent should prefer force_cache for this state
        state = agent._get_state_key(25, True, 3)
        if state in agent.q_table:
            q_force = agent.q_table[state].get("force_cache", 0)
            q_bypass = agent.q_table[state].get("bypass_cache", 0)
            assert q_force > q_bypass
