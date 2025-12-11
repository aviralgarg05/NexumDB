"""
Unit tests for translator.py - NL to SQL translation
"""

import os
from unittest.mock import Mock, patch
from nexum_ai.translator import NLTranslator


class TestNLTranslator:
    """Test suite for NLTranslator class"""
    
    def test_initialization_no_model(self):
        """Test NLTranslator initialization without model"""
        with patch.dict(os.environ, {}, clear=True):
            with patch('nexum_ai.translator.Llama', None):  # Force Llama to be None
                with patch('os.path.exists', return_value=False):  # No model file exists
                    with patch('nexum_ai.model_manager.ModelManager') as mock_manager:
                        mock_manager_instance = Mock()
                        mock_manager_instance.ensure_model.return_value = None
                        mock_manager.return_value = mock_manager_instance
                        
                        translator = NLTranslator(model_path=None)  # Explicitly pass None
                        assert translator.model is None
    
    @patch('nexum_ai.translator.Llama')
    def test_initialization_with_model(self, mock_llama):
        """Test NLTranslator initialization with model"""
        mock_model = Mock()
        mock_llama.return_value = mock_model
        
        with patch('os.path.exists', return_value=True):
            translator = NLTranslator(model_path="/fake/model.gguf")
            assert translator.model is not None
    
    def test_fallback_translation_all_users(self, sample_schema):
        """Test fallback translation for 'all users' query"""
        translator = NLTranslator()
        translator.model = None  # Force fallback
        
        sql = translator.translate("Show me all users", sample_schema)
        assert "SELECT * FROM users" in sql
    
    def test_fallback_translation_users_by_name(self, sample_schema):
        """Test fallback translation for users by name"""
        translator = NLTranslator()
        translator.model = None
        
        sql = translator.translate("Show me all users named Alice", sample_schema)
        sql_lower = sql.lower()
        assert "select * from users" in sql_lower
        assert "where name = 'alice'" in sql_lower  # Case insensitive check
    
    def test_fallback_translation_all_products(self, sample_schema):
        """Test fallback translation for 'all products' query"""
        translator = NLTranslator()
        translator.model = None
        
        sql = translator.translate("Show me all products", sample_schema)
        assert "SELECT * FROM products" in sql
    
    def test_fallback_translation_products_price_less_than(self, sample_schema):
        """Test fallback translation for products with price filter"""
        translator = NLTranslator()
        translator.model = None
        
        sql = translator.translate("Show products with price less than 100", sample_schema)
        assert "SELECT * FROM products" in sql
        assert "WHERE price < 100" in sql
    
    def test_fallback_translation_products_price_more_than(self, sample_schema):
        """Test fallback translation for products with price filter (greater than)"""
        translator = NLTranslator()
        translator.model = None
        
        sql = translator.translate("Show products with price more than 500", sample_schema)
        assert "SELECT * FROM products" in sql
        assert "WHERE price > 500" in sql
    
    def test_fallback_translation_unknown_query(self, sample_schema):
        """Test fallback translation for unknown query pattern"""
        translator = NLTranslator()
        translator.model = None
        
        sql = translator.translate("Some random query", sample_schema)
        assert "SELECT * FROM table" in sql
    
    def test_build_prompt(self, sample_schema):
        """Test prompt building"""
        translator = NLTranslator()
        
        prompt = translator._build_prompt("Show me all users", sample_schema)
        
        assert "Show me all users" in prompt
        assert "SQL" in prompt or "sql" in prompt.lower()
        assert sample_schema in prompt
    
    def test_build_prompt_no_schema(self):
        """Test prompt building without schema"""
        translator = NLTranslator()
        
        prompt = translator._build_prompt("Show me all users", "")
        
        assert "Show me all users" in prompt
        assert "No schema provided" in prompt
    
    def test_clean_sql_basic(self):
        """Test SQL cleaning with basic query"""
        translator = NLTranslator()
        
        sql = translator._clean_sql("  SELECT * FROM users  ")
        assert sql == "SELECT * FROM users"
    
    def test_clean_sql_with_prefix(self):
        """Test SQL cleaning with 'sql' prefix"""
        translator = NLTranslator()
        
        sql = translator._clean_sql("sql SELECT * FROM users")
        assert sql == "SELECT * FROM users"
    
    def test_clean_sql_with_code_block(self):
        """Test SQL cleaning with markdown code block"""
        translator = NLTranslator()
        
        sql = translator._clean_sql("```SELECT * FROM users```")
        assert sql == "SELECT * FROM users"
    
    def test_clean_sql_with_semicolon(self):
        """Test SQL cleaning removes trailing semicolon"""
        translator = NLTranslator()
        
        sql = translator._clean_sql("SELECT * FROM users; SELECT * FROM products")
        assert sql == "SELECT * FROM users"
        assert "products" not in sql
    
    @patch('nexum_ai.translator.Llama')
    def test_translate_with_model(self, mock_llama, sample_schema):
        """Test translation with LLM model"""
        mock_model = Mock()
        mock_model.return_value = {
            'choices': [{'text': 'SELECT * FROM users WHERE age > 25'}]
        }
        
        with patch('os.path.exists', return_value=True):
            translator = NLTranslator(model_path="/fake/model.gguf")
            translator.model = mock_model
            
            sql = translator.translate("Show users older than 25", sample_schema)
            
            assert "SELECT" in sql
            mock_model.assert_called_once()
    
    @patch('nexum_ai.translator.Llama')
    def test_translate_model_error_fallback(self, mock_llama, sample_schema):
        """Test translation falls back when model errors"""
        mock_model = Mock()
        mock_model.side_effect = Exception("Model error")
        
        with patch('os.path.exists', return_value=True):
            translator = NLTranslator(model_path="/fake/model.gguf")
            translator.model = mock_model
            
            sql = translator.translate("Show me all users", sample_schema)
            
            # Should fall back to rule-based translation
            assert "SELECT * FROM users" in sql


class TestNLTranslatorIntegration:
    """Integration tests for NLTranslator"""
    
    def test_multiple_translations(self, sample_schema):
        """Test multiple translations in sequence"""
        translator = NLTranslator()
        translator.model = None  # Use fallback
        
        queries = [
            ("Show me all users", "users"),
            ("Show me all products", "products"),
            ("Show users named Bob", "bob"),  # Lowercase for case-insensitive match
        ]
        
        for nl_query, expected in queries:
            sql = translator.translate(nl_query, sample_schema)
            assert expected in sql.lower()  # Case insensitive check
    
    def test_case_insensitive_translation(self, sample_schema):
        """Test that translation is case-insensitive"""
        translator = NLTranslator()
        translator.model = None
        
        sql1 = translator.translate("SHOW ME ALL USERS", sample_schema)
        sql2 = translator.translate("show me all users", sample_schema)
        
        assert sql1.upper() == sql2.upper()
