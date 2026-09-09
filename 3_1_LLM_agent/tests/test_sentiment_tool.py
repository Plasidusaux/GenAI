# 3_1_LLM_agent/tests/test_sentiment_tool.py

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import pytest

# Добавляем путь к родительской директории для импорта модулей
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from llm_agent.tool_sentiment import SentimentAnalyzerTool


# Вспомогательная функция для проверки установки transformers
def _check_transformers_installed():
    """Проверяет, установлена ли библиотека transformers."""
    try:
        import transformers
        return True
    except ImportError:
        return False


@pytest.mark.unit
class TestSentimentAnalyzerToolHeuristic:
    """Тесты эвристического анализа тональности."""
    
    @pytest.fixture(autouse=True)
    def setup_tool(self):
        """Подготовка инструмента перед каждым тестом."""
        self.tool = SentimentAnalyzerTool(use_local=True)
        self.tool.sentiment_pipeline = None  # Отключаем локальную модель
        self.tool.use_local = True
        self.tool.api_key = None  # Отключаем API
    
    def test_positive_text_heuristic(self):
        """Тест: позитивный текст должен быть определен как позитивный."""
        result = self.tool.use("Это отличный день! Я очень рад и счастлив!")
        assert "позитивная" in result.lower()
        assert "уверенность" in result.lower()
    
    def test_negative_text_heuristic(self):
        """Тест: негативный текст должен быть определен как негативный."""
        result = self.tool.use("Ужасный день, я ненавижу эту погоду!")
        assert "негативная" in result.lower()
        assert "уверенность" in result.lower()
    
    def test_neutral_text_heuristic(self):
        """Тест: нейтральный текст должен быть определен как нейтральный."""
        result = self.tool.use("Сегодня понедельник. Обычный день недели.")
        assert "нейтральная" in result.lower()
        assert "уверенность" in result.lower()
    
    def test_empty_text(self):
        """Тест: пустой текст должен обработаться без ошибок."""
        result = self.tool.use("")
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_english_positive_text(self):
        """Тест: английский позитивный текст."""
        result = self.tool.use("I love this amazing product! It's wonderful!")
        assert "позитивная" in result.lower()
    
    def test_english_negative_text(self):
        """Тест: английский негативный текст."""
        result = self.tool.use("This is terrible, I hate it!")
        assert "негативная" in result.lower()
    
    def test_mixed_sentiment(self):
        """Тест: смешанный текст с преобладающей тональностью."""
        result = self.tool.use("Фильм хороший, но актеры играют плохо.")
        assert "тональность" in result.lower()
    
    def test_analyze_batch(self):
        """Тест: анализ нескольких текстов."""
        texts = [
            "Отличная работа!",
            "Плохой результат",
            "Нейтральный текст"
        ]
        results = self.tool.analyze_batch(texts)
        assert len(results) == 3
        assert isinstance(results, list)
        for result in results:
            assert "text" in result
            assert "analysis" in result
    
    def test_heuristic_positive_score(self):
        """Тест: правильный подсчет позитивных слов."""
        result = self.tool.use("хорошо отлично прекрасно")
        assert "позитивная" in result.lower()
        assert "Найдено позитивных слов" in result


@pytest.mark.unit
class TestSentimentAnalyzerToolLocal:
    """Тесты локальной модели (mock)."""
    
    @pytest.fixture(autouse=True)
    def setup_tool(self):
        """Подготовка инструмента."""
        self.tool = SentimentAnalyzerTool(use_local=True)
    
    @patch('llm_agent.tool_sentiment.SentimentAnalyzerTool._load_local_model')
    def test_local_model_loading(self, mock_load):
        """Тест: загрузка локальной модели."""
        self.tool.use_local = True
        self.tool._load_local_model()
        mock_load.assert_called_once()
    
    def test_local_model_not_loaded(self):
        """Тест: если модель не загружена, используем эвристику."""
        tool = SentimentAnalyzerTool(use_local=True)
        tool.sentiment_pipeline = None
        result = tool.use("Это тестовый текст")
        assert isinstance(result, str)
    
    @patch('llm_agent.tool_sentiment.SentimentAnalyzerTool._analyze_local')
    def test_analyze_local_mocked(self, mock_analyze):
        """Тест: анализ через локальную модель (мок)."""
        mock_analyze.return_value = "Тональность: позитивная"
        tool = SentimentAnalyzerTool(use_local=True)
        tool.sentiment_pipeline = MagicMock()
        tool.use_local = True
        result = tool.use("Тест")
        mock_analyze.assert_called_once()
        assert result == "Тональность: позитивная"


@pytest.mark.unit
class TestSentimentAnalyzerToolAPI:
    """Тесты для Hugging Face API (mock)."""
    
    @pytest.fixture(autouse=True)
    def setup_tool(self):
        """Подготовка инструмента."""
        self.tool = SentimentAnalyzerTool(use_local=False)
        self.tool.api_key = "test_api_key"
        self.tool.use_local = False
    
    @patch('llm_agent.tool_sentiment.requests.post')
    def test_api_success(self, mock_post):
        """Тест: успешный ответ от API."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = [{"label": "POSITIVE", "score": 0.95}]
        mock_post.return_value = mock_response
        
        result = self.tool.use("I love this!")
        assert "позитивн" in result.lower()
        assert "95.0%" in result
        mock_post.assert_called_once()
    
    @patch('llm_agent.tool_sentiment.requests.post')
    def test_api_negative(self, mock_post):
        """Тест: негативный результат от API."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = [{"label": "NEGATIVE", "score": 0.89}]
        mock_post.return_value = mock_response
        
        result = self.tool.use("I hate this!")
        assert "негативн" in result.lower()
        assert "89.0%" in result
    
    @patch('llm_agent.tool_sentiment.requests.post')
    def test_api_error(self, mock_post):
        """Тест: ошибка API."""
        import requests
        mock_post.side_effect = requests.exceptions.RequestException("API Error")
        
        result = self.tool.use("Тест")
        assert "ошибк" in result.lower()
    
    @patch('llm_agent.tool_sentiment.requests.post')
    def test_api_invalid_response(self, mock_post):
        """Тест: некорректный ответ API."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {}
        mock_post.return_value = mock_response
        
        result = self.tool.use("Тест")
        assert "не удалось" in result.lower()
    
    def test_no_api_key(self):
        """Тест: отсутствие API-ключа."""
        tool = SentimentAnalyzerTool(use_local=False)
        tool.api_key = None
        tool.use_local = False
        result = tool.use("Тест")
        assert isinstance(result, str)


@pytest.mark.integration
class TestSentimentAnalyzerToolIntegration:
    """Интеграционные тесты (реальное выполнение)."""
    
    @pytest.mark.skipif(os.getenv("HUGGINGFACE_API_KEY") is None, 
                       reason="HUGGINGFACE_API_KEY не установлен")
    def test_real_api(self):
        """Тест с реальным API (требуется ключ)."""
        tool = SentimentAnalyzerTool(use_local=False)
        result = tool.use("This is a great day!")
        assert "позитивн" in result.lower()
    
    @pytest.mark.skipif(not _check_transformers_installed(), 
                       reason="transformers не установлен")
    def test_real_local_model(self):
        """Тест с реальной локальной моделью."""
        tool = SentimentAnalyzerTool(use_local=True)
        result = tool.use("I am very happy today!")
        assert "позитивн" in result.lower()


# Если используем pytest, main не нужен
if __name__ == "__main__":
    pytest.main([__file__, "-v"])