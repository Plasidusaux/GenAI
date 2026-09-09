# test_sentiment_tool.py

import unittest
from unittest.mock import patch, MagicMock
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from llm_agent.tool_sentiment import SentimentAnalyzerTool


class TestSentimentAnalyzerTool(unittest.TestCase):
    """Тесты для инструмента анализа тональности."""
    
    def setUp(self):
        """Подготовка перед каждым тестом."""
        # Создаем экземпляр инструмента с эвристическим методом (fallback)
        self.tool = SentimentAnalyzerTool(use_local=True)  # Принудительно используем "локальный" режим
        self.tool.sentiment_pipeline = None  # Отключаем локальную модель
        self.tool.use_local = True
        self.tool.api_key = None  # Отключаем API
    
    def test_positive_text_heuristic(self):
        """Тест: позитивный текст должен быть определен как позитивный."""
        result = self.tool.use("Это отличный день! Я очень рад и счастлив!")
        self.assertIn("позитивная", result.lower())
        self.assertIn("уверенность", result.lower())
    
    def test_negative_text_heuristic(self):
        """Тест: негативный текст должен быть определен как негативный."""
        result = self.tool.use("Ужасный день, я ненавижу эту погоду!")
        self.assertIn("негативная", result.lower())
        self.assertIn("уверенность", result.lower())
    
    def test_neutral_text_heuristic(self):
        """Тест: нейтральный текст должен быть определен как нейтральный."""
        result = self.tool.use("Сегодня понедельник. Обычный день недели.")
        self.assertIn("нейтральная", result.lower())
        self.assertIn("уверенность", result.lower())
    
    def test_empty_text(self):
        """Тест: пустой текст должен обработаться без ошибок."""
        result = self.tool.use("")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
    
    def test_english_positive_text(self):
        """Тест: английский позитивный текст."""
        result = self.tool.use("I love this amazing product! It's wonderful!")
        self.assertIn("позитивная", result.lower())
    
    def test_english_negative_text(self):
        """Тест: английский негативный текст."""
        result = self.tool.use("This is terrible, I hate it!")
        self.assertIn("негативная", result.lower())
    
    def test_mixed_sentiment(self):
        """Тест: смешанный текст с преобладающей тональностью."""
        result = self.tool.use("Фильм хороший, но актеры играют плохо.")
        self.assertIn("тональность", result.lower())
    
    def test_analyze_batch(self):
        """Тест: анализ нескольких текстов."""
        texts = [
            "Отличная работа!",
            "Плохой результат",
            "Нейтральный текст"
        ]
        results = self.tool.analyze_batch(texts)
        self.assertEqual(len(results), 3)
        self.assertIsInstance(results, list)
        for result in results:
            self.assertIn("text", result)
            self.assertIn("analysis", result)
    
    def test_heuristic_positive_score(self):
        """Тест: правильный подсчет позитивных слов."""
        result = self.tool.use("хорошо отлично прекрасно")
        self.assertIn("позитивн", result.lower())
        self.assertIn("Найдено позитивных слов", result)


class TestSentimentAnalyzerToolLocal(unittest.TestCase):
    """Тесты для локальной модели (mock)."""
    
    def setUp(self):
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
        self.assertIsInstance(result, str)
    
    @patch('llm_agent.tool_sentiment.SentimentAnalyzerTool._analyze_local')
    def test_analyze_local_mocked(self, mock_analyze):
        """Тест: анализ через локальную модель (мок)."""
        mock_analyze.return_value = "Тональность: позитивная"
        tool = SentimentAnalyzerTool(use_local=True)
        tool.sentiment_pipeline = MagicMock()  # Создаем мок пайплайна
        tool.use_local = True
        result = tool.use("Тест")
        mock_analyze.assert_called_once()
        self.assertEqual(result, "Тональность: позитивная")


class TestSentimentAnalyzerToolAPI(unittest.TestCase):
    """Тесты для Hugging Face API (mock)."""
    
    def setUp(self):
        self.tool = SentimentAnalyzerTool(use_local=False)
        self.tool.api_key = "test_api_key"
        self.tool.use_local = False
    
    @patch('llm_agent.tool_sentiment.requests.post')
    def test_api_success(self, mock_post):
        """Тест: успешный ответ от API."""
        # Настраиваем мок ответа
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = [{"label": "POSITIVE", "score": 0.95}]
        mock_post.return_value = mock_response
        
        result = self.tool.use("I love this!")
        self.assertIn("позитивн", result.lower())
        self.assertIn("95.0%", result)
        mock_post.assert_called_once()
    
    @patch('llm_agent.tool_sentiment.requests.post')
    def test_api_negative(self, mock_post):
        """Тест: негативный результат от API."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = [{"label": "NEGATIVE", "score": 0.89}]
        mock_post.return_value = mock_response
        
        result = self.tool.use("I hate this!")
        self.assertIn("негативн", result.lower())
        self.assertIn("89.0%", result)
    
    @patch('llm_agent.tool_sentiment.requests.post')
    def test_api_error(self, mock_post):
        """Тест: ошибка API."""
        mock_post.side_effect = requests.exceptions.RequestException("API Error")
        
        result = self.tool.use("Тест")
        self.assertIn("ошибк", result.lower())
    
    @patch('llm_agent.tool_sentiment.requests.post')
    def test_api_invalid_response(self, mock_post):
        """Тест: некорректный ответ API."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {}  # Пустой ответ
        mock_post.return_value = mock_response
        
        result = self.tool.use("Тест")
        self.assertIn("не удалось", result.lower())
    
    def test_no_api_key(self):
        """Тест: отсутствие API-ключа."""
        tool = SentimentAnalyzerTool(use_local=False)
        tool.api_key = None
        tool.use_local = False
        result = tool.use("Тест")
        self.assertIsInstance(result, str)


class TestSentimentAnalyzerToolIntegration(unittest.TestCase):
    """Интеграционные тесты (реальное выполнение, если доступно)."""
    
    @unittest.skipIf(os.getenv("HUGGINGFACE_API_KEY") is None, 
                     "HUGGINGFACE_API_KEY не установлен")
    def test_real_api(self):
        """Тест с реальным API (требуется ключ)."""
        tool = SentimentAnalyzerTool(use_local=False)
        result = tool.use("This is a great day!")
        self.assertIn("позитивн", result.lower())
    
    @unittest.skipIf(not _check_transformers_installed(), 
                     "transformers не установлен")
    def test_real_local_model(self):
        """Тест с реальной локальной моделью."""
        tool = SentimentAnalyzerTool(use_local=True)
        result = tool.use("I am very happy today!")
        self.assertIn("позитивн", result.lower())


def _check_transformers_installed():
    """Проверяет, установлена ли библиотека transformers."""
    try:
        import transformers
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    # Запуск тестов с подробным выводом
    unittest.main(verbosity=2)