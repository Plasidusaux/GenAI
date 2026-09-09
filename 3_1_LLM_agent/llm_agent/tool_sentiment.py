# llm_agent/tool_sentiment.py

import requests
from typing import Optional, Dict, List
from decouple import config


class SentimentAnalyzerTool:
    """
    Инструмент для анализа тональности текста.
    Поддерживает три режима:
    1. Hugging Face API (требуется API-ключ)
    2. Локальная модель через transformers (если установлена)
    3. Эвристический анализ (fallback)
    """
    
    name = "sentiment_analyzer"
    description = (
        "Анализирует тональность текста (позитивная, негативная, нейтральная). "
        "Принимает текст для анализа. Возвращает оценку тональности и уверенность."
    )
    
    def __init__(self, use_local: bool = False):
        """
        Инициализирует анализатор тональности.
        
        Args:
            use_local (bool): Если True, использует локальную модель transformers.
                             Если False, использует Hugging Face API.
        """
        self.use_local = use_local
        
        if not use_local:
            # Пытаемся получить API-ключ из .env
            self.api_key = config('HUGGINGFACE_API_KEY', default='')
            if not self.api_key:
                print("⚠️  Внимание: HUGGINGFACE_API_KEY не найден. Использую локальную модель.")
                self.use_local = True
        
        if self.use_local:
            self._load_local_model()
    
    def _load_local_model(self):
        """Загружает локальную модель sentiment analysis."""
        try:
            from transformers import pipeline
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
            print("✅ Локальная модель sentiment analysis загружена")
        except ImportError:
            print("❌ Библиотека transformers не установлена. Установите: pip install transformers")
            self.sentiment_pipeline = None
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            self.sentiment_pipeline = None
    
    def use(self, text: str) -> str:
        """
        Анализирует тональность текста.
        
        Args:
            text (str): Текст для анализа.
            
        Returns:
            str: Результат анализа тональности.
        """
        try:
            print(f"> Анализирую тональность текста: '{text[:50]}...'")
            
            if self.use_local and self.sentiment_pipeline is not None:
                return self._analyze_local(text)
            elif not self.use_local and self.api_key:
                return self._analyze_huggingface_api(text)
            else:
                # Fallback: простая эвристическая оценка
                return self._analyze_heuristic(text)
                
        except Exception as e:
            print(f"> Ошибка при анализе тональности: {e}")
            return f"Произошла ошибка при анализе тональности: {e}"
    
    def _analyze_local(self, text: str) -> str:
        """Анализ тональности с помощью локальной модели."""
        try:
            result = self.sentiment_pipeline(text[:512])  # Ограничиваем длину текста
            
            if result:
                sentiment = result[0]
                label = sentiment['label']
                score = sentiment['score']
                
                # Маппим английские метки на русские
                label_ru = {
                    'POSITIVE': 'позитивная',
                    'NEGATIVE': 'негативная',
                    'NEUTRAL': 'нейтральная'
                }.get(label, label.lower())
                
                return (
                    f"Результат анализа тональности:\n"
                    f"Тональность: {label_ru}\n"
                    f"Уверенность: {score*100:.1f}%"
                )
            else:
                return "Не удалось определить тональность текста."
                
        except Exception as e:
            return f"Ошибка локальной модели: {e}"
    
    def _analyze_huggingface_api(self, text: str) -> str:
        """Анализ тональности через Hugging Face API."""
        try:
            # Используем популярную модель для sentiment analysis
            API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            payload = {"inputs": text[:512]}  # Ограничиваем длину
            
            response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            
            if isinstance(result, list) and result:
                # Формат ответа: [{'label': 'POSITIVE', 'score': 0.999}]
                best_result = max(result, key=lambda x: x['score'])
                label = best_result['label']
                score = best_result['score']
                
                label_ru = {
                    'POSITIVE': 'позитивная',
                    'NEGATIVE': 'негативная',
                    'NEUTRAL': 'нейтральная'
                }.get(label, label.lower())
                
                return (
                    f"Результат анализа тональности:\n"
                    f"Тональность: {label_ru}\n"
                    f"Уверенность: {score*100:.1f}%"
                )
            else:
                return "Не удалось получить результат от Hugging Face API."
                
        except requests.exceptions.RequestException as e:
            return f"Ошибка при запросе к Hugging Face API: {e}"
        except (KeyError, IndexError, ValueError) as e:
            return f"Ошибка обработки ответа от Hugging Face API: {e}"
    
    def _analyze_heuristic(self, text: str) -> str:
        """
        Простой эвристический анализ тональности на основе словарей.
        Используется как fallback, если нет API и локальной модели.
        """
        # Расширенный словарь позитивных слов (русские + английские)
        positive_words = [
            # Русские позитивные слова
            "хорошо", "отлично", "прекрасно", "замечательно", "люблю", "нравится",
            "радость", "счастье", "успех", "победа", "позитив", "класс", "супер",
            "великолепно", "восхитительно", "превосходно", "потрясающе", "шикарно",
            "здорово", "круто", "awesome", "wonderful", "fantastic", "amazing",
            "love", "like", "happy", "joy", "excellent", "great", "good", "best",
            "perfect", "beautiful", "nice", "positive", "amazing", "brilliant",
            "success", "win", "victory", "enjoy", "satisfied", "pleased",
            "рад", "доволен", "счастлив", "весело", "интересно", "увлекательно",
            "впечатляюще", "великий", "лучший", "отличный", "прекрасный",
            "замечательный", "позитивный", "успешный", "победный", "радостный",
            "веселый", "счастливый", "довольный", "вдохновляющий", "мотивирующий"
        ]
        
        # Расширенный словарь негативных слов (русские + английские)
        negative_words = [
            # Русские негативные слова
            "плохо", "ужасно", "отвратительно", "ненавижу", "не нравится", "грусть",
            "печаль", "провал", "поражение", "негатив", "fail", "bad", "terrible",
            "hate", "sad", "angry", "worst", "awful", "horrible", "disappointed",
            "ужасный", "плохой", "худший", "отвратный", "мерзкий", "гадкий",
            "негативный", "печальный", "грустный", "злой", "расстроен", "разочарован",
            "обидно", "досадно", "неприятно", "провальный", "убыточный",
            "problem", "issue", "error", "failure", "mistake", "crisis",
            "проблема", "ошибка", "кризис", "беда", "катастрофа", "несчастье",
            "плохой", "ужасный", "отвратительный", "мерзкий", "омерзительный",
            "неудача", "фиаско", "крах", "провал", "позор", "стыд",
            "холодно", "жарко", "тяжело", "трудно", "сложно", "больно"
        ]
        
        text_lower = text.lower()
        
        # Считаем количество позитивных и негативных слов
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        print(f"> Найдено позитивных слов: {positive_count}, негативных: {negative_count}")
        
        if positive_count > negative_count:
            tone = "позитивная"
            confidence = positive_count / (positive_count + negative_count) if (positive_count + negative_count) > 0 else 0.5
        elif negative_count > positive_count:
            tone = "негативная"
            confidence = negative_count / (positive_count + negative_count) if (positive_count + negative_count) > 0 else 0.5
        else:
            tone = "нейтральная"
            confidence = 0.5
        
        return (
            f"Результат анализа тональности (эвристический):\n"
            f"Тональность: {tone}\n"
            f"Уверенность: {confidence*100:.1f}%\n"
            f"Найдено позитивных слов: {positive_count}, негативных: {negative_count}"
        )
    
    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """
        Анализирует тональность для нескольких текстов.
        
        Args:
            texts (List[str]): Список текстов для анализа.
            
        Returns:
            List[Dict]: Список результатов анализа.
        """
        results = []
        for text in texts:
            result = self.use(text)
            results.append({
                "text": text,
                "analysis": result
            })
        return results