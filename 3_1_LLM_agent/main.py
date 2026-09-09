# main.py

from llm_agent.core_v2 import LLMAgent
from llm_agent.tool_sentiment import SentimentAnalyzerTool

def main():
    """Основная функция для запуска агента."""
    print("Простой LLM-агент с инструментами")
    print("Инструменты: 'Калькулятор', 'Поиск в DuckDuckGo', 'Анализ тональности'")
    print("-" * 70)

    #agent = LLMAgent(model = "qwen/qwen3-next-80b-a3b-instruct:free")

    agent = LLMAgent(local = True, ollama_model = "qwen3.5:0.8b") #ollama_base_url = "10.10.34.24:5678"

    #agent = LLMAgent(model = "gpt-5.4-mini")
    #agent = LLMAgent(model = "grok4.1-fast")
    
    # Примеры запросов
    queries = [
        # "Сколько будет (5 + 3) * 2?",
        # "Какая погода в Москве?",
        # "Кто выиграл последний матч Спартак-Динамо?",
        # "Проанализируй тональность этого отзыва: 'Это ужасный фильм, полная трата времени!'",
        # "Проанализируй тональность этого текста: 'Отличная работа! Я очень доволен результатом!'",
        # "Какой настрой у этого сообщения: 'Все отлично, продолжаем работать!'",
        # "Определи тональность отзыва: 'Сервис работает хорошо, но цена немного высоковата.'",
        # "Сколько будет 15 * 3 + 7? И проанализируй тональность отзыва: 'Мне очень нравится этот продукт!'",
        # "Найди информацию о последних новостях в мире технологий",
        # "Извлеки информацию из PDF файла по пути '/path/to/document.pdf'"
    ]
    
    # Выбираем запрос для демонстрации (можно менять)
    query = "Проанализируй тональность этого отзыва: 'Это отличный фильм! Актеры играют просто великолепно, сюжет захватывает с первой минуты!'"

    print(f"Ваш запрос: {query}")
    print("-" * 70)

    response = agent.process_query(query)

    print("\n" + "=" * 70)
    print("Финальный ответ агента:\n")
    print(response)
    print("=" * 70)

    # Дополнительно: демонстрация прямого использования инструмента
    print("\n" + "=" * 70)
    print("Демонстрация прямого использования SentimentAnalyzerTool:")
    print("=" * 70)
    
    # Создаем инструмент для прямого использования
    sentiment_tool = SentimentAnalyzerTool(use_local=False)
    
    # Примеры текстов для анализа
    test_texts = [
        "Отличный день! Я счастлив и рад!",
        "Это ужасно, я полностью разочарован!",
        "Обычный день, ничего особенного.",
        "I love this amazing product!",
        "This is the worst experience ever!"
    ]
    
    for text in test_texts:
        result = sentiment_tool.use(text)
        print(f"\n Текст: {text}")
        print(f" Результат: {result}")
        print("-" * 40)

def demo_interactive():
    """Демонстрация интерактивного режима."""
    print("Интерактивный режим с SentimentAnalyzerTool")
    print("-" * 50)
    
    agent = LLMAgent(local=True, ollama_model="qwen3.5:0.8b")
    sentiment_tool = SentimentAnalyzerTool(use_local=False)
    
    while True:
        print("\n" + "=" * 50)
        print("Выберите действие:")
        print("1. Анализ тональности через агента")
        print("2. Анализ тональности напрямую")
        print("3. Выход")
        
        choice = input("Ваш выбор (1-3): ").strip()
        
        if choice == "1":
            text = input("\nВведите текст для анализа: ")
            query = f"Проанализируй тональность этого текста: '{text}'"
            response = agent.process_query(query)
            print(f"\nОтвет агента: {response}")
        
        elif choice == "2":
            text = input("\nВведите текст для анализа: ")
            result = sentiment_tool.use(text)
            print(f"\nРезультат анализа: {result}")
        
        elif choice == "3":
            print("До свидания!")
            break
        
        else:
            print("Неверный выбор. Попробуйте снова.")

if __name__ == "__main__":
    # Выберите режим:
    # main() - запуск с одним запросом
    # demo_interactive() - интерактивный режим
    
    # Можно переключать режимы:
    main()
    # demo_interactive()