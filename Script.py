from transformers import GPT2Tokenizer, GPT2LMHeadModel, OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
import torch
import warnings
warnings.filterwarnings("ignore")

class GPTChat:
    def __init__(self, model_type="gpt2"):
        """
        Инициализация чат-бота
        model_type: "gpt1", "gpt2", или "gpt2-xl"
        """
        self.model_type = model_type
        
        if model_type == "gpt1":
            print("🤖 Загружаю GPT-1...")
            self.tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")
            self.model = OpenAIGPTLMHeadModel.from_pretrained("openai-gpt")
        elif model_type == "gpt2":
            print("🤖 Загружаю GPT-2 Medium...")
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
            self.model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
            # Устанавливаем pad_token для GPT-2
            self.tokenizer.pad_token = self.tokenizer.eos_token
        elif model_type == "gpt2-xl":
            print("🤖 Загружаю GPT-2 XL (это может занять некоторое время)...")
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
            self.model = GPT2LMHeadModel.from_pretrained("gpt2-xl")
            # Устанавливаем pad_token для GPT-2
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()
        
        # Настройки генерации в зависимости от модели
        if model_type == "gpt2-xl":
            self.max_length = 1500  # Увеличено для XL модели
            self.temperature = 0.8
            self.top_k = 50
            self.top_p = 0.9
        else:
            self.max_length = 1000
            self.temperature = 0.8
            self.top_k = 50
            self.top_p = 0.9
        
        print(f"✅ {model_type.upper()} готов к общению!")
        self.show_commands()
        
        # История разговора
        self.conversation_history = ""

    def show_commands(self):
        print("💡 Команды:")
        print("   /settings - изменить настройки")
        print("   /clear - очистить историю")
        print("   /info - информация о модели")
        print("   /quit или /exit - выйти")
        print("=" * 50)

    def get_model_info(self):
        """Возвращает информацию о модели"""
        if self.model_type == "gpt1":
            return """
📊 GPT-1 (2018):
• Параметры: 117M
• Слоев: 12
• Размер эмбеддингов: 768
• Контекст: 512 токенов
• Особенности: Первая модель семейства, базовая архитектура Transformer
• Обучение: Несуперевизорное на корпусе BookCorpus
            """
        elif self.model_type == "gpt2":
            return """
📊 GPT-2 Medium (2019):
• Параметры: 355M
• Слоев: 24
• Размер эмбеддингов: 1024
• Контекст: 1024 токена
• Особенности: Средняя версия GPT-2, лучшее качество чем Small
• Обучение: 40GB текста из интернета (WebText)
            """
        elif self.model_type == "gpt2-xl":
            return """
📊 GPT-2 XL (2019):
• Параметры: 1.5B
• Слоев: 48
• Размер эмбеддингов: 1600
• Контекст: 1024 токена
• Особенности: Самая большая публично доступная версия GPT-2
• Обучение: 40GB текста из интернета (WebText)
• Требования: ~6GB видеопамяти или оперативной памяти
            """

    def generate_response(self, user_input):
        """Генерирует ответ на основе ввода пользователя"""
        try:
            # Формируем промпт с историей
            if self.conversation_history:
                prompt = f"{self.conversation_history}\nHuman: {user_input}\nAssistant:"
            else:
                prompt = f"Human: {user_input}\nAssistant:"
            
            # Настройки токенизации в зависимости от модели
            if self.model_type == "gpt1":
                max_input_length = 512
                max_context = 400
            elif self.model_type == "gpt2":
                max_input_length = 800
                max_context = 600
            else:  # gpt2-xl
                max_input_length = 900
                max_context = 700
            
            # Токенизируем с attention_mask
            encoded = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=max_input_length
            )
            
            inputs = encoded['input_ids']
            attention_mask = encoded.get('attention_mask', None)
            
            # Ограничиваем длину контекста
            if inputs.size(1) > max_context:
                inputs = inputs[:, -max_context:]
                if attention_mask is not None:
                    attention_mask = attention_mask[:, -max_context:]
            
            # Очищаем память перед генерацией
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Настройки генерации в зависимости от модели
            if self.model_type == "gpt1":
                max_new_tokens = 150
            elif self.model_type == "gpt2":
                max_new_tokens = 200
            else:  # gpt2-xl
                max_new_tokens = 250
            
            # Генерируем с исправленными параметрами
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=2,
                    repetition_penalty=1.1,
                )
            
            # Декодируем полный ответ
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Извлекаем только ответ бота
            prompt_text = self.tokenizer.decode(inputs[0], skip_special_tokens=True)
            bot_response = full_response[len(prompt_text):].strip()
            
            # Очищаем ответ от лишних частей
            for separator in ["\nHuman:", "\nAssistant:", "\n\n", "Human:", "Assistant:"]:
                if separator in bot_response:
                    bot_response = bot_response.split(separator)[0].strip()
            
            # Удаляем кавычки если ответ в них обернут
            if bot_response.startswith('"') and bot_response.endswith('"'):
                bot_response = bot_response[1:-1].strip()
            
            # Обрезаем ответ до максимальной длины
            if len(bot_response) > self.max_length:
                truncated = bot_response[:self.max_length]
                # Ищем последнее завершение предложения
                last_sentence_end = max(
                    truncated.rfind('.'),
                    truncated.rfind('!'),
                    truncated.rfind('?')
                )
                if last_sentence_end > 50:  # Минимальная длина ответа
                    bot_response = truncated[:last_sentence_end + 1]
                else:
                    bot_response = truncated.rstrip() + "..."
            
            # Очищаем память после генерации
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return bot_response if bot_response else "I'm not sure how to respond to that. Could you try rephrasing?"
            
        except Exception as e:
            # Очищаем память при ошибке
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return f"❌ Ошибка генерации: {str(e)}"

    def show_settings(self):
        """Показывает текущие настройки"""
        print(f"\n⚙️ Текущие настройки ({self.model_type.upper()}):")
        print(f"   Максимальная длина ответа: {self.max_length} символов")
        print(f"   Температура (креативность): {self.temperature}")
        print(f"   Top-k сэмплирование: {self.top_k}")
        print(f"   Top-p сэмплирование: {self.top_p}")
        print()

    def change_settings(self):
        """Позволяет изменить настройки"""
        self.show_settings()
        
        try:
            new_temp = input(f"Новая температура (0.1-2.0, текущая {self.temperature:.1f}): ")
            if new_temp.strip():
                self.temperature = max(0.1, min(2.0, float(new_temp)))
            
            max_possible_length = 2500 if self.model_type == "gpt2-xl" else 2000
            new_length = input(f"Новая максимальная длина (50-{max_possible_length}, текущая {self.max_length}): ")
            if new_length.strip():
                self.max_length = max(50, min(max_possible_length, int(new_length)))
            
            new_top_k = input(f"Новый top-k (1-100, текущий {self.top_k}): ")
            if new_top_k.strip():
                self.top_k = max(1, min(100, int(new_top_k)))
            
            new_top_p = input(f"Новый top-p (0.1-1.0, текущий {self.top_p:.1f}): ")
            if new_top_p.strip():
                self.top_p = max(0.1, min(1.0, float(new_top_p)))
            
            print("✅ Настройки обновлены!")
            self.show_settings()
            
        except ValueError:
            print("❌ Неверный формат. Настройки не изменены.")

    def run_chat(self):
        """Основной цикл чата"""
        while True:
            try:
                user_input = input(f"\n👤 Вы: ").strip()
                
                if not user_input:
                    continue
                
                # Обрабатываем команды
                if user_input.lower() in ['/quit', '/exit', 'quit', 'exit']:
                    print("👋 До свидания!")
                    break
                
                elif user_input.lower() == '/clear':
                    self.conversation_history = ""
                    print("🗑️ История очищена!")
                    continue
                
                elif user_input.lower() == '/settings':
                    self.change_settings()
                    continue
                
                elif user_input.lower() == '/info':
                    print(self.get_model_info())
                    continue
                
                elif user_input.lower() == '/help':
                    self.show_commands()
                    continue
                
                # Генерируем ответ
                print(f"🤖 {self.model_type.upper()} думает...", end="", flush=True)
                response = self.generate_response(user_input)
                print(f"\r🤖 {self.model_type.upper()}: {response}")
                
                # Обновляем историю
                self.conversation_history += f"\nHuman: {user_input}\nAssistant: {response}"
                
                # Ограничиваем историю (увеличиваем лимиты для XL модели)
                if self.model_type == "gpt1":
                    max_history = 1000
                elif self.model_type == "gpt2":
                    max_history = 1500
                else:  # gpt2-xl
                    max_history = 2000
                    
                if len(self.conversation_history) > max_history:
                    self.conversation_history = self.conversation_history[-max_history:]
                
            except KeyboardInterrupt:
                print(f"\n\n👋 Чат с {self.model_type.upper()} прерван. До свидания!")
                break
            except Exception as e:
                print(f"\n❌ Ошибка: {e}")

def main():
    """Главная функция для выбора модели"""
    print("🚀 Добро пожаловать в GPT Chat!")
    print("\nВыберите модель:")
    print("1. GPT-1 (117M параметров, 2018)")
    print("2. GPT-2 Medium (355M параметров, 2019)")
    print("3. GPT-2 XL (1.5B параметров, 2019) - требует больше памяти")
    
    while True:
        choice = input("\nВаш выбор (1/2/3): ").strip()
        
        if choice == "1":
            model_type = "gpt1"
            break
        elif choice == "2":
            model_type = "gpt2"
            break
        elif choice == "3":
            model_type = "gpt2-xl"
            print("⚠️ Внимание: GPT-2 XL требует около 6GB памяти и может работать медленно без GPU.")
            confirm = input("Продолжить? (y/n): ").strip().lower()
            if confirm in ['y', 'yes', 'да', 'д']:
                break
            else:
                continue
        else:
            print("❌ Неверный выбор. Введите 1, 2 или 3.")
    
    # Создаем и запускаем чат
    chat = GPTChat(model_type)
    chat.run_chat()

if __name__ == "__main__":
    main()
