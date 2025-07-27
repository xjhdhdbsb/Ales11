from transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
import torch
import warnings
warnings.filterwarnings("ignore")

class GPT1Chat:
    def __init__(self):
        print("🤖 Загружаю GPT-1...")
        self.tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")
        self.model = OpenAIGPTLMHeadModel.from_pretrained("openai-gpt")
        self.model.eval()
        
        # Настройки генерации
        self.max_length = 100
        self.temperature = 0.8
        self.top_k = 50
        
        print("✅ GPT-1 готов к общению!")
        print("💡 Команды:")
        print("   /settings - изменить настройки")
        print("   /clear - очистить историю")
        print("   /quit или /exit - выйти")
        print("=" * 50)
        
        # История разговора
        self.conversation_history = ""

    def generate_response(self, user_input):
        """Генерирует ответ на основе ввода пользователя"""
        try:
            # Формируем промпт с историей
            if self.conversation_history:
                prompt = f"{self.conversation_history}\nYou: {user_input}\nBot:"
            else:
                prompt = f"You: {user_input}\nBot:"
            
            # Токенизируем
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            
            # Ограничиваем длину контекста
            if inputs.size(1) > 500:  # Если контекст слишком длинный
                inputs = inputs[:, -400:]  # Берем последние 400 токенов
            
            # Генерируем
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.size(1) + 50,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=2
                )
            
            # Декодируем полный ответ
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Извлекаем только ответ бота
            bot_response = full_response[len(self.tokenizer.decode(inputs[0], skip_special_tokens=True)):].strip()
            
            # Убираем лишние части если есть
            if "\nYou:" in bot_response:
                bot_response = bot_response.split("\nYou:")[0].strip()
            if "\nBot:" in bot_response:
                bot_response = bot_response.split("\nBot:")[0].strip()
            
            return bot_response if bot_response else "I don't understand. Could you rephrase?"
            
        except Exception as e:
            return f"❌ Ошибка генерации: {str(e)}"

    def show_settings(self):
        """Показывает текущие настройки"""
        print(f"\n⚙️ Текущие настройки:")
        print(f"   Максимальная длина ответа: {self.max_length}")
        print(f"   Температура (креативность): {self.temperature}")
        print(f"   Top-k сэмплирование: {self.top_k}")
        print()

    def change_settings(self):
        """Позволяет изменить настройки"""
        self.show_settings()
        
        try:
            new_temp = input("Новая температура (0.1-2.0, текущая {:.1f}): ".format(self.temperature))
            if new_temp.strip():
                self.temperature = max(0.1, min(2.0, float(new_temp)))
            
            new_length = input(f"Новая максимальная длина (10-200, текущая {self.max_length}): ")
            if new_length.strip():
                self.max_length = max(10, min(200, int(new_length)))
            
            new_top_k = input(f"Новый top-k (1-100, текущий {self.top_k}): ")
            if new_top_k.strip():
                self.top_k = max(1, min(100, int(new_top_k)))
            
            print("✅ Настройки обновлены!")
            self.show_settings()
            
        except ValueError:
            print("❌ Неверный формат. Настройки не изменены.")

    def run_chat(self):
        """Основной цикл чата"""
        while True:
            try:
                # Получаем ввод пользователя
                user_input = input("\n👤 Вы: ").strip()
                
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
                
                elif user_input.lower() == '/help':
                    print("💡 Доступные команды:")
                    print("   /settings - изменить настройки генерации")
                    print("   /clear - очистить историю разговора")
                    print("   /help - показать эту справку")
                    print("   /quit или /exit - выйти из чата")
                    continue
                
                # Генерируем ответ
                print("🤖 GPT-1 думает...", end="", flush=True)
                response = self.generate_response(user_input)
                print(f"\r🤖 GPT-1: {response}")
                
                # Обновляем историю
                self.conversation_history += f"\nYou: {user_input}\nBot: {response}"
                
                # Ограничиваем историю
                if len(self.conversation_history) > 1000:
                    # Оставляем последние 800 символов
                    self.conversation_history = self.conversation_history[-800:]
                
            except KeyboardInterrupt:
                print("\n\n👋 Чат прерван пользователем. До свидания!")
                break
            except Exception as e:
                print(f"\n❌ Ошибка: {e}")
                print("Попробуйте еще раз или используйте /quit для выхода.")

if __name__ == "__main__":
    chat = GPT1Chat()
    chat.run_chat()
