import openai
import os

class TargetModel:
    def __init__(self, model_name="gpt-3.5-turbo", api_key=None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        # 多轮对话上下文历史
        self.conversation_history = []
        if self.api_key:
            openai.api_key = self.api_key

    def reset_conversation(self):
        """重置多轮对话会话"""
        self.conversation_history = []

    def get_response(self, prompt: str, system_prompt: str = None) -> str:
        """
        向目标模型发送请求，保持多轮上下文。
        """
        if not self.conversation_history and system_prompt:
            self.conversation_history.append({"role": "system", "content": system_prompt})
        
        self.conversation_history.append({"role": "user", "content": prompt})

        try:
            # 这是一个示例调用方式（使用OpenAI库的较老版本接口或者需要根据实际版本替换）
            # 新版OpenAI (1.0+) 使用 client = OpenAI() 
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model_name,
                messages=self.conversation_history,
                temperature=0.7
            )
            reply = response.choices[0].message.content
            self.conversation_history.append({"role": "assistant", "content": reply})
            return reply
        except Exception as e:
            print(f"Error calling model: {e}")
            return "Error"
