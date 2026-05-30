import requests
import json

class LocalLLMSentiment:
    def __init__(self, model_name="llama3", endpoint="http://localhost:11434/api/generate"):
        self.model_name = model_name
        self.endpoint = endpoint

    def analyze_sentiment(self, text_context: str) -> float:
        """
        Sends the text to Ollama and expects a sentiment score from -1.0 to 1.0.
        """
        prompt = f"""
        You are a financial analyst. Read the following Forex market news/context and output ONLY a single float number between -1.0 (strongly bearish) and 1.0 (strongly bullish). 
        Do not add any text, markdown, or explanations. Only the number.

        Context: {text_context}
        """

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }

        try:
            response = requests.post(self.endpoint, json=payload, timeout=10)
            response.raise_for_status()
            result_text = response.json().get("response", "").strip()
            # Attempt to cast the clean output to float
            score = float(result_text)
            # Clamp between -1.0 and 1.0
            return max(-1.0, min(1.0, score))
        except Exception as e:
            print(f"LLM Sentiment Error: {e}")
            return 0.0 # Neutral if failed

# Stub for the RL Agent
class RLAgentInterface:
    def __init__(self, agent_id, db_session, engine):
        self.agent_id = agent_id
        self.session = db_session
        self.engine = engine
        
    def decide_action(self, chart_data, sentiment_score):
        # Placeholder for actual Stable Baselines3 model logic
        # 0 = Hold, 1 = Buy, 2 = Sell
        return 0
