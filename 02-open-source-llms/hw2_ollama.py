from openai import OpenAI


prompt = "What's the formula for energy"


client = OpenAI(
    base_url='http://localhost:11434/v1/',
    api_key='ollama'
)

response = client.chat.completions.create(
    model="gemma2b",
    messages=[
    {"role": "user", "content": prompt}
  ],
  temperature=0.0)