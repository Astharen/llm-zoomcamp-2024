import os
import openai
from openai import OpenAI
from tools.llms.openai.base import BaseOpenAIClient


class OpenAIClient(BaseOpenAIClient):

    def __init__(self, model_name=None) -> None:
        super().__init__()
        self.client = OpenAI(
                api_key=os.getenv("OPENAI_TOKEN"),
            )
            
        self.model_name = model_name
