from openai import OpenAI
from src.pinecone_db import vectorDB
from dotenv import load_dotenv
from dataclasses import dataclass

from src import pinecone_db

load_dotenv()

client = OpenAI()

class AIResponse:
    def __init__(self, userResponse, source_knowledge):
        self.userResponse = userResponse
        self.source_knowledge = source_knowledge

    def generatePrompt(self):
        try:
            aug_prompt = f"""
            You are a chatbot that is trained to answer for user response.
            You are given the following context:
            {self.source_knowledge}
            You are asked to 
              short and accurate answer for the following question using above context.
            question:
            {self.userResponse}
            strictly do not hallucinate. only use above context to generate answer.
            """
            return aug_prompt

        except Exception as e:
            raise e
    
    def generateResponse(self):
        try:
            prompt = self.generatePrompt()
            response = client.responses.create(
                model="gpt-6-astra",
                input=prompt
            )
            model_response = response.output_text
            return model_response
        
        except Exception as e:
            raise e
        
