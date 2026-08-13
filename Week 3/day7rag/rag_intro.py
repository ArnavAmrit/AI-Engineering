import os
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set.")

client = Groq(api_key=my_api_key)
model = "llama-3.3-70b-versatile"


#step 1: knowledge base
knowledge_base = {
    "age" : "The age of Arnav is 20 years",
    "net worth" : "The net worth of Arnav is 2000",
}

#step 2: retreival

def retrieve_info(question):
    question = question.lower()
    if("age" in question):
        return knowledge_base["age"]
    elif("net worth" in question):
        return knowledge_base["net worth"]
    else:
        return None

def ask_llm(question):
    context = retrieve_info(question)
    system_prompt = f"""Answer in one line only. answer only based on this context. Do not hallucinate. Context: {context}"""
    system_message = {
        "role" : "system",
        "content" : system_prompt
    }
    message = {
        "role" : "user",
        "content" : question
    }
    messages = [system_message, message]
    response = client.chat.completions.create(model=model, messages=messages)
    answer = response.choices[0].message.content
    return answer

question = "How old is Arnav?"
print(ask_llm(question))