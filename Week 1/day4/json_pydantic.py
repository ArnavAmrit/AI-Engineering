import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set.")

client = Groq(api_key=my_api_key)

model="llama-3.3-70b-versatile"
role="user"

class Ticket(BaseModel):
    name:str
    email:str
    issue:str

schema = Ticket.model_json_schema()

response_format = {
    "type": "json_object"
}

system_prompt=f"""
Extract the personal detailsfrom the ticket strictly based on this schema and give me a json output.
{schema}
"""

message_system={
    "role": "system",
    "content": system_prompt
}
text = "Hello, My name is Arnav. I have an iphone which is not working at all. My address is Ghaziabad. My email is abc@gmail.com. My phone number is 1234567890. I have a problem with my iphone. Can you help me?"
prompt=f"""
This is a customer ticket. Please extract the personal information from the text:
{text}
"""
message={
    "role": role,
    "content": prompt
}

messages=[message_system, message]

response = client.chat.completions.create(model=model, messages=messages, response_format=response_format)
answer=response.choices[0].message.content
print(answer)


#now we will read it
import json
raw_json = answer
data_file = json.loads(raw_json)
ticket = Ticket(**data_file)
print(ticket.name)
print(ticket.email)
print(ticket.issue)