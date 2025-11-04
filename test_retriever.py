from ollama import chat
from ollama import ChatResponse
import os
from query import *
import json


with open('test_examples.json', 'r') as fp:
    test_examples = json.load(fp)

results_to_check = get_results('type of catalysts', retriever)

req = requests.get("http://localhost:11434/api/tags").json()
model_is_gemma = [True for model in req['models'] if "gemma3" in model['name']]
if not any(model_is_gemma):
    os.system("ollama pull gemma3:4b")

outputs = []

for (i, item1), (j, item2) in zip(test_examples.items(), results_to_check.items()):
  response = chat(
    model='gemma3:4b', 
    messages=[
      {
        'role': 'user',
        'content': f'Please compare {item1} and {item2}. If they are similar, output "OK". If they are not, output "Not OK".'
      },
    ]
  )
  outputs.append(response['message']['content'])

outputs = [True if output == "OK" else False for output in outputs]
if outputs.count(True) >= 8: 
  print("The retriever is OK !")