import inspect
from google import genai

try:
    print("aio:", dir(genai.Client().aio.models))
except Exception as e:
    print(e)
