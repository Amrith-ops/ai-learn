from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client()  # reads GEMINI_API_KEY automatically

reply = client.models.generate_content(
    model="gemini-3.8-flash",
    contents="A customer texts: 'my AC stopped working, can someone come today?' "
             "Reply as a friendly home-service receptionist in 2 sentences.",
)
print(reply.text)