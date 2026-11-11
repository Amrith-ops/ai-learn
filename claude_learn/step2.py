from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(http_options=types.HttpOptions(timeout=30000))
MODEL = "gemini-3.5-flash-lite"
# "gemini-3.7-flash", 
def check_availability(date: str) -> dict:
    """Check open appointment slots for a date in YYYY-MM-DD format."""
    fake_calendar = {"2026-11-08": ["10:00", "14:00"], "2026-11-09": ["09:00"]}
    return {"date": date, "open_slots": fake_calendar.get(date, [])}

config = types.GenerateContentConfig(
    system_instruction=(
        "You are a friendly receptionist for an HVAC company. Today is 2026-11-07. "
        "Always use tools to check availability before offering times."
    ),
    tools=[check_availability],
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)

contents = [types.Content(role="user", parts=[
    types.Part(text="My AC stopped working, can someone come tomorrow?")])]

# Turn 1: model decides whether it needs a tool
response = client.models.generate_content(model=MODEL, contents=contents, config=config)
model_msg = response.candidates[0].content
call = next((p.function_call for p in model_msg.parts if p.function_call), None)
print("Model wants to call:", call)

# Turn 2: we run the tool and hand the result back
if call:
    result = check_availability(**call.args)
    print("Tool returned:", result)
    contents.append(model_msg)
    contents.append(types.Content(role="user", parts=[
        types.Part.from_function_response(name=call.name, response=result)]))
    final = client.models.generate_content(model=MODEL, contents=contents, config=config)
    print("\nFinal reply:", final.text)