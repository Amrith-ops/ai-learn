import time
from dotenv import load_dotenv
from google import genai
from google.genai import types, errors

import re   # add near the top of agent.py with the other imports

_last_call = 0.0
MIN_GAP = 4.5   # 15 requests/min means at most one call per 4 seconds


load_dotenv()
client = genai.Client(http_options=types.HttpOptions(timeout=30000))
MODEL = "gemini-3.5-flash-lite"   # use whichever model worked for you

BOOKINGS = []

# ---------- tools ----------
def check_availability(date: str) -> dict:
    """Check open appointment slots for a date in YYYY-MM-DD format."""
    calendar = {"2026-11-08": ["10:00", "14:00"], "2026-11-09": ["09:00"]}
    taken = {(b["date"], b["time_slot"]) for b in BOOKINGS}
    return {"date": date,
            "open_slots": [t for t in calendar.get(date, []) if (date, t) not in taken]}

def book_appointment(date: str, time_slot: str, customer_name: str,
                     address: str, issue: str) -> dict:
    """Book an appointment. Only call after the customer has chosen a slot
    and given their name and address."""
    if time_slot not in check_availability(date)["open_slots"]:
        return {"success": False, "error": "That slot is not available."}
    BOOKINGS.append({"date": date, "time_slot": time_slot,
                     "customer_name": customer_name, "address": address, "issue": issue})
    return {"success": True, "confirmation": f"HV-{len(BOOKINGS):03d}"}

TOOLS = {f.__name__: f for f in [check_availability, book_appointment]}

config = types.GenerateContentConfig(
    system_instruction=(
        "You are a friendly receptionist for an HVAC company. Today is 2026-11-07. "
        "Check availability with the tool before offering times. "
        "Never book without the customer's chosen slot, name, and address; ask for what's missing."
        
    ),
    tools=list(TOOLS.values()),
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)

# ---------- harness: retries ----------
def call_model(**kwargs):
    global _last_call
    for attempt in range(6):
        gap = time.time() - _last_call
        if gap < MIN_GAP:
            time.sleep(MIN_GAP - gap)          # throttle before calling
        _last_call = time.time()
        try:
            return client.models.generate_content(**kwargs)
        except (errors.ServerError, errors.ClientError) as e:
            if isinstance(e, errors.ClientError) and e.code != 429:
                raise                          # bad key or bad request: fail loudly
            m = re.search(r"retry in ([\d.]+)s", str(e))
            wait = float(m.group(1)) + 2 if m else min(5 * 2 ** attempt, 60)
            print(f"  (error {e.code}, retrying in {wait:.0f}s)")
            time.sleep(wait)
    raise RuntimeError("Still rate-limited or unavailable after 6 tries")

# ---------- the agent loop ----------
def run_agent(contents, max_steps=2):
    for step in range(max_steps):
        print(f"[step {step+1}] model sees")
        response = call_model(model=MODEL, contents=contents, config=config)
        msg = response.candidates[0].content
        contents.append(msg)

        calls = [p.function_call for p in (msg.parts or []) if p.function_call]
        if not calls:                      # no tool request = the agent is done
            return response.text

        results = []
        for call in calls:
            print(f"  [step {step+1}] tool: {call.name}({dict(call.args)})")
            try:
                result = TOOLS[call.name](**call.args)
            except Exception as e:         # errors go back to the model, not a crash
                result = {"error": str(e)}
            results.append(types.Part.from_function_response(
                name=call.name, response=result))
        contents.append(types.Content(role="user", parts=results))

    return "Sorry, I couldn't finish that. A team member will follow up."

# ---------- chat ----------
contents = []
print("Type 'quit' to exit.")
while True:
    user = input("\nCustomer: ")
    if user.lower() == "quit":
        break
    contents.append(types.Content(role="user", parts=[types.Part(text=user)]))
    print("Agent:", run_agent(contents))
print("\nBookings:", BOOKINGS)