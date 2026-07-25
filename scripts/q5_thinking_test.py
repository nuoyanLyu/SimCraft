"""Q5: compare Simulator (AgentWorld-35B) next-state prediction quality with
thinking OFF (our current forced setting) vs ON (the paper/official eval
setting). Runs against the live server on :8800."""
import json, re, time
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8800/v1", api_key="EMPTY")
MODEL = "Qwen-AgentWorld-35B-A3B"

SIM_SYS = (
    "You simulate the effect of one tool call on an environment's canonical state. "
    "Given the current state and a tool call (name + arguments), predict the resulting state. "
    'Reply with a single JSON object: {"next_state": {...}}. Do not explain your reasoning.'
)

SCENARIOS = [
    {
        "name": "write_note (create)",
        "state": {"notes": [{"id": "n1", "title": "Grocery List", "body": "flour, eggs, butter, yeast, salt"}]},
        "call": ("write_note", {"title": "Baking Plan", "body": "Preheat oven to 220C"}),
    },
    {
        "name": "delete_note (by id)",
        "state": {"notes": [{"id": "n1", "title": "Grocery List", "body": "milk"},
                             {"id": "n2", "title": "Temp", "body": "scratch"}]},
        "call": ("delete_note", {"id": "n2"}),
    },
    {
        "name": "add_tag (touch one field)",
        "state": {"notes": [{"id": "n1", "title": "Trip", "body": "Paris", "tags": []}], "cwd": "/home/u"},
        "call": ("add_tag", {"id": "n1", "tag": "travel"}),
    },
]

def user_prompt(state, call):
    name, args = call
    return (f"Current state:\n{json.dumps(state, indent=2)}\n\n"
            f"Tool call: {name}({json.dumps(args)})\n\n"
            "Produce the JSON object described in the system prompt.")

def extract_json(text):
    # strip <think>...</think> if present
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return m.group(0) if m else text

def call(state, tc, thinking, max_tokens):
    t0 = time.time()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": SIM_SYS},
                  {"role": "user", "content": user_prompt(state, tc)}],
        max_tokens=max_tokens, temperature=0.6,
        extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
    )
    dt = time.time() - t0
    msg = resp.choices[0].message
    content = msg.content or ""
    reasoning = getattr(msg, "reasoning_content", None) or ""
    return content, reasoning, dt

for sc in SCENARIOS:
    print("=" * 78)
    print("SCENARIO:", sc["name"], "| tool:", sc["call"][0])
    for thinking, mt in [(False, 800), (True, 7000)]:
        label = "THINK-ON " if thinking else "THINK-OFF"
        try:
            content, reasoning, dt = call(sc["state"], sc["call"], thinking, mt)
            parsed_ok = False
            try:
                obj = json.loads(extract_json(content))
                parsed_ok = "next_state" in obj
            except Exception:
                obj = None
            print(f"\n[{label}] {dt:.1f}s  content_len={len(content)}  reasoning_len={len(reasoning)}  parsed_ok={parsed_ok}")
            if parsed_ok:
                print("  next_state:", json.dumps(obj["next_state"], ensure_ascii=False)[:400])
            else:
                print("  RAW content[:400]:", content[:400].replace("\n", " "))
        except Exception as e:
            print(f"\n[{label}] ERROR: {e}")
    print()
