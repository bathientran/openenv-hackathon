"""Interactive CLI to play the recruiting environment manually."""

import json
import requests

BASE_URL = "http://localhost:8000"

SHORTCUTS = {
    "r": '{"tool":"crm","action":"read_candidate"}',
    "rr": '{"tool":"messaging","action":"read_reply"}',
    "w": '{"tool":"workflow","action":"wait"}',
    "ca": '{"tool":"approval","action":"check_approval"}',
    "hi": '{"tool":"crm","action":"update_stage","stage":"hired"}',
    "lost": '{"tool":"crm","action":"update_stage","stage":"lost"}',
}

TOPIC_SHORTCUTS = {
    "g": "greeting", "c": "call", "exp": "experience", "ht": "home_time",
    "pay": "pay", "eq": "equipment", "rt": "route", "db": "deal_breakers",
    "av": "availability", "vio": "violations", "med": "medical_card",
    "ref": "references", "pitch": "pitch", "offer": "offer",
    "np": "negotiate_pay", "nht": "negotiate_home_time",
    "sb": "signing_bonus", "ac": "address_concern",
}

def print_obs(obs, reward):
    print(f"\n{'='*60}")
    print(f"Driver: {obs['driver_name']}")
    if obs.get('crm_summary'):
        print(f"\nCRM:\n{obs['crm_summary']}")
    if obs.get('jobs_summary'):
        print(f"\nJobs:\n{obs['jobs_summary']}")
    if obs.get('discovered_info'):
        print(f"\nDiscovered:\n{obs['discovered_info']}")
    status = f"Stage: {obs['stage']}"
    if obs.get('pending_reply'):
        status += " | PENDING REPLY"
    print(f"\n{status}")
    print(f"Reward this step: {reward}")
    if obs.get('feedback'):
        try:
            fb = json.loads(obs['feedback'])
            print(f"Response: {json.dumps(fb, indent=2)}")
        except (json.JSONDecodeError, TypeError):
            print(f"Response: {obs['feedback']}")

def print_help():
    print("\nShortcuts:")
    print("  r     = read CRM")
    print("  rr    = read reply")
    print("  w     = wait")
    print("  ca    = check approval")
    print("  hi    = update stage to hired")
    print("  lost  = update stage to lost")
    print("\nSend message:  s <topic>     e.g. s g, s exp, s offer")
    print("  Topics: g=greeting c=call exp=experience ht=home_time pay eq=equipment")
    print("  rt=route db=deal_breakers av=availability vio=violations med=medical_card")
    print("  ref=references pitch offer np=negotiate_pay nht=negotiate_home_time")
    print("  sb=signing_bonus ac=address_concern")
    print("\nWith job_id:   s pitch 2     s offer 3")
    print("\nUpdate stage:  st <stage>    e.g. st contacted")
    print("Update field:  f <field> <value>  e.g. f cdl_class A")
    print("Add note:      n <text>      e.g. n Driver prefers OTR")
    print("Request approval: ra <job_id> e.g. ra 2")
    print("\nOr paste raw JSON: {\"tool\":\"crm\",\"action\":\"read_candidate\"}")
    print("  q = quit, h = help, reset = new episode")

def parse_input(user_input):
    user_input = user_input.strip()
    if not user_input:
        return None

    # Shortcuts
    if user_input in SHORTCUTS:
        return json.loads(SHORTCUTS[user_input])

    # Raw JSON
    if user_input.startswith("{"):
        return json.loads(user_input)

    parts = user_input.split(None, 2)
    cmd = parts[0]

    # Send message: s <topic> [job_id]
    if cmd == "s" and len(parts) >= 2:
        topic = TOPIC_SHORTCUTS.get(parts[1], parts[1])
        action = {"tool": "messaging", "action": "send_message", "topic": topic}
        if len(parts) >= 3:
            action["job_id"] = int(parts[2])
        return action

    # Update stage: st <stage>
    if cmd == "st" and len(parts) >= 2:
        return {"tool": "crm", "action": "update_stage", "stage": parts[1]}

    # Update field: f <field> <value>
    if cmd == "f" and len(parts) >= 3:
        return {"tool": "crm", "action": "update_field", "field": parts[1], "value": parts[2]}

    # Add note: n <text>
    if cmd == "n" and len(parts) >= 2:
        return {"tool": "crm", "action": "add_note", "value": " ".join(parts[1:])}

    # Request approval: ra <job_id>
    if cmd == "ra" and len(parts) >= 2:
        return {"tool": "approval", "action": "request_approval", "job_id": int(parts[1])}

    print(f"Unknown command: {user_input}. Type 'h' for help.")
    return None

def main():
    session = requests.Session()
    total_reward = 0.0

    print("\n🚛 DRIVER RECRUITING ENVIRONMENT — INTERACTIVE MODE")
    print_help()

    # Reset
    resp = session.post(f"{BASE_URL}/reset", json={})
    data = resp.json()
    obs = data["observation"]
    print_obs(obs, 0)

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if user_input == "q":
            break
        if user_input == "h":
            print_help()
            continue
        if user_input == "reset":
            resp = session.post(f"{BASE_URL}/reset", json={})
            data = resp.json()
            obs = data["observation"]
            total_reward = 0.0
            print_obs(obs, 0)
            continue

        action = parse_input(user_input)
        if action is None:
            continue

        print(f"→ {action['tool']}.{action['action']}"
              + (f"({action.get('topic', '')})" if action.get('topic') else "")
              + (f"[job={action['job_id']}]" if action.get('job_id', -1) >= 0 else "")
              + (f"({action.get('stage', '')})" if action.get('stage') else "")
              + (f"({action.get('field', '')}={action.get('value', '')})" if action.get('field') else ""))

        resp = session.post(f"{BASE_URL}/step", json=action)
        data = resp.json()
        obs = data["observation"]
        reward = data["reward"]
        done = data["done"]
        total_reward += reward

        print_obs(obs, reward)
        print(f"Total reward: {total_reward:.1f}")

        if done:
            print(f"\n{'='*60}")
            print(f"EPISODE OVER — Final stage: {obs['stage']} | Total reward: {total_reward:.1f}")
            print(f"{'='*60}")
            print("Type 'reset' for a new episode or 'q' to quit.")


if __name__ == "__main__":
    main()
