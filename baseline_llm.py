"""LLM agent baseline — test how well a base model performs without RL training."""

import json
import requests
from recruitopenenv import RecruitopenenvEnv, RecruitopenenvAction

LLM_URL = "http://localhost:8033/v1/chat/completions"
MODEL = "qwen2.5-32b-instruct-q5_k_m-00001-of-00006.gguf"
NUM_EPISODES = 20

SYSTEM_PROMPT = """You are a truck driver recruiter using a CRM system. You only know the driver's name. You must discover their qualifications through conversation, record info in the CRM, get approval, and hire them.

You have 4 tools:

## crm
- read_candidate: Read the current CRM record
- update_stage: Advance pipeline (contacted → interested → approval_pending → offer_sent → hired)
- update_field: Record info (field + value)
- add_note: Add a free-text note

## messaging
- send_message: Send a message (topic: greeting, call, experience, home_time, pay, equipment, route, deal_breakers, availability, violations, medical_card, references, pitch, offer, negotiate_pay, negotiate_home_time, signing_bonus, address_concern)
- read_reply: Read the driver's response

## approval
- request_approval: Request approval for a job (needs job_id)
- check_approval: Check approval status

## workflow
- wait: Advance time (needed for approval processing)

## Rules
- Must read CRM before messaging
- Must read_reply before sending another message
- Must request_approval and wait before sending offer
- Must follow stage order: lead → contacted → interested → approval_pending → offer_sent → hired
- Record important info in CRM with update_field
- Too many messages hurt trust

## Strategy
1. crm.read_candidate → see the lead
2. messaging.send_message(greeting or call) → messaging.read_reply → crm.update_stage(contacted)
3. Screen: send_message(experience) → read_reply → update_field(cdl_class, value) ... repeat for key questions
4. crm.update_stage(interested)
5. approval.request_approval(job_id) → workflow.wait → approval.check_approval
6. crm.update_stage(approval_pending)
7. messaging.send_message(offer) → messaging.read_reply
8. crm.update_stage(offer_sent) → crm.update_stage(hired)

Tips:
- ask_experience is critical (CDL class filters jobs)
- ask_deal_breakers helps avoid trap jobs
- ask_violations and ask_medical_card reveal fatal blockers
- If driver has concerns about offer, use negotiate_pay/negotiate_home_time/address_concern
- If no good match exists, update_stage to lost

Respond with ONLY JSON:
{"tool": "crm", "action": "read_candidate"}
{"tool": "messaging", "action": "send_message", "topic": "experience"}
{"tool": "messaging", "action": "read_reply"}
{"tool": "crm", "action": "update_field", "field": "cdl_class", "value": "A"}
{"tool": "approval", "action": "request_approval", "job_id": 2}
{"tool": "crm", "action": "update_stage", "stage": "hired"}"""


def format_observation(obs):
    parts = [f"Driver: {obs.driver_name}"]
    if obs.crm_summary:
        parts.append(f"CRM:\n{obs.crm_summary}")
    if obs.jobs_summary:
        parts.append(f"Jobs:\n{obs.jobs_summary}")
    if obs.discovered_info:
        parts.append(f"Discovered:\n{obs.discovered_info}")
    status = f"Stage: {obs.stage} | Trust: {obs.trust_level} | Step: {obs.steps_taken}"
    if obs.pending_reply:
        status += " | PENDING REPLY"
    if obs.negotiation_round > 0:
        status += f" | Negotiation round: {obs.negotiation_round}"
    parts.append(status)
    if obs.feedback:
        parts.append(f"Result: {obs.feedback}")
    return "\n".join(parts)


def ask_llm(messages):
    resp = requests.post(LLM_URL, json={
        "model": MODEL,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 150,
    })
    content = resp.json()["choices"][0]["message"]["content"]
    return content


def parse_action(text):
    """Try to extract action from LLM response."""
    text = text.strip()

    # Remove markdown code fences
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                text = part
                break

    # Try JSON parse
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "tool" in data and "action" in data:
            return RecruitopenenvAction(
                tool=data["tool"],
                action=data["action"],
                topic=data.get("topic", ""),
                job_id=data.get("job_id", -1),
                stage=data.get("stage", ""),
                field=data.get("field", ""),
                value=data.get("value", ""),
            )
    except (json.JSONDecodeError, KeyError):
        pass

    # Fallback
    text_lower = text.lower()
    if "read_candidate" in text_lower:
        return RecruitopenenvAction(tool="crm", action="read_candidate")
    if "read_reply" in text_lower:
        return RecruitopenenvAction(tool="messaging", action="read_reply")
    if "check_approval" in text_lower:
        return RecruitopenenvAction(tool="approval", action="check_approval")
    if "wait" in text_lower:
        return RecruitopenenvAction(tool="workflow", action="wait")

    return RecruitopenenvAction(tool="crm", action="read_candidate")


def run_baseline():
    rewards = []
    successes = 0
    total_steps = 0

    with RecruitopenenvEnv(base_url="http://localhost:8000").sync() as env:
        for ep in range(NUM_EPISODES):
            result = env.reset()
            obs = result.observation
            ep_reward = 0.0
            steps = 0

            messages = [{"role": "system", "content": SYSTEM_PROMPT}]

            while not result.done and steps < 100:
                obs_text = format_observation(obs)
                messages.append({"role": "user", "content": obs_text})

                llm_response = ask_llm(messages)
                messages.append({"role": "assistant", "content": llm_response})

                action = parse_action(llm_response)
                result = env.step(action)
                obs = result.observation
                ep_reward += result.reward
                steps += 1

                print(f"  Step {steps}: {action.tool}.{action.action}"
                      f"{'(' + action.topic + ')' if action.topic else ''}"
                      f"{'[job=' + str(action.job_id) + ']' if action.job_id >= 0 else ''}"
                      f" → reward={result.reward:.1f}, feedback: {obs.feedback[:80]}")

            rewards.append(ep_reward)
            total_steps += steps
            if obs.stage == "hired":
                successes += 1

            print(f"Episode {ep+1}: total_reward={ep_reward:.1f}, steps={steps}, "
                  f"{'HIRED' if obs.stage == 'hired' else 'FAIL (' + obs.stage + ')'}")
            print()

    avg_reward = sum(rewards) / len(rewards)
    avg_steps = total_steps / NUM_EPISODES

    print("\n========== LLM BASELINE (no RL) ==========")
    print(f"Model:              {MODEL}")
    print(f"Episodes:           {NUM_EPISODES}")
    print(f"Avg reward:         {avg_reward:.2f}")
    print(f"Min reward:         {min(rewards):.2f}")
    print(f"Max reward:         {max(rewards):.2f}")
    print(f"Hire rate:          {successes}/{NUM_EPISODES} ({100*successes/NUM_EPISODES:.1f}%)")
    print(f"Avg steps/episode:  {avg_steps:.1f}")
    print("==========================================")


if __name__ == "__main__":
    run_baseline()
