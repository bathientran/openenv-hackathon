"""LLM agent baseline — test how well a base model performs without RL training."""

import json
import requests
from recruitopenenv import RecruitopenenvEnv, RecruitopenenvAction

LLM_URL = "http://localhost:8033/v1/chat/completions"
MODEL = "qwen2.5-32b-instruct-q5_k_m-00001-of-00006.gguf"
NUM_EPISODES = 20

SYSTEM_PROMPT = """You are a truck driver recruiter. You only know the driver's name. You must discover their qualifications and preferences through screening questions, then match them to the best job.

Valid actions:
- send_text: Send a text message to the candidate
- call_candidate: Call the candidate on the phone
- ask_experience: Ask about CDL class, years of experience, endorsements, location
- ask_home_time: Ask about home time preference (daily, weekends, weekly, biweekly)
- ask_pay: Ask about pay expectations (cents per mile)
- ask_equipment: Ask about equipment preference (dry van, flatbed, reefer, tanker)
- ask_route: Ask about route preference (OTR, regional, local, dedicated)
- ask_deal_breakers: Ask what they absolutely won't do (touch freight, forced dispatch, team driving, etc.)
- pitch_job <job_id>: Describe a specific job and get their reaction (0-5)
- match_to_job <job_id>: Formally match candidate to a job (0-5)
- submit_application: Submit the matched application
- reject_candidate: Reject — no good match exists

Strategy tips:
- You MUST contact first (send_text or call_candidate), then screen, then match
- Each question costs a step and slightly reduces trust — be strategic about which to ask
- ask_experience is critical — you need CDL class to filter jobs
- ask_deal_breakers helps avoid trap jobs that look good but have hidden issues
- Some drivers are impatient and only reveal partial info
- Not all drivers will answer honestly if trust is low
- You can pitch_job to test a driver's reaction before formally matching

Respond with ONLY the action in JSON format:
{"action_type": "send_text"}
{"action_type": "ask_experience"}
{"action_type": "pitch_job", "job_id": 2}
{"action_type": "match_to_job", "job_id": 0}"""


def format_observation(obs):
    parts = [f"Driver: {obs.driver_name}"]
    if obs.jobs_summary:
        parts.append(f"Jobs:\n{obs.jobs_summary}")
    if obs.discovered_info:
        parts.append(f"Discovered info:\n{obs.discovered_info}")
    parts.append(f"Stage: {obs.stage} | Trust: {obs.trust_level} | Step: {obs.steps_taken}/{obs.max_steps} | Matched: {obs.matched_job_id}")
    parts.append(f"Feedback: {obs.feedback}")
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
        return RecruitopenenvAction(
            action_type=data["action_type"],
            job_id=data.get("job_id", -1),
        )
    except (json.JSONDecodeError, KeyError):
        pass

    # Fallback: look for known action names
    text_lower = text.lower()
    all_actions = [
        "submit_application", "reject_candidate",
        "ask_experience", "ask_home_time", "ask_pay",
        "ask_equipment", "ask_route", "ask_deal_breakers",
        "pitch_job", "match_to_job",
        "send_text", "call_candidate",
    ]
    for act in all_actions:
        if act in text_lower:
            job_id = -1
            if act in ("match_to_job", "pitch_job"):
                for c in text:
                    if c in "012345":
                        job_id = int(c)
                        break
            return RecruitopenenvAction(action_type=act, job_id=job_id)

    return RecruitopenenvAction(action_type="ask_experience")


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

            while not result.done and steps < 15:
                obs_text = format_observation(obs)
                messages.append({"role": "user", "content": obs_text})

                llm_response = ask_llm(messages)
                messages.append({"role": "assistant", "content": llm_response})

                action = parse_action(llm_response)
                result = env.step(action)
                obs = result.observation
                ep_reward += result.reward
                steps += 1

                print(f"  Step {steps}: {action.action_type}"
                      f"{'(' + str(action.job_id) + ')' if action.job_id >= 0 else ''}"
                      f" → reward={result.reward}, feedback: {obs.feedback}")

            rewards.append(ep_reward)
            total_steps += steps
            if obs.stage == "submitted":
                successes += 1

            print(f"Episode {ep+1}: total_reward={ep_reward:.1f}, steps={steps}, "
                  f"{'SUCCESS' if obs.stage == 'submitted' else 'FAIL'}")
            print()

    avg_reward = sum(rewards) / len(rewards)
    avg_steps = total_steps / NUM_EPISODES

    print("\n========== LLM BASELINE (no RL) ==========")
    print(f"Model:              {MODEL}")
    print(f"Episodes:           {NUM_EPISODES}")
    print(f"Avg reward:         {avg_reward:.2f}")
    print(f"Min reward:         {min(rewards):.2f}")
    print(f"Max reward:         {max(rewards):.2f}")
    print(f"Placement rate:     {successes}/{NUM_EPISODES} ({100*successes/NUM_EPISODES:.1f}%)")
    print(f"Avg steps/episode:  {avg_steps:.1f}")
    print("==========================================")


if __name__ == "__main__":
    run_baseline()
