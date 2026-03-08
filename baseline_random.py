"""Random agent baseline — establishes the floor for reward."""

import random
from recruitopenenv import RecruitopenenvEnv, RecruitopenenvAction

TOOLS_ACTIONS = {
    "crm": ["read_candidate", "update_stage", "update_field", "add_note"],
    "messaging": ["send_message", "read_reply"],
    "approval": ["request_approval", "check_approval"],
    "workflow": ["wait"],
}

TOPICS = [
    "greeting", "call", "experience", "home_time", "pay", "equipment",
    "route", "deal_breakers", "availability", "violations", "medical_card",
    "references", "pitch", "offer", "negotiate_pay", "negotiate_home_time",
    "signing_bonus", "address_concern",
]

STAGES = ["contacted", "interested", "approval_pending", "offer_sent", "hired", "lost"]

NUM_EPISODES = 100


def random_action():
    tool = random.choice(list(TOOLS_ACTIONS.keys()))
    action = random.choice(TOOLS_ACTIONS[tool])

    topic = ""
    job_id = -1
    stage = ""
    field = ""
    value = ""

    if tool == "messaging" and action == "send_message":
        topic = random.choice(TOPICS)
        if topic in ("pitch", "offer"):
            job_id = random.randint(0, 5)
    elif tool == "crm" and action == "update_stage":
        stage = random.choice(STAGES)
    elif tool == "crm" and action == "update_field":
        field = random.choice(["cdl_class", "years_exp", "home_time_pref"])
        value = "A"
    elif tool == "approval" and action == "request_approval":
        job_id = random.randint(0, 5)

    return RecruitopenenvAction(
        tool=tool, action=action, topic=topic,
        job_id=job_id, stage=stage, field=field, value=value,
    )


def run_baseline():
    rewards = []
    successes = 0
    total_steps = 0

    with RecruitopenenvEnv(base_url="http://localhost:8000").sync() as env:
        for ep in range(NUM_EPISODES):
            result = env.reset()
            ep_reward = 0.0
            steps = 0

            while not result.done and steps < 100:
                action = random_action()
                result = env.step(action)
                ep_reward += result.reward
                steps += 1

            rewards.append(ep_reward)
            total_steps += steps

            if result.observation.stage == "hired":
                successes += 1

            if (ep + 1) % 10 == 0:
                avg_so_far = sum(rewards) / len(rewards)
                print(f"  Episode {ep+1}: reward={ep_reward:.1f}, running avg={avg_so_far:.2f}")

    avg_reward = sum(rewards) / len(rewards)
    avg_steps = total_steps / NUM_EPISODES

    print("\n========== RANDOM BASELINE ==========")
    print(f"Episodes:           {NUM_EPISODES}")
    print(f"Avg reward:         {avg_reward:.2f}")
    print(f"Min reward:         {min(rewards):.2f}")
    print(f"Max reward:         {max(rewards):.2f}")
    print(f"Hire rate:          {successes}/{NUM_EPISODES} ({100*successes/NUM_EPISODES:.1f}%)")
    print(f"Avg steps/episode:  {avg_steps:.1f}")
    print("======================================")


if __name__ == "__main__":
    run_baseline()
