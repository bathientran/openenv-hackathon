"""Random agent baseline — establishes the floor for reward."""

import random
from recruitopenenv import RecruitopenenvEnv, RecruitopenenvAction

ACTIONS = [
    "send_text", "call_candidate",
    "ask_experience", "ask_home_time", "ask_pay", "ask_equipment",
    "ask_route", "ask_deal_breakers",
    "pitch_job", "match_to_job",
    "submit_application", "reject_candidate",
]

NUM_EPISODES = 100


def random_action():
    act = random.choice(ACTIONS)
    job_id = random.randint(0, 5) if act in ("match_to_job", "pitch_job") else -1
    return RecruitopenenvAction(action_type=act, job_id=job_id)


def run_baseline():
    rewards = []
    successes = 0
    dropouts = 0
    total_steps = 0

    with RecruitopenenvEnv(base_url="http://localhost:8000").sync() as env:
        for ep in range(NUM_EPISODES):
            result = env.reset()
            obs = result.observation
            ep_reward = 0.0
            steps = 0

            while not result.done and steps < 15:
                action = random_action()
                result = env.step(action)
                obs = result.observation
                ep_reward += result.reward
                steps += 1

            rewards.append(ep_reward)
            total_steps += steps

            if obs.stage == "submitted":
                successes += 1
            if obs.stage == "rejected" and "blocked" in obs.feedback.lower() or "stopped responding" in obs.feedback.lower():
                dropouts += 1

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
    print(f"Placement rate:     {successes}/{NUM_EPISODES} ({100*successes/NUM_EPISODES:.1f}%)")
    print(f"Trust dropout rate: {dropouts}/{NUM_EPISODES} ({100*dropouts/NUM_EPISODES:.1f}%)")
    print(f"Avg steps/episode:  {avg_steps:.1f}")
    print("======================================")


if __name__ == "__main__":
    run_baseline()
