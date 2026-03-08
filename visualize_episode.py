"""Visualize episodes step-by-step with detailed state after each turn.

Modes:
  interactive  — you type actions manually
  llm          — watch an LLM agent play (uses local vLLM/llama.cpp endpoint)
  random       — watch random actions

Usage:
    python visualize_episode.py                          # interactive
    python visualize_episode.py --mode llm               # watch LLM play
    python visualize_episode.py --mode random             # random agent
    python visualize_episode.py --episodes 5 --mode llm   # run 5 LLM episodes
"""

import argparse
import json
import random
import requests

from recruitopenenv import RecruitopenenvEnv, RecruitopenenvAction

# ── Colors ──────────────────────────────────────────────────────────

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"
BG_RED = "\033[41m"
BG_GREEN = "\033[42m"
BG_YELLOW = "\033[43m"
BG_BLUE = "\033[44m"

# ── Display helpers ─────────────────────────────────────────────────

def hr(char="─", width=70):
    print(f"{DIM}{char * width}{RESET}")

def section(title):
    print(f"\n{BOLD}{CYAN}┌{'─' * (len(title) + 2)}┐{RESET}")
    print(f"{BOLD}{CYAN}│ {title} │{RESET}")
    print(f"{BOLD}{CYAN}└{'─' * (len(title) + 2)}┘{RESET}")

def trust_bar(level_str):
    """Render a visual trust bar from the trust level string."""
    colors = {"high": GREEN, "medium": YELLOW, "low": RED}
    icons = {"high": "████████", "medium": "█████░░░", "low": "██░░░░░░"}
    color = colors.get(level_str, WHITE)
    icon = icons.get(level_str, "????")
    return f"{color}{icon} {level_str}{RESET}"

def reward_color(r):
    if r > 0:
        return f"{GREEN}+{r:.1f}{RESET}"
    elif r < 0:
        return f"{RED}{r:.1f}{RESET}"
    return f"{DIM}0.0{RESET}"

def stage_color(stage):
    colors = {
        "outreach": BLUE,
        "screening": YELLOW,
        "matching": MAGENTA,
        "matched": CYAN,
        "submitted": GREEN,
        "rejected": RED,
    }
    color = colors.get(stage, WHITE)
    return f"{color}{BOLD}{stage}{RESET}"

def trust_gauge(trust_val):
    """Render a proportional trust gauge from raw 0.0-1.0 value."""
    bar_len = 20
    filled = int(trust_val * bar_len)
    empty = bar_len - filled
    if trust_val >= 0.5:
        color = GREEN
    elif trust_val >= 0.2:
        color = YELLOW
    else:
        color = RED
    return f"{color}{'█' * filled}{DIM}{'░' * empty}{RESET} {color}{trust_val:.3f}{RESET}"


def print_observation(obs, step_num, reward, action_taken=None, prev_trust=None):
    """Print a full state snapshot after a turn."""
    hr("═")

    # Step header
    if action_taken:
        act_str = action_taken.action_type
        if action_taken.job_id >= 0:
            act_str += f" (job {action_taken.job_id})"
        print(f"{BOLD}Step {step_num}{RESET}  ▸ Action: {BOLD}{YELLOW}{act_str}{RESET}  ▸ Reward: {reward_color(reward)}")
    else:
        print(f"{BOLD}Step {step_num}{RESET}  ▸ {DIM}Episode start{RESET}")

    hr()

    # State bar
    steps_left = obs.max_steps - obs.steps_taken
    steps_color = GREEN if steps_left > 5 else (YELLOW if steps_left > 2 else RED)
    personality_colors = {
        "chatty": GREEN, "professional": CYAN,
        "impatient": RED, "suspicious": MAGENTA,
    }
    p_color = personality_colors.get(obs.personality, WHITE)
    print(
        f"  Driver: {BOLD}{obs.driver_name}{RESET}  │  "
        f"Personality: {p_color}{obs.personality}{RESET}  │  "
        f"Stage: {stage_color(obs.stage)}  │  "
        f"Steps: {steps_color}{obs.steps_taken}/{obs.max_steps}{RESET}"
    )

    # Trust with delta
    trust_line = f"  Trust:  {trust_gauge(obs.trust)}"
    if prev_trust is not None:
        delta = obs.trust - prev_trust
        if delta > 0:
            trust_line += f"  {GREEN}▲ +{delta:.3f}{RESET}"
        elif delta < 0:
            trust_line += f"  {RED}▼ {delta:.3f}{RESET}"
        else:
            trust_line += f"  {DIM}= no change{RESET}"
    print(trust_line)

    # Placeability
    if obs.was_placeable:
        print(f"  Placeable: {GREEN}YES{RESET} (best fit score: {obs.best_possible_score})")
    else:
        print(f"  Placeable: {RED}NO{RESET} (best fit score: {obs.best_possible_score}) — correct action is reject_candidate")

    # Questions asked
    if obs.questions_asked:
        asked = ", ".join(q.replace("ask_", "") for q in obs.questions_asked)
        all_q = ["experience", "home_time", "pay", "equipment", "route", "deal_breakers"]
        remaining = [q for q in all_q if f"ask_{q}" not in obs.questions_asked]
        remaining_str = f"  {DIM}remaining: {', '.join(remaining)}{RESET}" if remaining else ""
        print(f"  Asked:  {CYAN}{asked}{RESET}{remaining_str}")

    # Matched job
    if obs.matched_job_id >= 0:
        print(f"  Matched job: {BOLD}{obs.matched_job_id}{RESET}")

    # Feedback
    if obs.feedback:
        print(f"\n  {BOLD}Feedback:{RESET}")
        for line in obs.feedback.split("\n"):
            print(f"    {line}")

    # Discovered info
    if obs.discovered_info:
        print(f"\n  {BOLD}Discovered info:{RESET}")
        for line in obs.discovered_info.split("\n"):
            if line.strip():
                print(f"    {DIM}•{RESET} {line.strip()}")

    # Jobs (show on first step or if requested)
    if step_num == 0 and obs.jobs_summary:
        print(f"\n  {BOLD}Available jobs:{RESET}")
        for line in obs.jobs_summary.split("\n"):
            print(f"    {line}")

    print()


def print_episode_summary(episode_num, total_reward, steps, final_obs, was_placeable):
    """Print end-of-episode summary."""
    section(f"Episode {episode_num} Summary")

    outcome = final_obs.stage
    if outcome == "submitted":
        print(f"  Outcome:  {BG_GREEN}{BOLD} PLACED {RESET}")
    elif outcome == "rejected":
        print(f"  Outcome:  {BG_RED}{BOLD} REJECTED / FAILED {RESET}")
    else:
        print(f"  Outcome:  {BG_YELLOW}{BOLD} {outcome.upper()} {RESET}")

    print(f"  Reward:   {reward_color(total_reward)}")
    print(f"  Steps:    {steps}")

    if was_placeable and outcome != "submitted":
        print(f"  Verdict:  {RED}{BOLD}REGRETTABLE FAILURE{RESET} — a good match existed")
    elif not was_placeable and outcome != "submitted":
        print(f"  Verdict:  {YELLOW}INTRINSIC FAILURE{RESET} — no good match was available")
        if final_obs.stage == "rejected" and total_reward > 0:
            print(f"            {GREEN}(correctly rejected!){RESET}")
    elif was_placeable and outcome == "submitted":
        print(f"  Verdict:  {GREEN}{BOLD}SUCCESS{RESET}")

    hr("═")
    print()


# ── Action input ────────────────────────────────────────────────────

ALL_ACTIONS = [
    "send_text", "call_candidate",
    "ask_experience", "ask_home_time", "ask_pay",
    "ask_equipment", "ask_route", "ask_deal_breakers",
    "pitch_job", "match_to_job",
    "submit_application", "reject_candidate",
]

def get_interactive_action():
    """Prompt user for an action."""
    print(f"  {BOLD}Actions:{RESET}")
    for i, act in enumerate(ALL_ACTIONS):
        print(f"    {DIM}{i:2d}{RESET}  {act}")
    print()
    while True:
        raw = input(f"  {CYAN}>{RESET} ").strip()
        if not raw:
            continue
        # By number
        try:
            idx = int(raw.split()[0])
            if 0 <= idx < len(ALL_ACTIONS):
                act = ALL_ACTIONS[idx]
                job_id = -1
                if act in ("pitch_job", "match_to_job"):
                    parts = raw.split()
                    if len(parts) > 1:
                        job_id = int(parts[1])
                    else:
                        job_id = int(input(f"    job_id (0-5): ").strip())
                return RecruitopenenvAction(action_type=act, job_id=job_id)
        except (ValueError, IndexError):
            pass
        # By name
        for act in ALL_ACTIONS:
            if act in raw.lower():
                job_id = -1
                if act in ("pitch_job", "match_to_job"):
                    for c in raw:
                        if c in "012345":
                            job_id = int(c)
                            break
                    if job_id < 0:
                        job_id = int(input(f"    job_id (0-5): ").strip())
                return RecruitopenenvAction(action_type=act, job_id=job_id)
        print(f"    {RED}Unknown action. Try a number (0-11) or action name.{RESET}")


def get_random_action(obs):
    """Pick a random plausible action based on current stage."""
    if obs.stage == "outreach":
        return RecruitopenenvAction(action_type=random.choice(["send_text", "call_candidate"]))
    elif obs.stage in ("screening", "matching"):
        choices = [
            "ask_experience", "ask_home_time", "ask_pay",
            "ask_equipment", "ask_route", "ask_deal_breakers",
            "pitch_job", "match_to_job", "submit_application", "reject_candidate",
        ]
        act = random.choice(choices)
        job_id = random.randint(0, 5) if act in ("pitch_job", "match_to_job") else -1
        return RecruitopenenvAction(action_type=act, job_id=job_id)
    else:
        return RecruitopenenvAction(action_type="submit_application")


LLM_SYSTEM_PROMPT = """You are a truck driver recruiter. You only know the driver's name. You must discover their qualifications and preferences through screening questions, then match them to the best job.

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
- Contact first (send_text or call_candidate), then screen, then match
- ask_experience is critical — you need CDL class to filter jobs
- ask_deal_breakers helps avoid trap jobs
- Not every driver can be placed — use reject_candidate if no good match exists

Respond with ONLY the action in JSON format:
{"action_type": "send_text"}
{"action_type": "pitch_job", "job_id": 2}"""


def get_llm_action(obs, messages, llm_url, model):
    """Ask an LLM for the next action."""
    obs_text = format_obs_for_llm(obs)
    messages.append({"role": "user", "content": obs_text})

    resp = requests.post(llm_url, json={
        "model": model,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 150,
    })
    content = resp.json()["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": content})

    # Show raw LLM output
    print(f"  {DIM}LLM → {content.strip()}{RESET}")

    return parse_llm_action(content)


def format_obs_for_llm(obs):
    parts = [f"Driver: {obs.driver_name}"]
    if obs.jobs_summary:
        parts.append(f"Jobs:\n{obs.jobs_summary}")
    if obs.discovered_info:
        parts.append(f"Discovered info:\n{obs.discovered_info}")
    parts.append(
        f"Stage: {obs.stage} | Trust: {obs.trust_level} | "
        f"Step: {obs.steps_taken}/{obs.max_steps} | Matched: {obs.matched_job_id}"
    )
    if obs.feedback:
        parts.append(f"Feedback: {obs.feedback}")
    return "\n".join(parts)


def parse_llm_action(text):
    text = text.strip()
    if "```" in text:
        for part in text.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                text = part
                break
    try:
        data = json.loads(text)
        if isinstance(data, list):
            data = data[0] if data else {}
        if isinstance(data, dict) and "action_type" in data:
            return RecruitopenenvAction(
                action_type=data["action_type"],
                job_id=data.get("job_id", -1),
            )
    except (json.JSONDecodeError, KeyError, IndexError):
        pass
    text_lower = text.lower()
    for act in ALL_ACTIONS:
        if act in text_lower:
            job_id = -1
            if act in ("match_to_job", "pitch_job"):
                for c in text:
                    if c in "012345":
                        job_id = int(c)
                        break
            return RecruitopenenvAction(action_type=act, job_id=job_id)
    return RecruitopenenvAction(action_type="ask_experience")


# ── Main loop ───────────────────────────────────────────────────────

def run_episode(env, episode_num, mode, llm_url=None, model=None):
    """Run and visualize a single episode."""
    result = env.reset()
    obs = result.observation

    section(f"Episode {episode_num}")
    print_observation(obs, 0, 0.0)

    total_reward = 0.0
    steps = 0
    prev_trust = obs.trust
    messages = [{"role": "system", "content": LLM_SYSTEM_PROMPT}] if mode == "llm" else []
    was_placeable = obs.was_placeable

    while not result.done and steps < 15:
        if mode == "interactive":
            action = get_interactive_action()
        elif mode == "llm":
            action = get_llm_action(obs, messages, llm_url, model)
        else:
            action = get_random_action(obs)

        prev_trust = obs.trust
        result = env.step(action)
        obs = result.observation
        total_reward += result.reward
        steps += 1

        print_observation(obs, steps, result.reward, action_taken=action, prev_trust=prev_trust)

        if result.done:
            break

    print_episode_summary(episode_num, total_reward, steps, obs, was_placeable)
    return total_reward, steps, obs.stage == "submitted", was_placeable


def main():
    parser = argparse.ArgumentParser(description="Visualize recruiting episodes step-by-step")
    parser.add_argument("--mode", default="interactive", choices=["interactive", "llm", "random"])
    parser.add_argument("--env-url", default="http://localhost:8000", help="Environment server URL")
    parser.add_argument("--llm-url", default="http://localhost:8033/v1/chat/completions", help="LLM endpoint")
    parser.add_argument("--model", default="qwen2.5-32b-instruct-q5_k_m-00001-of-00006.gguf", help="Model name")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run")
    parser.add_argument("--show-jobs", action="store_true", help="Show jobs table every step (not just step 0)")
    args = parser.parse_args()

    all_rewards = []
    all_successes = 0
    regrettable = 0
    intrinsic = 0

    with RecruitopenenvEnv(base_url=args.env_url).sync() as env:
        for ep in range(1, args.episodes + 1):
            reward, steps, placed, was_placeable = run_episode(
                env, ep, args.mode,
                llm_url=args.llm_url, model=args.model,
            )
            all_rewards.append(reward)
            if placed:
                all_successes += 1
            elif was_placeable:
                regrettable += 1
            else:
                intrinsic += 1

    # Aggregate summary for multi-episode runs
    if args.episodes > 1:
        section("Aggregate Results")
        avg = sum(all_rewards) / len(all_rewards)
        print(f"  Episodes:            {args.episodes}")
        print(f"  Avg reward:          {reward_color(avg)}")
        print(f"  Placement rate:      {GREEN}{all_successes}/{args.episodes} ({100*all_successes/args.episodes:.0f}%){RESET}")
        print(f"  Regrettable fails:   {RED}{regrettable}{RESET} (had good match, didn't place)")
        print(f"  Intrinsic fails:     {YELLOW}{intrinsic}{RESET} (no good match existed)")
        hr("═")


if __name__ == "__main__":
    main()
