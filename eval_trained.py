"""Evaluate a trained model against the recruiting environment."""

import argparse
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from recruitopenenv import RecruitopenenvEnv, RecruitopenenvAction

SYSTEM_PROMPT = """You are a truck driver recruiter. You only know the driver's name. You must discover their qualifications and preferences through screening questions, then match them to the best job.

Valid actions:
- send_text: Send a text message to the candidate
- call_candidate: Call the candidate on the phone
- ask_experience: Ask about CDL class, years of experience, endorsements, location
- ask_home_time: Ask about home time preference
- ask_pay: Ask about pay expectations
- ask_equipment: Ask about equipment preference
- ask_route: Ask about route preference
- ask_deal_breakers: Ask what they absolutely won't do
- pitch_job <job_id>: Describe a job and get their reaction (0-5)
- match_to_job <job_id>: Select a job for the candidate (0-5)
- submit_application: Submit the application
- reject_candidate: No good match exists

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
    parts.append(
        f"Stage: {obs.stage} | Trust: {obs.trust_level} | "
        f"Step: {obs.steps_taken}/{obs.max_steps} | Matched: {obs.matched_job_id}"
    )
    if obs.feedback:
        parts.append(f"Feedback: {obs.feedback}")
    return "\n".join(parts)


def parse_action(text):
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


def generate(model, tokenizer, messages, device):
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./recruit-grpo-output", help="Path to trained model")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct", help="Base model for comparison")
    parser.add_argument("--env-url", default="http://localhost:8001")
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--compare", action="store_true", help="Also run base model for comparison")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    models_to_eval = [("TRAINED", args.model)]
    if args.compare:
        models_to_eval.append(("BASE", args.base_model))

    for label, model_path in models_to_eval:
        print(f"\n{'='*50}")
        print(f"Evaluating: {label} ({model_path})")
        print(f"{'='*50}")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto"
        )

        rewards = []
        successes = 0
        total_steps = 0

        with RecruitopenenvEnv(base_url=args.env_url) as env:
            for ep in range(args.num_episodes):
                result = env.reset()
                obs = result.observation
                ep_reward = 0.0
                steps = 0
                messages = [{"role": "system", "content": SYSTEM_PROMPT}]

                while not result.done and steps < 15:
                    obs_text = format_observation(obs)
                    messages.append({"role": "user", "content": obs_text})

                    response = generate(model, tokenizer, messages, device)
                    messages.append({"role": "assistant", "content": response})

                    action = parse_action(response)
                    result = env.step(action)
                    obs = result.observation
                    ep_reward += result.reward
                    steps += 1

                    print(f"  Step {steps}: {action.action_type}"
                          f"{'(' + str(action.job_id) + ')' if action.job_id >= 0 else ''}"
                          f" -> reward={result.reward:.1f}")

                rewards.append(ep_reward)
                total_steps += steps
                placed = obs.stage == "submitted"
                if placed:
                    successes += 1

                print(f"Episode {ep+1}: reward={ep_reward:.1f}, steps={steps}, "
                      f"{'SUCCESS' if placed else 'FAIL'}")
                print()

        avg_reward = sum(rewards) / len(rewards)
        avg_steps = total_steps / args.num_episodes

        print(f"\n{'='*40}")
        print(f"  {label} RESULTS")
        print(f"{'='*40}")
        print(f"Model:              {model_path}")
        print(f"Episodes:           {args.num_episodes}")
        print(f"Avg reward:         {avg_reward:.2f}")
        print(f"Min reward:         {min(rewards):.2f}")
        print(f"Max reward:         {max(rewards):.2f}")
        print(f"Placement rate:     {successes}/{args.num_episodes} ({100*successes/args.num_episodes:.1f}%)")
        print(f"Avg steps/episode:  {avg_steps:.1f}")
        print(f"{'='*40}")

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
