"""Evaluate a trained model against the recruiting environment."""

import argparse
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from recruitopenenv import RecruitopenenvEnv, RecruitopenenvAction

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

Respond with ONLY JSON:
{"tool": "crm", "action": "read_candidate"}
{"tool": "messaging", "action": "send_message", "topic": "experience"}
{"tool": "messaging", "action": "read_reply"}
{"tool": "crm", "action": "update_field", "field": "cdl_class", "value": "A"}
{"tool": "crm", "action": "update_stage", "stage": "contacted"}
{"tool": "approval", "action": "request_approval", "job_id": 2}
{"tool": "workflow", "action": "wait"}
{"tool": "approval", "action": "check_approval"}
{"tool": "messaging", "action": "send_message", "topic": "offer", "job_id": 2}
{"tool": "crm", "action": "update_stage", "stage": "hired"}"""


def format_observation(obs):
    parts = [f"Driver: {obs.driver_name}"]
    if obs.crm_summary:
        parts.append(f"CRM:\n{obs.crm_summary}")
    if obs.jobs_summary:
        parts.append(f"Jobs:\n{obs.jobs_summary}")
    if obs.discovered_info:
        parts.append(f"Discovered:\n{obs.discovered_info}")
    status = f"Stage: {obs.stage} | Trust: {obs.trust_level} | Step: {obs.steps_taken}/{obs.max_steps}"
    if obs.pending_reply:
        status += " | PENDING REPLY"
    if obs.approval_status != "none":
        status += f" | Approval: {obs.approval_status}"
    if obs.negotiation_round > 0:
        status += f" | Negotiation: {obs.negotiation_round}/5"
    parts.append(status)
    if obs.feedback:
        parts.append(f"Result: {obs.feedback}")
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
    except (json.JSONDecodeError, KeyError, IndexError):
        pass

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

                while not result.done and steps < 75:
                    obs_text = format_observation(obs)
                    messages.append({"role": "user", "content": obs_text})

                    response = generate(model, tokenizer, messages, device)
                    messages.append({"role": "assistant", "content": response})

                    action = parse_action(response)
                    result = env.step(action)
                    obs = result.observation
                    ep_reward += result.reward
                    steps += 1

                    print(f"  Step {steps}: {action.tool}.{action.action}"
                          f"{'(' + action.topic + ')' if action.topic else ''}"
                          f"{'[job=' + str(action.job_id) + ']' if action.job_id >= 0 else ''}"
                          f" -> reward={result.reward:.1f}")

                rewards.append(ep_reward)
                total_steps += steps
                hired = obs.stage == "hired"
                if hired:
                    successes += 1

                print(f"Episode {ep+1}: reward={ep_reward:.1f}, steps={steps}, "
                      f"{'HIRED' if hired else 'FAIL (' + obs.stage + ')'}")
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
        print(f"Hire rate:          {successes}/{args.num_episodes} ({100*successes/args.num_episodes:.1f}%)")
        print(f"Avg steps/episode:  {avg_steps:.1f}")
        print(f"{'='*40}")

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
