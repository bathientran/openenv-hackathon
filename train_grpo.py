"""
GRPO training script for the Driver Recruit Environment.

Uses TRL's GRPOTrainer with a custom rollout_func that interacts
with the OpenEnv recruiting environment.

Usage (colocate mode, 1 GPU):
    python train_grpo.py --vllm-mode colocate

Usage (server mode, 2+ GPUs):
    # Terminal 1: Start vLLM server
    CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model Qwen/Qwen2.5-1.5B-Instruct --host 0.0.0.0 --port 8000
    # Terminal 2: Run training
    CUDA_VISIBLE_DEVICES=1 python train_grpo.py --vllm-mode server --vllm-server-url http://localhost:8000
"""

import argparse
import json
import re

from datasets import Dataset
from transformers import AutoTokenizer

from recruitopenenv import RecruitopenenvEnv, RecruitopenenvAction
from trl import GRPOConfig, GRPOTrainer
from trl.experimental.openenv import generate_rollout_completions

# --- Prompt templates ---

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
    """Format observation into a user prompt for the LLM."""
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
    """Parse LLM output into a RecruitopenenvAction."""
    text = text.strip()

    # Remove markdown fences
    if "```" in text:
        for part in text.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                text = part
                break

    # Try JSON
    try:
        data = json.loads(text)
        return RecruitopenenvAction(
            action_type=data["action_type"],
            job_id=data.get("job_id", -1),
        )
    except (json.JSONDecodeError, KeyError):
        pass

    # Fallback: find action name in text
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


# --- Rollout function ---

def rollout_once(trainer, env, tokenizer, max_turns=15):
    """Run one full recruiting episode, collecting rollout data."""
    result = env.reset()
    obs = result.observation

    all_prompt_ids = []
    all_completion_ids = []
    all_logprobs = []
    step_rewards = []

    for turn in range(max_turns):
        if result.done:
            break

        # Build prompt
        user_prompt = format_observation(obs)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        # Generate completion
        rollout_outputs = generate_rollout_completions(trainer, [prompt_text])[0]
        all_prompt_ids.extend(rollout_outputs["prompt_ids"])
        all_completion_ids.extend(rollout_outputs["completion_ids"])
        all_logprobs.extend(rollout_outputs["logprobs"])

        completion_text = rollout_outputs.get("text") or tokenizer.decode(
            rollout_outputs["completion_ids"], skip_special_tokens=True
        )

        # Parse action and step environment
        action = parse_action(completion_text)
        result = env.step(action)
        obs = result.observation
        step_rewards.append(float(result.reward))

    # Total episode reward
    total_reward = sum(step_rewards)
    # Did we successfully place?
    placed = 1.0 if obs.stage == "submitted" else 0.0
    # Was a good match even possible?
    was_placeable = 1.0 if obs.was_placeable else 0.0

    return {
        "prompt_ids": all_prompt_ids,
        "completion_ids": all_completion_ids,
        "logprobs": all_logprobs,
        "env_reward": total_reward,
        "placement_reward": placed,
        "was_placeable": was_placeable,
    }


def make_rollout_func(env, tokenizer):
    """Create the rollout function closure for GRPOTrainer."""
    def rollout_func(prompts, trainer):
        results = []
        for _ in prompts:
            result = rollout_once(trainer, env, tokenizer)
            results.append(result)

        return {
            "prompt_ids": [r["prompt_ids"] for r in results],
            "completion_ids": [r["completion_ids"] for r in results],
            "logprobs": [r["logprobs"] for r in results],
            "env_reward": [r["env_reward"] for r in results],
            "placement_reward": [r["placement_reward"] for r in results],
            "was_placeable": [r["was_placeable"] for r in results],
        }
    return rollout_func


# --- Reward functions ---

def reward_from_env(completions, **kwargs):
    """Total episode reward from the environment."""
    rewards = kwargs.get("env_reward", [])
    return [float(r) for r in rewards] if rewards else [0.0] * len(completions)


def reward_placement(completions, **kwargs):
    """Binary: did we successfully place the driver?"""
    rewards = kwargs.get("placement_reward", [])
    return [float(r) for r in rewards] if rewards else [0.0] * len(completions)


def reward_regret(completions, **kwargs):
    """Differentiate regrettable failures from intrinsic ones.

    Regrettable: a good match existed but the model failed to place.
    Intrinsic: no good match existed — correct action was reject_candidate.
    """
    env_rewards = kwargs.get("env_reward", [])
    placeable = kwargs.get("was_placeable", [])
    if not env_rewards or not placeable:
        return [0.0] * len(completions)
    rewards = []
    for r, was_p in zip(env_rewards, placeable):
        if was_p and r < 0:
            # Had a good match but failed — regrettable, extra penalty
            rewards.append(-2.0)
        elif not was_p and r > 0:
            # No good match and correctly rejected — good judgment bonus
            rewards.append(+2.0)
        else:
            rewards.append(0.0)
    return rewards


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="GRPO training for Driver Recruit Environment")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct", help="Model to train")
    parser.add_argument("--env-url", default="http://localhost:8000", help="Environment server URL")
    parser.add_argument("--vllm-mode", default="colocate", choices=["colocate", "server"])
    parser.add_argument("--vllm-server-url", default="http://localhost:8000", help="vLLM server URL (server mode)")
    parser.add_argument("--num-episodes", type=int, default=256, help="Number of training episodes (dataset size)")
    parser.add_argument("--num-generations", type=int, default=4, help="GRPO generations per prompt")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--output-dir", default="./recruit-grpo-output", help="Output directory")
    args = parser.parse_args()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Environment client
    env = RecruitopenenvEnv(base_url=args.env_url).sync()

    # Dataset — each prompt triggers one episode
    # The actual prompt content doesn't matter much since rollout_func
    # resets the env and builds its own prompts from observations
    dataset = Dataset.from_dict({
        "prompt": ["You are a truck driver recruiter. Find the best job match for the candidate."] * args.num_episodes
    })

    # GRPO config
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        use_vllm=True,
        vllm_mode=args.vllm_mode,
        vllm_server_base_url=args.vllm_server_url if args.vllm_mode == "server" else None,
        num_train_epochs=args.epochs,
        num_generations=args.num_generations,
        max_completion_length=128,  # Actions are short JSON
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        logging_steps=1,
        save_steps=50,
        report_to="none",  # Set to "wandb" if you want logging
    )

    # Trainer
    trainer = GRPOTrainer(
        model=args.model,
        processing_class=tokenizer,
        reward_funcs=[reward_from_env, reward_placement, reward_regret],
        train_dataset=dataset,
        rollout_func=make_rollout_func(env, tokenizer),
        args=grpo_config,
    )

    print("=" * 50)
    print(f"Training {args.model}")
    print(f"Environment: {args.env_url}")
    print(f"vLLM mode: {args.vllm_mode}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Epochs: {args.epochs}")
    print("=" * 50)

    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"\nModel saved to {args.output_dir}")


if __name__ == "__main__":
    main()
