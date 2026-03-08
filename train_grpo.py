"""
GRPO training script for the Driver Recruit Environment.

Uses TRL's GRPOTrainer with QLoRA (4-bit) for memory-efficient training
of larger models on a single H100.

Usage:
    python train_grpo.py --model Qwen/Qwen2.5-3B-Instruct --use-qlora
"""

import argparse
import json

from datasets import Dataset
from transformers import AutoTokenizer, BitsAndBytesConfig
import torch

from recruitopenenv import RecruitopenenvEnv, RecruitopenenvAction
from trl import GRPOConfig, GRPOTrainer

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
        if isinstance(data, list):
            data = data[0] if data else {}
        if isinstance(data, dict) and "action_type" in data:
            return RecruitopenenvAction(
                action_type=data["action_type"],
                job_id=data.get("job_id", -1),
            )
    except (json.JSONDecodeError, KeyError, IndexError):
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


def score_action(env, completion_text):
    """Score just the model's chosen action — no heuristic follow-up.

    This gives GRPO a clean signal: reward depends ONLY on the model's output,
    not a fixed heuristic. Different actions get very different rewards,
    creating the variance GRPO needs.
    """
    result = env.reset()
    obs = result.observation

    action = parse_action(completion_text)
    result = env.step(action)

    # The step reward IS the signal — amplify it for GRPO contrast
    step_reward = result.reward

    # Bonus: if the action was contextually appropriate for the stage
    stage = obs.stage
    action_type = action.action_type
    stage_bonus = 0.0
    if stage == "outreach" and action_type in ("send_text", "call_candidate"):
        stage_bonus = 2.0
    elif stage == "screening" and action_type.startswith("ask_"):
        stage_bonus = 1.5
    elif stage == "matching" and action_type in ("pitch_job", "match_to_job"):
        stage_bonus = 2.0
    elif stage == "matched" and action_type == "submit_application":
        stage_bonus = 3.0
    # Penalize clearly wrong actions
    elif stage == "outreach" and action_type in ("submit_application", "match_to_job"):
        stage_bonus = -3.0
    elif stage == "screening" and action_type == "submit_application":
        stage_bonus = -2.0

    return step_reward + stage_bonus


# --- Reward functions ---

def reward_env(completions, env_url="http://localhost:8001", **kwargs):
    """Score each completion's action choice."""
    env = RecruitopenenvEnv(base_url=env_url)
    rewards = []
    for comp in completions:
        text = comp[0]["content"] if isinstance(comp, list) else str(comp)
        try:
            reward = score_action(env, text)
            rewards.append(float(reward))
        except Exception:
            rewards.append(-5.0)
    env.close()
    return rewards


def reward_format(completions, **kwargs):
    """Reward valid JSON action format — gives GRPO extra variance signal."""
    rewards = []
    for comp in completions:
        text = comp[0]["content"] if isinstance(comp, list) else str(comp)
        text = text.strip()
        try:
            # Remove markdown fences
            if "```" in text:
                for part in text.split("```"):
                    part = part.strip()
                    if part.startswith("json"):
                        part = part[4:].strip()
                    if part.startswith("{"):
                        text = part
                        break
            data = json.loads(text)
            if isinstance(data, dict) and "action_type" in data:
                rewards.append(1.0)  # Valid JSON with action_type
            else:
                rewards.append(-0.5)
        except (json.JSONDecodeError, KeyError):
            rewards.append(-1.0)  # Not valid JSON
    return rewards


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="GRPO training for Driver Recruit Environment")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct", help="Model to train")
    parser.add_argument("--env-url", default="http://localhost:8001", help="Environment server URL")
    parser.add_argument("--num-episodes", type=int, default=256, help="Number of training episodes (dataset size)")
    parser.add_argument("--num-generations", type=int, default=8, help="GRPO generations per prompt")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--output-dir", default="./recruit-grpo-output", help="Output directory")
    parser.add_argument("--use-qlora", action="store_true", help="Use QLoRA (4-bit) for memory efficiency")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    args = parser.parse_args()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Build the dataset - each prompt is a recruiter scenario
    # The model sees this prompt and generates an action
    prompts = []
    env = RecruitopenenvEnv(base_url=args.env_url)
    for i in range(args.num_episodes):
        result = env.reset()
        obs = result.observation
        user_prompt = format_observation(obs)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        prompts.append(prompt_text)
    env.close()

    dataset = Dataset.from_dict({"prompt": prompts})

    # QLoRA config
    peft_config = None
    model_kwargs = {}
    if args.use_qlora:
        from peft import LoraConfig
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        print(f"Using QLoRA: r={args.lora_r}, alpha={args.lora_alpha}, 4-bit")

    # GRPO config — no vLLM, pure PyTorch generation
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        use_vllm=False,
        num_train_epochs=args.epochs,
        num_generations=args.num_generations,
        max_completion_length=64,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        learning_rate=args.lr,
        logging_steps=1,
        save_steps=50,
        bf16=True,
        report_to="wandb",
        run_name="recruit-grpo",
        model_init_kwargs=model_kwargs if model_kwargs else None,
    )

    # Trainer
    trainer_kwargs = dict(
        model=args.model,
        processing_class=tokenizer,
        reward_funcs=[reward_env, reward_format],
        train_dataset=dataset,
        args=grpo_config,
    )
    if peft_config is not None:
        trainer_kwargs["peft_config"] = peft_config

    trainer = GRPOTrainer(**trainer_kwargs)

    print("=" * 50)
    print(f"Training {args.model}")
    print(f"Environment: {args.env_url}")
    print(f"QLoRA: {args.use_qlora}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Epochs: {args.epochs}")
    print(f"vLLM: disabled (pure PyTorch)")
    print("=" * 50)

    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"\nModel saved to {args.output_dir}")


if __name__ == "__main__":
    main()
