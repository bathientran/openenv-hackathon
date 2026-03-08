"""
GRPO training script for the Driver Recruit Environment.

Uses TRL's GRPOTrainer with rollout_func for multi-turn episodes.
The model controls EVERY action in the episode, not just the first one.

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


# --- Multi-turn rollout ---

ENV_URL = "http://localhost:8001"


def rollout_func(prompts, trainer):
    """Multi-turn rollout: model controls every action in the episode.

    For each prompt, we:
    1. Reset the env to get initial observation
    2. Generate an action from the model
    3. Step the env, append observation, generate next action
    4. Repeat until done or max steps
    5. Concatenate all actions into one "completion" for GRPO

    Returns dict with prompt_ids, completion_ids, logprobs, and rewards.
    """
    tokenizer = trainer.processing_class
    model = trainer.model
    device = model.device

    all_prompt_ids = []
    all_completion_ids = []
    all_logprobs = []
    all_rewards = []

    env = RecruitopenenvEnv(base_url=ENV_URL)

    for prompt_text in prompts:
        # Tokenize the initial prompt
        prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt")[0]
        all_prompt_ids.append(prompt_ids)

        # Run a full episode
        result = env.reset()
        obs = result.observation
        total_reward = 0.0

        # Build conversation as we go
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": format_observation(obs)},
        ]

        episode_completion_ids = []
        episode_logprobs = []
        steps = 0

        while not result.done and steps < 15:
            # Build current prompt from conversation
            current_prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            inputs = tokenizer(
                current_prompt, return_tensors="pt",
                truncation=True, max_length=2048
            )
            input_ids = inputs["input_ids"].to(device)

            # Generate one action
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    attention_mask=inputs["attention_mask"].to(device),
                    max_new_tokens=64,
                    temperature=1.2,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_logits=True,
                )

            # Extract new tokens and their logprobs
            new_token_ids = outputs.sequences[0][input_ids.shape[1]:]

            # Compute logprobs from logits
            if hasattr(outputs, 'logits') and outputs.logits:
                step_logprobs = []
                for i, logits in enumerate(outputs.logits):
                    if i < len(new_token_ids):
                        log_probs = torch.log_softmax(logits[0], dim=-1)
                        token_logprob = log_probs[new_token_ids[i]].item()
                        step_logprobs.append(token_logprob)
                episode_logprobs.extend(step_logprobs)
            else:
                episode_logprobs.extend([0.0] * len(new_token_ids))

            episode_completion_ids.extend(new_token_ids.tolist())

            # Decode and parse action
            response = tokenizer.decode(new_token_ids, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": response})

            action = parse_action(response)
            result = env.step(action)
            obs = result.observation
            total_reward += result.reward
            steps += 1

            # Add next observation to conversation
            if not result.done:
                messages.append({"role": "user", "content": format_observation(obs)})

        # Convert to tensors
        if episode_completion_ids:
            all_completion_ids.append(torch.tensor(episode_completion_ids))
            all_logprobs.append(episode_logprobs)
        else:
            # Empty episode fallback
            all_completion_ids.append(torch.tensor([tokenizer.eos_token_id]))
            all_logprobs.append([0.0])

        all_rewards.append(total_reward)

    env.close()

    return {
        "prompt_ids": all_prompt_ids,
        "completion_ids": all_completion_ids,
        "logprobs": all_logprobs,
        "env_reward": all_rewards,
    }


# --- Reward function (receives env_reward from rollout) ---

def reward_total(completions, env_reward=None, **kwargs):
    """Use the total episode reward computed during rollout."""
    if env_reward is not None:
        return env_reward
    # Fallback if env_reward not passed
    return [0.0] * len(completions)


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="GRPO training for Driver Recruit Environment")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct", help="Model to train")
    parser.add_argument("--env-url", default="http://localhost:8001", help="Environment server URL")
    parser.add_argument("--num-episodes", type=int, default=256, help="Number of training episodes (dataset size)")
    parser.add_argument("--num-generations", type=int, default=16, help="GRPO generations per prompt")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--output-dir", default="./recruit-grpo-output", help="Output directory")
    parser.add_argument("--use-qlora", action="store_true", help="Use QLoRA (4-bit) for memory efficiency")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    args = parser.parse_args()

    global ENV_URL
    ENV_URL = args.env_url

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Build the dataset — just needs prompt strings
    # Each prompt is a different recruiting scenario
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

    # GRPO config
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        use_vllm=False,
        num_train_epochs=args.epochs,
        num_generations=args.num_generations,
        max_completion_length=512,  # Longer for multi-turn
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        learning_rate=args.lr,
        logging_steps=1,
        save_steps=50,
        bf16=True,
        report_to="wandb",
        run_name="recruit-grpo-multiturn",
        model_init_kwargs=model_kwargs if model_kwargs else None,
    )

    # Trainer with rollout_func — model controls every turn
    trainer_kwargs = dict(
        model=args.model,
        processing_class=tokenizer,
        reward_funcs=[reward_total],
        train_dataset=dataset,
        args=grpo_config,
        rollout_func=rollout_func,
    )
    if peft_config is not None:
        trainer_kwargs["peft_config"] = peft_config

    trainer = GRPOTrainer(**trainer_kwargs)

    print("=" * 50)
    print(f"Training {args.model} (MULTI-TURN rollout)")
    print(f"Environment: {args.env_url}")
    print(f"QLoRA: {args.use_qlora}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Epochs: {args.epochs}")
    print(f"Generations per prompt: {args.num_generations}")
    print("=" * 50)

    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"\nModel saved to {args.output_dir}")


if __name__ == "__main__":
    main()
