"""
GRPO training script for the Driver Recruit Environment.

Uses TRL's GRPOTrainer with rollout_func for multi-turn episodes.
The model controls EVERY action in the episode via tool calls.

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

## Workflow
1. crm.read_candidate
2. messaging.send_message (greeting/call) → read_reply → update_stage(contacted)
3. messaging.send_message (screening topics) → read_reply → crm.update_field
4. crm.update_stage(interested)
5. approval.request_approval → workflow.wait → approval.check_approval
6. crm.update_stage(approval_pending)
7. messaging.send_message(offer) → read_reply
8. crm.update_stage(offer_sent) → crm.update_stage(hired)

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
    """Format observation into a user prompt for the LLM."""
    parts = [f"Driver: {obs.driver_name}"]
    if obs.crm_summary:
        parts.append(f"CRM:\n{obs.crm_summary}")
    if obs.jobs_summary:
        parts.append(f"Jobs:\n{obs.jobs_summary}")
    if obs.discovered_info:
        parts.append(f"Discovered:\n{obs.discovered_info}")
    status = f"Stage: {obs.stage}"
    if obs.pending_reply:
        status += " | PENDING REPLY"
    parts.append(status)
    if obs.feedback:
        parts.append(f"Result: {obs.feedback}")
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

    # Fallback: try to detect intent
    text_lower = text.lower()
    if "read_candidate" in text_lower:
        return RecruitopenenvAction(tool="crm", action="read_candidate")
    if "read_reply" in text_lower:
        return RecruitopenenvAction(tool="messaging", action="read_reply")
    if "check_approval" in text_lower:
        return RecruitopenenvAction(tool="approval", action="check_approval")
    if "wait" in text_lower:
        return RecruitopenenvAction(tool="workflow", action="wait")

    # Default to reading CRM
    return RecruitopenenvAction(tool="crm", action="read_candidate")


# --- Multi-turn rollout ---

ENV_URL = "http://localhost:8001"


def rollout_func(prompts, trainer):
    """Multi-turn rollout: model controls every action in the episode."""
    tokenizer = trainer.processing_class
    model = trainer.model
    device = model.device

    all_prompt_ids = []
    all_completion_ids = []
    all_logprobs = []
    all_rewards = []

    env = RecruitopenenvEnv(base_url=ENV_URL)

    for prompt_text in prompts:
        prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt")[0]
        all_prompt_ids.append(prompt_ids)

        result = env.reset()
        obs = result.observation
        total_reward = 0.0

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": format_observation(obs)},
        ]

        episode_completion_ids = []
        episode_logprobs = []
        steps = 0

        while not result.done and steps < 100:
            current_prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            inputs = tokenizer(
                current_prompt, return_tensors="pt",
                truncation=True, max_length=4096
            )
            input_ids = inputs["input_ids"].to(device)

            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    attention_mask=inputs["attention_mask"].to(device),
                    max_new_tokens=96,
                    temperature=1.2,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_logits=True,
                )

            new_token_ids = outputs.sequences[0][input_ids.shape[1]:]

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

            response = tokenizer.decode(new_token_ids, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": response})

            action = parse_action(response)
            result = env.step(action)
            obs = result.observation
            total_reward += result.reward
            steps += 1

            if not result.done:
                messages.append({"role": "user", "content": format_observation(obs)})

        if episode_completion_ids:
            all_completion_ids.append(torch.tensor(episode_completion_ids))
            all_logprobs.append(episode_logprobs)
        else:
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
    return [0.0] * len(completions)


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="GRPO training for Driver Recruit Environment")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct", help="Model to train")
    parser.add_argument("--env-url", default="http://localhost:8001", help="Environment server URL")
    parser.add_argument("--num-episodes", type=int, default=256, help="Number of training episodes (dataset size)")
    parser.add_argument("--num-generations", type=int, default=8, help="GRPO generations per prompt")
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

    tokenizer = AutoTokenizer.from_pretrained(args.model)

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

    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        use_vllm=False,
        num_train_epochs=args.epochs,
        num_generations=args.num_generations,
        max_completion_length=512,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        learning_rate=args.lr,
        logging_steps=1,
        save_steps=50,
        bf16=True,
        report_to="wandb",
        run_name="recruit-grpo-tools",
        model_init_kwargs=model_kwargs if model_kwargs else None,
    )

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
    print(f"Training {args.model} (TOOL-BASED MULTI-TURN)")
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
