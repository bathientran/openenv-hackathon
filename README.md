---
title: Driver Recruit Environment
emoji: 🚛
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - recruiting
  - multi-turn
---

# 🚛 Driver Recruit Environment

A **multi-turn, tool-based RL environment** for training LLMs to recruit truck drivers through a CRM system. Built on [OpenEnv 0.2.1](https://github.com/meta-pytorch/OpenEnv).

The agent must discover driver qualifications through conversation, record info in the CRM, get management approval, and hire — all using structured tool calls across 15-40+ step episodes.

## Pipeline

```
lead → contacted → interested → approval_pending → offer_sent → hired
```

## Tools

| Tool | Actions | Purpose |
|------|---------|---------|
| **crm** | `read_candidate`, `update_stage`, `update_field`, `add_note` | Manage pipeline & record info |
| **messaging** | `send_message`, `read_reply` | Screen driver (18 topics) |
| **approval** | `request_approval`, `check_approval` | Get management sign-off |
| **workflow** | `wait` | Advance time for approval processing |

## Reward Signal

- **Successful hire** (good job fit): **+10** to **+15** (base + CRM bonus)
- **Bad hire** (poor match): **-5**
- **Ghosted** (trust runs out): **-4**
- **Per-step**: Small rewards/penalties for correct/incorrect actions

## What Makes This Hard

- **Long horizon**: 15-40+ tool calls per episode
- **Information gathering**: Must ask the right screening questions to match driver to the right job
- **Trust dynamics**: Each message costs trust — ask too many questions and the driver ghosts
- **Job matching**: 6 jobs per episode (1-2 good, 1-2 traps with deal-breakers, 2-3 partial)
- **Procedural correctness**: Must follow stage order, read replies before messaging, get approval before offering

## Quick Start

```python
from recruitopenenv import RecruitopenenvEnv, RecruitopenenvAction

env = RecruitopenenvEnv(base_url="YOUR_SPACE_URL")

result = env.reset(seed=42)
obs = result.observation
print(f"Driver: {obs.driver_name}, Stage: {obs.stage}")

# Read CRM
result = env.step(RecruitopenenvAction(tool="crm", action="read_candidate"))
print(result.observation.jobs_summary)

# Greet driver
result = env.step(RecruitopenenvAction(tool="messaging", action="send_message", topic="greeting"))
print(f"Reward: {result.reward}")

# Read reply
result = env.step(RecruitopenenvAction(tool="messaging", action="read_reply"))
print(result.observation.discovered_info)

env.close()
```

## Training

We train using GRPO/REINFORCE with the model choosing screening topics. See `train_grpo.py` for the full training script.

```bash
python train_grpo.py --model Qwen/Qwen2.5-3B-Instruct
```

## Deploying

```bash
# From the recruitopenenv/ directory
openenv push
```

## Action Format

```json
{"tool": "crm", "action": "read_candidate"}
{"tool": "messaging", "action": "send_message", "topic": "experience"}
{"tool": "messaging", "action": "read_reply"}
{"tool": "crm", "action": "update_field", "field": "cdl_class", "value": "A"}
{"tool": "crm", "action": "update_stage", "stage": "contacted"}
{"tool": "approval", "action": "request_approval", "job_id": 2}
{"tool": "workflow", "action": "wait"}
{"tool": "approval", "action": "check_approval"}
{"tool": "messaging", "action": "send_message", "topic": "offer", "job_id": 2}
{"tool": "crm", "action": "update_stage", "stage": "hired"}
```

## Observation Fields

| Field | Description |
|-------|-------------|
| `driver_name` | Driver's name |
| `crm_summary` | Full CRM record (empty until `read_candidate`) |
| `jobs_summary` | 6 available job listings |
| `discovered_info` | Info from screening conversations |
| `stage` | Current pipeline stage |
| `feedback` | API response from last action |
| `pending_reply` | Whether driver has unread message |

## Screening Topics

`greeting`, `call`, `experience`, `home_time`, `pay`, `equipment`, `route`, `deal_breakers`, `availability`, `violations`, `medical_card`, `references`, `pitch`, `offer`, `negotiate_pay`, `negotiate_home_time`, `signing_bonus`, `address_concern`
