"""
Data models for the Driver Recruit Environment.

Tool-based action interface for long-horizon recruiting pipeline.
Agent uses CRM, messaging, approval, and workflow tools.
"""

from pydantic import Field

from openenv.core.env_server.types import Action, Observation


class RecruitopenenvAction(Action):
    """Tool-based action the agent takes."""

    tool: str = Field(
        ...,
        description="Tool: crm, messaging, approval, workflow",
    )
    action: str = Field(
        ...,
        description=(
            "Action within tool. "
            "crm: read_candidate, update_stage, update_field, add_note. "
            "messaging: send_message, read_reply. "
            "approval: request_approval, check_approval. "
            "workflow: wait."
        ),
    )
    topic: str = Field(
        default="",
        description=(
            "Message topic for messaging.send_message: "
            "greeting, call, experience, home_time, pay, equipment, route, "
            "deal_breakers, availability, violations, medical_card, references, "
            "pitch, offer, negotiate_pay, negotiate_home_time, signing_bonus, address_concern"
        ),
    )
    job_id: int = Field(
        default=-1,
        description="Job index (0-5). Used with pitch, offer, request_approval.",
    )
    stage: str = Field(
        default="",
        description="Target stage for crm.update_stage: contacted, interested, approval_pending, offer_sent, hired, lost",
    )
    field: str = Field(
        default="",
        description="CRM field for crm.update_field",
    )
    value: str = Field(
        default="",
        description="Value for crm.update_field or text for crm.add_note",
    )


class RecruitopenenvObservation(Observation):
    """What the agent sees after each action."""

    driver_name: str = Field(default="", description="Driver's name")
    crm_summary: str = Field(default="", description="CRM record (empty until read_candidate)")
    jobs_summary: str = Field(default="", description="Available job listings")
    discovered_info: str = Field(default="", description="Info discovered through conversation")

    stage: str = Field(default="lead", description="Current pipeline stage")
    feedback: str = Field(default="", description="API response from last action")
    pending_reply: bool = Field(default=False, description="Whether an unread message is waiting")
