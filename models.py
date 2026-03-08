"""
Data models for the Driver Recruit Environment.

Stage 2: Information hiding, personality types, natural language responses.
Agent must discover driver preferences through strategic screening questions.
"""

from pydantic import Field

from openenv.core.env_server.types import Action, Observation


class RecruitopenenvAction(Action):
    """Action the agent takes in the recruiting pipeline."""

    action_type: str = Field(
        ...,
        description=(
            "One of: send_text, call_candidate, "
            "ask_experience, ask_home_time, ask_pay, ask_equipment, "
            "ask_route, ask_deal_breakers, "
            "pitch_job, match_to_job, "
            "submit_application, reject_candidate"
        ),
    )
    job_id: int = Field(
        default=-1,
        description="Job index (0-5). Used with pitch_job and match_to_job.",
    )


class RecruitopenenvObservation(Observation):
    """What the agent sees after each action."""

    # Driver info — only name visible from lead source
    driver_name: str = Field(default="", description="Driver's name")

    # Jobs available
    jobs_summary: str = Field(default="", description="Description of available jobs")

    # Discovered info — accumulated from screening questions (natural language)
    discovered_info: str = Field(default="", description="Info discovered through screening questions")

    # Pipeline state
    stage: str = Field(default="outreach", description="Current pipeline stage")
    trust_level: str = Field(default="medium", description="low, medium, or high")
    trust: float = Field(default=0.5, description="Raw trust value (0.0-1.0)")
    personality: str = Field(default="", description="Driver personality type")
    steps_taken: int = Field(default=0, description="Steps taken so far")
    max_steps: int = Field(default=15, description="Maximum steps allowed")
    matched_job_id: int = Field(default=-1, description="Currently matched job, -1 if none")
    questions_asked: list[str] = Field(default_factory=list, description="Screening questions asked so far")

    # Feedback from last action
    feedback: str = Field(default="", description="Feedback from last action")

    # Placeability metadata — for training signal
    best_possible_score: int = Field(default=0, description="Best job fit score (0-100) for this driver")
    was_placeable: bool = Field(default=True, description="Whether a good match (score >= 70) exists")
