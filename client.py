"""Recruitopenenv Environment Client."""

from typing import Dict

from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from openenv.core import EnvClient

from .models import RecruitopenenvAction, RecruitopenenvObservation


class RecruitopenenvEnv(
    EnvClient[RecruitopenenvAction, RecruitopenenvObservation, State]
):
    """Client for the Driver Recruit Environment."""

    def _step_payload(self, action: RecruitopenenvAction) -> Dict:
        return {
            "action_type": action.action_type,
            "job_id": action.job_id,
        }

    def _parse_result(self, payload: Dict) -> StepResult[RecruitopenenvObservation]:
        obs_data = payload.get("observation", {})
        observation = RecruitopenenvObservation(
            driver_name=obs_data.get("driver_name", ""),
            jobs_summary=obs_data.get("jobs_summary", ""),
            discovered_info=obs_data.get("discovered_info", ""),
            stage=obs_data.get("stage", ""),
            trust_level=obs_data.get("trust_level", ""),
            trust=obs_data.get("trust", 0.5),
            personality=obs_data.get("personality", ""),
            steps_taken=obs_data.get("steps_taken", 0),
            max_steps=obs_data.get("max_steps", 15),
            matched_job_id=obs_data.get("matched_job_id", -1),
            questions_asked=obs_data.get("questions_asked", []),
            feedback=obs_data.get("feedback", ""),
            best_possible_score=obs_data.get("best_possible_score", 0),
            was_placeable=obs_data.get("was_placeable", True),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
