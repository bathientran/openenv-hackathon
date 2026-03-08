"""Test a full recruiting episode with the new hard-mode environment."""

from recruitopenenv import RecruitopenenvEnv, RecruitopenenvAction


def run_episode():
    with RecruitopenenvEnv(base_url="http://localhost:8000").sync() as env:
        # 1. Reset
        result = env.reset()
        obs = result.observation
        print("=== NEW EPISODE ===")
        print(f"Driver: {obs.driver_name}")
        print(f"Feedback: {obs.feedback}")
        print(f"Jobs:\n{obs.jobs_summary}")
        print()

        total_reward = 0.0

        # 2. Contact — we don't know preference so just try text
        action = RecruitopenenvAction(action_type="send_text")
        result = env.step(action)
        obs = result.observation
        total_reward += result.reward
        print(f"Step 1 (send_text): reward={result.reward}, trust={obs.stage}")
        print(f"  Feedback: {obs.feedback}")
        print()

        # 3. Ask experience — need to know CDL, years, endorsements
        result = env.step(RecruitopenenvAction(action_type="ask_experience"))
        obs = result.observation
        total_reward += result.reward
        print(f"Step 2 (ask_experience): reward={result.reward}")
        print(f"  Response: {obs.feedback}")
        print()

        # 4. Ask home time
        result = env.step(RecruitopenenvAction(action_type="ask_home_time"))
        obs = result.observation
        total_reward += result.reward
        print(f"Step 3 (ask_home_time): reward={result.reward}")
        print(f"  Response: {obs.feedback}")
        print()

        # 5. Ask pay
        result = env.step(RecruitopenenvAction(action_type="ask_pay"))
        obs = result.observation
        total_reward += result.reward
        print(f"Step 4 (ask_pay): reward={result.reward}")
        print(f"  Response: {obs.feedback}")
        print()

        # 6. Ask deal breakers
        result = env.step(RecruitopenenvAction(action_type="ask_deal_breakers"))
        obs = result.observation
        total_reward += result.reward
        print(f"Step 5 (ask_deal_breakers): reward={result.reward}")
        print(f"  Response: {obs.feedback}")
        print()

        # 7. Print discovered info so far
        print("--- DISCOVERED INFO ---")
        print(obs.discovered_info)
        print(f"Trust: {obs.stage}, Stage: {obs.stage}")
        print()

        # 8. Try pitching job 0 to see reaction
        result = env.step(RecruitopenenvAction(action_type="pitch_job", job_id=0))
        obs = result.observation
        total_reward += result.reward
        print(f"Step 6 (pitch_job 0): reward={result.reward}")
        print(f"  Response: {obs.feedback}")
        print()

        # 9. Try matching job 0
        result = env.step(RecruitopenenvAction(action_type="match_to_job", job_id=0))
        obs = result.observation
        total_reward += result.reward
        print(f"Step 7 (match_to_job 0): reward={result.reward}")
        print(f"  Feedback: {obs.feedback}")
        print()

        # 10. Submit if matched
        if result.reward >= 0:
            result = env.step(RecruitopenenvAction(action_type="submit_application"))
            obs = result.observation
            total_reward += result.reward
            print(f"Step 8 (submit): reward={result.reward}, done={result.done}")
            print(f"  Feedback: {obs.feedback}")
        else:
            print("No match found, trying other jobs...")
            for job_id in range(1, 6):
                result = env.step(RecruitopenenvAction(action_type="match_to_job", job_id=job_id))
                obs = result.observation
                total_reward += result.reward
                print(f"  match_to_job {job_id}: reward={result.reward} — {obs.feedback}")
                if result.reward >= 0:
                    result = env.step(RecruitopenenvAction(action_type="submit_application"))
                    obs = result.observation
                    total_reward += result.reward
                    print(f"  submit: reward={result.reward} — {obs.feedback}")
                    break

        print(f"\n=== TOTAL REWARD: {total_reward} ===")


if __name__ == "__main__":
    run_episode()
