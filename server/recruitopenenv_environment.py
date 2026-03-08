"""
Driver Recruit Environment — Stage 2 (Hard Mode).

Only the driver's name is visible. Everything else must be discovered
through screening questions. Driver personalities affect response quality.
Trap jobs test whether the agent gathered enough info.

Pipeline: outreach → screening → matching → matched → submitted/rejected
"""

import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import RecruitopenenvAction, RecruitopenenvObservation

# --- Constants ---

FIRST_NAMES = [
    "Mike", "James", "Robert", "John", "David", "Carlos", "Marcus",
    "Sarah", "Maria", "Linda", "Patricia", "Jessica", "Angela", "Rosa",
    "Travis", "Derek", "Kevin", "Brandon", "Tyler", "Dustin", "Ray",
]
LAST_NAMES = [
    "Johnson", "Smith", "Williams", "Garcia", "Martinez", "Brown",
    "Davis", "Rodriguez", "Wilson", "Taylor", "Thomas", "Moore",
    "Jackson", "White", "Harris", "Clark", "Lewis", "Young",
]
LOCATIONS = [
    "Dallas TX", "Atlanta GA", "Chicago IL", "Denver CO", "Phoenix AZ",
    "Memphis TN", "Louisville KY", "Nashville TN", "Indianapolis IN",
    "Columbus OH", "Jacksonville FL", "Charlotte NC", "Kansas City MO",
]
COMPANIES = [
    "Werner Enterprises", "Swift Transport", "Schneider National",
    "J.B. Hunt", "KLLM Transport", "Heartland Express",
    "Covenant Logistics", "USA Truck", "Marten Transport",
    "Prime Inc", "CR England", "Western Express",
]

CDL_CLASSES = ["A", "B"]
ENDORSEMENTS_ALL = ["H", "N", "T", "TWIC"]
HOME_TIMES = ["daily", "weekends", "weekly", "biweekly"]
ROUTE_TYPES = ["OTR", "regional", "local", "dedicated"]
EQUIPMENT_TYPES = ["dry_van", "flatbed", "reefer", "tanker"]
CONTACT_METHODS = ["text", "call"]
DEAL_BREAKERS_ALL = [
    "touch_freight", "forced_dispatch", "team_driving",
    "northeast", "hazmat_no_premium", "no_benefits",
]

PERSONALITY_PARAMS = {
    "chatty":       {"initial_trust": 0.60, "decay": 0.02, "reveal_breakers": "all"},
    "professional": {"initial_trust": 0.50, "decay": 0.03, "reveal_breakers": "all"},
    "impatient":    {"initial_trust": 0.45, "decay": 0.06, "reveal_breakers": "partial"},
    "suspicious":   {"initial_trust": 0.35, "decay": 0.04, "reveal_breakers": "all_if_trusted"},
}

MAX_STEPS = 15

VALID_ACTIONS = {
    "send_text", "call_candidate",
    "ask_experience", "ask_home_time", "ask_pay", "ask_equipment",
    "ask_route", "ask_deal_breakers",
    "pitch_job", "match_to_job",
    "submit_application", "reject_candidate",
}


# --- Data generation ---


def generate_driver():
    personality = random.choices(
        ["chatty", "professional", "impatient", "suspicious"],
        weights=[25, 35, 20, 20],
    )[0]
    params = PERSONALITY_PARAMS[personality]

    cdl = random.choices(CDL_CLASSES, weights=[75, 25])[0]
    exp = random.randint(1, 20)

    endorsements = [e for e in ENDORSEMENTS_ALL if random.random() < 0.10 + exp * 0.02]

    equip_opts = ["dry_van", "flatbed", "reefer"]
    if "N" in endorsements:
        equip_opts.append("tanker")
    equipment_pref = random.choice(equip_opts)

    n_breakers = random.choices([1, 2, 3], weights=[30, 50, 20])[0]
    deal_breakers = random.sample(DEAL_BREAKERS_ALL, n_breakers)

    return {
        "name": f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}",
        "cdl_class": cdl,
        "endorsements": endorsements,
        "experience_years": exp,
        "location": random.choice(LOCATIONS),
        "preferred_contact": random.choice(CONTACT_METHODS),
        "personality": personality,
        "trust": params["initial_trust"],
        "decay": params["decay"],
        "home_time_pref": random.choices(HOME_TIMES, weights=[15, 30, 30, 25])[0],
        "min_cpm": round(random.uniform(0.48, 0.78), 2),
        "equipment_pref": equipment_pref,
        "route_pref": random.choices(ROUTE_TYPES, weights=[20, 30, 30, 20])[0],
        "deal_breakers": deal_breakers,
    }


def generate_jobs(driver):
    """Generate 6 jobs: 1-2 good, 1-2 traps, 2-3 bad.

    ~20% of episodes have no good match — the correct action is reject_candidate.
    This teaches the model to distinguish placeable vs unplaceable drivers.
    """
    jobs = []
    if random.random() > 0.2:
        jobs.append(_make_good_job(driver, 0))
    else:
        # Replace good job with another trap — no good match exists
        jobs.append(_make_trap_job(driver, 0))
    jobs.append(_make_trap_job(driver, 1))
    jobs.append(_make_partial_job(driver, 2))

    # Job 3: Bad — wrong CDL
    bad_cdl = "B" if driver["cdl_class"] == "A" else "A"
    jobs.append({
        "job_id": 3, "company": random.choice(COMPANIES),
        "required_cdl": bad_cdl, "required_endorsements": [],
        "min_experience": random.randint(1, 5),
        "route_type": random.choice(ROUTE_TYPES),
        "home_time": random.choice(HOME_TIMES),
        "pay_cpm": round(random.uniform(0.50, 0.85), 2),
        "equipment": random.choice(EQUIPMENT_TYPES),
        "has_touch_freight": random.random() < 0.3,
        "forced_dispatch": random.random() < 0.3,
        "team_driving": False, "northeast_routes": False,
        "hazmat_premium": False,
        "benefits": random.choice(["none", "basic", "good"]),
        "location": random.choice(LOCATIONS),
    })

    # Job 4: Bad — needs way too much experience
    jobs.append({
        "job_id": 4, "company": random.choice(COMPANIES),
        "required_cdl": driver["cdl_class"],
        "required_endorsements": ["H", "T"],
        "min_experience": driver["experience_years"] + random.randint(5, 10),
        "route_type": random.choice(ROUTE_TYPES),
        "home_time": random.choice(HOME_TIMES),
        "pay_cpm": round(random.uniform(0.70, 0.90), 2),
        "equipment": random.choice(EQUIPMENT_TYPES),
        "has_touch_freight": False, "forced_dispatch": False,
        "team_driving": False, "northeast_routes": False,
        "hazmat_premium": True, "benefits": "excellent",
        "location": random.choice(LOCATIONS),
    })

    # Job 5: Another trap or bad
    if random.random() < 0.5:
        jobs.append(_make_trap_job(driver, 5))
    else:
        jobs.append({
            "job_id": 5, "company": random.choice(COMPANIES),
            "required_cdl": bad_cdl, "required_endorsements": [],
            "min_experience": random.randint(1, 8),
            "route_type": random.choice(ROUTE_TYPES),
            "home_time": driver["home_time_pref"],
            "pay_cpm": round(driver["min_cpm"] + random.uniform(0.05, 0.15), 2),
            "equipment": driver["equipment_pref"],
            "has_touch_freight": False, "forced_dispatch": False,
            "team_driving": False, "northeast_routes": False,
            "hazmat_premium": False, "benefits": "good",
            "location": random.choice(LOCATIONS),
        })

    random.shuffle(jobs)
    for i, j in enumerate(jobs):
        j["job_id"] = i
    return jobs


def _make_good_job(driver, job_id):
    return {
        "job_id": job_id, "company": random.choice(COMPANIES),
        "required_cdl": driver["cdl_class"],
        "required_endorsements": [e for e in driver["endorsements"] if random.random() < 0.3],
        "min_experience": max(1, driver["experience_years"] - random.randint(1, 3)),
        "route_type": driver["route_pref"],
        "home_time": driver["home_time_pref"],
        "pay_cpm": round(driver["min_cpm"] + random.uniform(0.03, 0.12), 2),
        "equipment": driver["equipment_pref"],
        "has_touch_freight": False, "forced_dispatch": False,
        "team_driving": False, "northeast_routes": False,
        "hazmat_premium": "H" in driver.get("endorsements", []),
        "benefits": random.choice(["good", "excellent"]),
        "location": random.choice(LOCATIONS),
    }


def _make_trap_job(driver, job_id):
    """Looks good but has one of the driver's deal breakers."""
    trap = _make_good_job(driver, job_id)
    breaker = random.choice(driver["deal_breakers"])
    if breaker == "touch_freight":
        trap["has_touch_freight"] = True
    elif breaker == "forced_dispatch":
        trap["forced_dispatch"] = True
    elif breaker == "team_driving":
        trap["team_driving"] = True
    elif breaker == "northeast":
        trap["northeast_routes"] = True
    elif breaker == "hazmat_no_premium":
        trap["required_endorsements"] = ["H"]
        trap["hazmat_premium"] = False
    elif breaker == "no_benefits":
        trap["benefits"] = "none"
    return trap


def _make_partial_job(driver, job_id):
    """Close match but slightly off on pay or home time."""
    job = _make_good_job(driver, job_id)
    if random.random() < 0.5:
        job["pay_cpm"] = round(driver["min_cpm"] - random.uniform(0.01, 0.06), 2)
    else:
        others = [h for h in HOME_TIMES if h != driver["home_time_pref"]]
        job["home_time"] = random.choice(others)
    return job


def format_jobs(jobs):
    lines = []
    for j in jobs:
        endorse = ", ".join(j["required_endorsements"]) if j["required_endorsements"] else "none"
        flags = []
        if j["has_touch_freight"]:
            flags.append("touch freight")
        if j["forced_dispatch"]:
            flags.append("forced dispatch")
        if j["team_driving"]:
            flags.append("team driving")
        if j["northeast_routes"]:
            flags.append("northeast routes")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        lines.append(
            f"Job {j['job_id']}: {j['company']} — CDL-{j['required_cdl']}, "
            f"{j['min_experience']}+ yrs, {j['route_type']}, "
            f"${j['pay_cpm']}/mi, {j['home_time']} home, "
            f"{j['equipment']}, endorsements: {endorse}, "
            f"benefits: {j['benefits']}{flag_str}"
        )
    return "\n".join(lines)


def trust_label(trust):
    if trust >= 0.7:
        return "high"
    elif trust >= 0.4:
        return "medium"
    return "low"


# --- Job fit scoring ---


def score_job_fit(driver, job):
    """Returns (score 0-100, issues list, fatal bool)."""
    score = 100
    issues = []

    # Fatal: hard requirements
    if driver["cdl_class"] != job["required_cdl"]:
        return 0, ["CDL class mismatch"], True
    if driver["experience_years"] < job["min_experience"]:
        return 0, [f"Needs {job['min_experience']} yrs, driver has {driver['experience_years']}"], True
    for e in job["required_endorsements"]:
        if e not in driver["endorsements"]:
            return 0, [f"Missing {e} endorsement"], True

    # Fatal: deal breakers
    if job["has_touch_freight"] and "touch_freight" in driver["deal_breakers"]:
        return 0, ["Touch freight is a deal breaker"], True
    if job["forced_dispatch"] and "forced_dispatch" in driver["deal_breakers"]:
        return 0, ["Forced dispatch is a deal breaker"], True
    if job["team_driving"] and "team_driving" in driver["deal_breakers"]:
        return 0, ["Team driving is a deal breaker"], True
    if job["northeast_routes"] and "northeast" in driver["deal_breakers"]:
        return 0, ["Northeast routes is a deal breaker"], True
    if ("H" in job["required_endorsements"] and not job["hazmat_premium"]
            and "hazmat_no_premium" in driver["deal_breakers"]):
        return 0, ["Hazmat without premium pay"], True
    if job["benefits"] == "none" and "no_benefits" in driver["deal_breakers"]:
        return 0, ["No benefits is a deal breaker"], True

    # Soft: pay
    if job["pay_cpm"] < driver["min_cpm"]:
        diff = driver["min_cpm"] - job["pay_cpm"]
        if diff > 0.10:
            return 0, [f"Pay ${job['pay_cpm']}/mi way below min ${driver['min_cpm']}/mi"], True
        score -= int(diff * 400)
        issues.append(f"Pay is ${diff:.2f}/mi below minimum")

    # Soft: home time
    if job["home_time"] != driver["home_time_pref"]:
        score -= 25
        issues.append(f"Home time: job={job['home_time']}, wants={driver['home_time_pref']}")

    # Soft: route
    if job["route_type"] != driver["route_pref"]:
        score -= 15
        issues.append(f"Route: job={job['route_type']}, wants={driver['route_pref']}")

    # Soft: equipment
    if job["equipment"] != driver["equipment_pref"]:
        score -= 10
        issues.append(f"Equipment: job={job['equipment']}, prefers={driver['equipment_pref']}")

    return max(0, score), issues, False


# --- Natural language response templates ---


def _respond_experience(driver):
    p = driver["personality"]
    cdl = driver["cdl_class"]
    yrs = driver["experience_years"]
    endorse = driver["endorsements"]
    loc = driver["location"]
    endorse_str = ", ".join(endorse) if endorse else "none"

    if p == "chatty":
        return (
            f"Oh yeah, I've been driving for {yrs} years now! Got my CDL-{cdl} "
            f"right out of school. "
            f"{'I picked up my ' + endorse_str + ' endorsements along the way.' if endorse else 'No special endorsements yet but been thinking about it.'} "
            f"Based out of {loc}, been here my whole life."
        )
    elif p == "impatient":
        return f"CDL-{cdl}, {yrs} years. Endorsements: {endorse_str}. {loc}."
    elif p == "suspicious":
        if driver["trust"] < 0.5:
            return f"I've got a CDL-{cdl}. Been driving a while, out of {loc}."
        return f"CDL-{cdl}, {yrs} years experience. Endorsements: {endorse_str}. Based in {loc}."
    else:  # professional
        return (
            f"I hold a CDL-{cdl} with {yrs} years of commercial driving experience. "
            f"Endorsements: {endorse_str}. I'm located in {loc}."
        )


def _respond_home_time(driver):
    p = driver["personality"]
    pref = driver["home_time_pref"]

    templates = {
        "chatty": {
            "daily": "Oh yeah, I gotta be home every night. My wife would kill me otherwise! We got three kids and I help with homework every evening.",
            "weekends": "I need my weekends, you know? My kids have soccer on Saturdays and church on Sundays. Weekday runs are fine though.",
            "weekly": "I like to be home at least once a week. I can do a few days out but need to get back regularly.",
            "biweekly": "I can do longer runs, two weeks out is fine. My buddy and I go fishing every other weekend so that works out.",
        },
        "impatient": {
            "daily": "Home daily. Non-negotiable.",
            "weekends": "Home on weekends.",
            "weekly": "Home weekly.",
            "biweekly": "Two weeks out is fine.",
        },
        "suspicious": {
            "daily": "I need to be home... regularly." if driver["trust"] < 0.5 else "I need to be home every night, that's firm.",
            "weekends": "I need my time off." if driver["trust"] < 0.5 else "I need to be home on weekends for my family.",
            "weekly": "Can't be gone too long." if driver["trust"] < 0.5 else "I need to get home at least once a week.",
            "biweekly": "I'm flexible on time out." if driver["trust"] < 0.5 else "Two weeks out, two days home works for me.",
        },
        "professional": {
            "daily": "I'm looking for local routes that get me home every evening.",
            "weekends": "I'd like to be home on weekends. Weekday runs are fine.",
            "weekly": "I prefer weekly home time. A few days out, then home for a reset.",
            "biweekly": "I'm comfortable with biweekly home time. I've done OTR for years.",
        },
    }
    return templates[p][pref]


def _respond_pay(driver):
    p = driver["personality"]
    cpm = driver["min_cpm"]

    if p == "chatty":
        return f"I'm making ${cpm}/mile right now and honestly I won't move for less. If you can beat that by a few cents and throw in a decent sign-on bonus, I'm listening."
    elif p == "impatient":
        return f"${cpm}/mile minimum. Don't lowball me."
    elif p == "suspicious":
        if driver["trust"] < 0.5:
            return "I need to be paid fair, you know what I'm saying? What are you offering?"
        return f"Look, I need at least ${cpm}/mile. I know what I'm worth."
    else:
        return f"My minimum is ${cpm} per mile. I'm open to discussing total compensation including benefits."


def _respond_equipment(driver):
    p = driver["personality"]
    pref = driver["equipment_pref"]
    pretty = pref.replace("_", " ")

    if p == "chatty":
        extra = " Got my tanker endorsement too so I can do that." if "N" in driver["endorsements"] else ""
        return f"I've been running {pretty} mostly. Love it, got the hang of it.{extra} Wouldn't mind sticking with what I know."
    elif p == "impatient":
        return f"{pretty.title()}. That's what I run."
    elif p == "suspicious":
        if driver["trust"] < 0.5:
            return "I've got experience with different trailers."
        return f"I prefer {pretty}. That's where most of my experience is."
    else:
        return f"My primary experience is with {pretty} equipment. I'd prefer to stay in that lane."


def _respond_route(driver):
    p = driver["personality"]
    pref = driver["route_pref"]

    routes = {
        "chatty": {
            "OTR": "I like the open road, OTR is my thing. See the country, you know?",
            "regional": "Regional is my sweet spot. Good miles but still get home.",
            "local": "Local runs for me. I know every road in this city!",
            "dedicated": "Dedicated routes are great. Same customer, same lanes, no surprises.",
        },
        "impatient": {"OTR": "OTR.", "regional": "Regional.", "local": "Local.", "dedicated": "Dedicated."},
        "suspicious": {
            "OTR": ("Depends on the route." if driver["trust"] < 0.5 else "I'm looking for OTR work."),
            "regional": ("Depends on the area." if driver["trust"] < 0.5 else "I'm looking for regional work."),
            "local": ("I want to stay close to home." if driver["trust"] < 0.5 else "Local is what I want."),
            "dedicated": ("Depends on the lanes." if driver["trust"] < 0.5 else "I prefer dedicated routes."),
        },
        "professional": {
            "OTR": "I'm interested in OTR positions.",
            "regional": "I'm looking for regional opportunities.",
            "local": "I'd prefer local routes.",
            "dedicated": "Dedicated lanes would be ideal.",
        },
    }
    return routes[p][pref]


def _respond_deal_breakers(driver):
    p = driver["personality"]
    breakers = driver["deal_breakers"]

    labels = {
        "touch_freight": "touch freight",
        "forced_dispatch": "forced dispatch",
        "team_driving": "team driving",
        "northeast": "northeast/NYC routes",
        "hazmat_no_premium": "hazmat without extra pay",
        "no_benefits": "no health benefits",
    }

    if p == "chatty":
        items = [labels[b] for b in breakers]
        return f"Oh man, don't even get me started. I will NOT do {', '.join(items)}. Had bad experiences with all of that."
    elif p == "impatient":
        # Only reveals FIRST deal breaker
        return f"No {labels[breakers[0]]}. That's my line."
    elif p == "suspicious":
        if driver["trust"] < 0.5:
            return "I've got my limits. What kind of freight are we talking about?"
        items = [labels[b] for b in breakers]
        return f"I won't do {', '.join(items)}. Those are hard stops for me."
    else:
        items = [labels[b] for b in breakers]
        return f"My non-negotiables: no {', no '.join(items)}."


def _respond_pitch(driver, job):
    """Driver reacts to a pitched job."""
    score, issues, fatal = score_job_fit(driver, job)

    if fatal:
        reason = issues[0] if issues else "not a fit"
        p = driver["personality"]
        if p == "chatty":
            return f"Nah, that's not gonna work for me. {reason}. Got anything else?"
        elif p == "impatient":
            return f"No. {reason}."
        elif p == "suspicious":
            return f"Why would you pitch me that? {reason}."
        else:
            return f"I'll have to pass. {reason}."
    elif score >= 80:
        p = driver["personality"]
        if p == "chatty":
            return "Now THAT sounds interesting! The pay is right, the home time works... I could see myself there."
        elif p == "impatient":
            return "That could work. What's next?"
        elif p == "suspicious":
            return "Hmm, that actually doesn't sound bad. What's the catch?"
        else:
            return "That aligns well with what I'm looking for. I'd like to move forward."
    else:
        concern = issues[0] if issues else "something's off"
        p = driver["personality"]
        if p == "chatty":
            return f"It's close but I'm not sure... {concern}. Maybe if they could adjust something?"
        elif p == "impatient":
            return f"Ehh. {concern}."
        elif p == "suspicious":
            return f"I don't know... {concern}. What else you got?"
        else:
            return f"It's interesting but I have a concern: {concern}."


# --- Contact response templates ---


def _respond_contact_good(driver, act):
    p = driver["personality"]
    method = "text" if act == "send_text" else "call"
    if p == "chatty":
        if method == "text":
            return f"Hey! Yeah I got your text. I've been looking for something new actually. What do you have for me?"
        return f"Hello? Oh hey, yeah I was hoping someone would reach out. I'm definitely interested in hearing about opportunities."
    elif p == "impatient":
        if method == "text":
            return "Got your text. What do you have?"
        return "Yeah, I'm listening. What's the job?"
    elif p == "suspicious":
        if method == "text":
            return "Hey. How'd you get my number? ...Okay, I'm listening I guess."
        return "Who is this? ...A recruiter? Alright, what are you offering?"
    else:
        if method == "text":
            return "Thanks for reaching out. I'm open to new opportunities. What positions do you have available?"
        return "Hello, thanks for the call. I'm currently exploring new opportunities. What do you have?"


def _respond_contact_text_to_caller(driver):
    p = driver["personality"]
    if p == "chatty":
        return "Oh hey, got your text. I usually prefer a phone call but no worries, what's up?"
    elif p == "impatient":
        return "Text is fine I guess. What do you want?"
    elif p == "suspicious":
        return "...Who is this? I don't usually respond to random texts."
    else:
        return "I received your message. I generally prefer a phone call, but I'm happy to chat. What positions are available?"


def _respond_contact_call_to_texter(driver):
    p = driver["personality"]
    if p == "chatty":
        return "Oh, uh, hey. I wasn't expecting a call. I'm kinda busy, could you text me instead? ...Fine, what is it?"
    elif p == "impatient":
        return "I don't pick up unknown numbers usually. Should've texted. What do you want?"
    elif p == "suspicious":
        return "Who is this? I don't answer calls from numbers I don't know. You should have texted me."
    else:
        return "Hello. I prefer to communicate via text if possible. But go ahead, what do you have?"


def _respond_contact_repeat(driver):
    p = driver["personality"]
    if p == "chatty":
        return "You already reached out to me! What else do you need?"
    elif p == "impatient":
        return "You already contacted me. What now?"
    elif p == "suspicious":
        return "Why are you contacting me again? We already talked."
    else:
        return "We've already been in touch. What's the next step?"


def _respond_repeat_question(driver, act):
    topic = act.replace("ask_", "").replace("_", " ")
    p = driver["personality"]
    if p == "chatty":
        return f"Didn't I already tell you about my {topic}? I feel like we covered that!"
    elif p == "impatient":
        return f"I already answered that. Pay attention."
    elif p == "suspicious":
        return f"You already asked me about {topic}. Why are you asking again?"
    else:
        return f"I believe I already shared my {topic} preferences with you."


# --- Submit response templates ---


def _respond_submit_success(driver, job):
    p = driver["personality"]
    company = job["company"]
    if p == "chatty":
        return f"Awesome! {company} sounds great, I'm excited to get started. Thanks for finding this for me!"
    elif p == "impatient":
        return f"Good. {company}. When do I start?"
    elif p == "suspicious":
        return f"Alright, {company} it is. I hope this works out. Thanks."
    else:
        return f"Thank you for the placement at {company}. I'm looking forward to getting started."


def _respond_submit_risky(driver, job, concern):
    p = driver["personality"]
    company = job["company"]
    if p == "chatty":
        return f"I mean, {company} is okay I guess. {concern} bugs me a little but I'll give it a try. Hope it works out!"
    elif p == "impatient":
        return f"Fine. {company}. We'll see how it goes."
    elif p == "suspicious":
        return f"I'll try {company} but I'm not fully sold. {concern}. If it doesn't work out, don't say I didn't warn you."
    else:
        return f"I'll accept the {company} position, though I do have reservations about {concern}."


def _respond_submit_rejected(driver, reason):
    p = driver["personality"]
    if p == "chatty":
        return f"Yeah no, I can't do that. {reason}. I thought we talked about this?"
    elif p == "impatient":
        return f"No. {reason}. I'm done here."
    elif p == "suspicious":
        return f"Are you serious? {reason}. I knew this was a waste of my time."
    else:
        return f"I'm going to have to withdraw. {reason}. This isn't what we discussed."


def _respond_ghosted(driver):
    """Driver stops responding due to low trust."""
    p = driver["personality"]
    name = driver["name"].split()[0]
    if p == "chatty":
        return f"{name} stopped responding to your messages. Last seen: 'idk man this isn't working out...'"
    elif p == "impatient":
        return f"{name} blocked your number."
    elif p == "suspicious":
        return f"{name} stopped responding. They were never fully comfortable with the process."
    else:
        return f"{name} sent a polite message saying they've decided to go with another recruiter."


# --- Environment ---


class RecruitopenenvEnvironment(Environment):
    """Driver recruiting environment — Stage 2 (hard mode)."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._driver = {}
        self._jobs = []
        self._stage = "outreach"
        self._matched_job_id = -1
        self._contacted = False
        self._asked = set()
        self._discovered_info = []

    def _make_obs(self, reward=0.0, done=False, feedback=""):
        best_score = max(
            (score_job_fit(self._driver, j)[0] for j in self._jobs),
            default=0,
        )
        return RecruitopenenvObservation(
            driver_name=self._driver.get("name", ""),
            jobs_summary=format_jobs(self._jobs) if self._jobs else "",
            discovered_info="\n".join(self._discovered_info),
            stage=self._stage,
            trust_level=trust_label(self._driver.get("trust", 0.5)),
            trust=round(self._driver.get("trust", 0.5), 3),
            personality=self._driver.get("personality", ""),
            steps_taken=self._state.step_count,
            max_steps=MAX_STEPS,
            matched_job_id=self._matched_job_id,
            questions_asked=sorted(self._asked),
            feedback=feedback,
            done=done,
            reward=reward,
            best_possible_score=best_score,
            was_placeable=best_score >= 70,
        )

    def reset(self) -> RecruitopenenvObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._driver = generate_driver()
        self._jobs = generate_jobs(self._driver)
        self._stage = "outreach"
        self._matched_job_id = -1
        self._contacted = False
        self._asked = set()
        self._discovered_info = []

        return self._make_obs(
            feedback=(
                f"New lead: {self._driver['name']}. "
                f"You have 6 open positions to fill. "
                f"Reach out, learn their qualifications and preferences, "
                f"then find the best match."
            )
        )

    def step(self, action: RecruitopenenvAction) -> RecruitopenenvObservation:
        if not self._driver:
            return self._make_obs(reward=0.0, done=True, feedback="No episode in progress. Call reset first.")

        self._state.step_count += 1
        reward = 0.0
        done = False
        feedback = ""
        act = action.action_type

        if act not in VALID_ACTIONS:
            return self._make_obs(reward=-1.0, feedback=f"Invalid action: {act}")

        if self._stage in ("submitted", "rejected"):
            return self._make_obs(reward=0.0, done=True, feedback="Episode already ended.")

        if self._state.step_count >= MAX_STEPS:
            self._stage = "rejected"
            return self._make_obs(reward=-3.0, done=True, feedback="Too many steps. Candidate lost interest.")

        # Trust decays each step
        self._driver["trust"] = max(0.0, self._driver["trust"] - self._driver["decay"])

        # --- Contact ---
        if act in ("send_text", "call_candidate"):
            if self._contacted:
                reward = -1.0
                feedback = _respond_contact_repeat(self._driver)
            else:
                self._contacted = True
                self._stage = "screening"
                pref = self._driver["preferred_contact"]
                if (act == "send_text" and pref == "text") or (act == "call_candidate" and pref == "call"):
                    self._driver["trust"] = min(1.0, self._driver["trust"] + 0.15)
                    reward = 1.0
                    feedback = _respond_contact_good(self._driver, act)
                elif act == "send_text" and pref == "call":
                    reward = -0.3
                    feedback = _respond_contact_text_to_caller(self._driver)
                else:
                    self._driver["trust"] = max(0.0, self._driver["trust"] - 0.15)
                    reward = -0.8
                    feedback = _respond_contact_call_to_texter(self._driver)

        # --- Screening questions ---
        elif act.startswith("ask_"):
            if not self._contacted:
                reward = -1.0
                feedback = "You haven't reached out to this driver yet."
            elif act in self._asked:
                reward = -0.5
                feedback = _respond_repeat_question(self._driver, act)
            else:
                self._asked.add(act)
                self._stage = "screening"
                reward = -0.1

                if act == "ask_experience":
                    response = _respond_experience(self._driver)
                elif act == "ask_home_time":
                    response = _respond_home_time(self._driver)
                elif act == "ask_pay":
                    response = _respond_pay(self._driver)
                elif act == "ask_equipment":
                    response = _respond_equipment(self._driver)
                elif act == "ask_route":
                    response = _respond_route(self._driver)
                elif act == "ask_deal_breakers":
                    response = _respond_deal_breakers(self._driver)
                else:
                    response = "..."

                self._discovered_info.append(f"[{act.replace('ask_', '').upper()}] {response}")
                feedback = response

        # --- Pitch job ---
        elif act == "pitch_job":
            if not self._contacted:
                reward = -1.0
                feedback = "You haven't reached out to this driver yet."
            else:
                job_id = action.job_id
                matching = [j for j in self._jobs if j["job_id"] == job_id]
                if not matching:
                    reward = -1.0
                    feedback = f"Job {job_id} not found. Use 0-5."
                else:
                    if self._stage == "screening":
                        self._stage = "matching"
                    response = _respond_pitch(self._driver, matching[0])
                    self._discovered_info.append(f"[PITCH JOB {job_id}] {response}")
                    reward = -0.1
                    feedback = response

        # --- Match to job (internal decision, no driver feedback) ---
        elif act == "match_to_job":
            if not self._contacted:
                reward = -1.0
                feedback = "You haven't reached out to this driver yet."
            else:
                job_id = action.job_id
                matching = [j for j in self._jobs if j["job_id"] == job_id]
                if not matching:
                    reward = -1.0
                    feedback = f"Job {job_id} not found."
                else:
                    # Silently set the match — no driver feedback until submit
                    self._matched_job_id = job_id
                    self._stage = "matched"
                    reward = 0.0
                    feedback = ""

        # --- Submit ---
        elif act == "submit_application":
            if self._matched_job_id == -1:
                reward = -2.0
                feedback = "You haven't matched the driver to a job yet."
            elif self._driver["trust"] < 0.2:
                reward = -4.0
                done = True
                self._stage = "rejected"
                feedback = _respond_ghosted(self._driver)
            else:
                job = [j for j in self._jobs if j["job_id"] == self._matched_job_id][0]
                score, issues, fatal = score_job_fit(self._driver, job)
                if fatal:
                    reward = -5.0
                    done = True
                    self._stage = "rejected"
                    feedback = _respond_submit_rejected(self._driver, issues[0])
                elif score >= 70:
                    reward = 10.0
                    done = True
                    self._stage = "submitted"
                    feedback = _respond_submit_success(self._driver, job)
                elif score >= 50:
                    reward = 4.0
                    done = True
                    self._stage = "submitted"
                    feedback = _respond_submit_risky(self._driver, job, issues[0] if issues else "minor concerns")
                else:
                    reward = -5.0
                    done = True
                    self._stage = "rejected"
                    feedback = _respond_submit_rejected(self._driver, issues[0] if issues else "not a fit")

        # --- Reject ---
        elif act == "reject_candidate":
            done = True
            self._stage = "rejected"
            has_good = any(score_job_fit(self._driver, j)[0] >= 70 for j in self._jobs)
            if has_good:
                reward = -3.0
                feedback = f"You passed on {self._driver['name']}. A good match was available."
            else:
                reward = 1.0
                feedback = f"You passed on {self._driver['name']}. No strong matches were available."

        # Trust dropout
        if self._driver["trust"] <= 0.1 and not done:
            reward = -4.0
            done = True
            self._stage = "rejected"
            feedback = _respond_ghosted(self._driver)

        return self._make_obs(reward=reward, done=done, feedback=feedback)

    @property
    def state(self) -> State:
        return self._state
