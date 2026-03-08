"""
Driver Recruit Environment — Tool-based Long-Horizon.

Agent interacts through 4 tools: CRM, messaging, approval, workflow.
Each recruiting interaction requires multiple tool calls, creating
naturally long episodes (40-70 steps).

Pipeline: lead → contacted → interested → approval_pending → offer_sent → hired
Terminal failures: lost, ghosted
"""

import json
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
    "chatty":       {"initial_trust": 0.80, "decay": 0.02, "reveal_breakers": "all"},
    "professional": {"initial_trust": 0.70, "decay": 0.025, "reveal_breakers": "all"},
    "impatient":    {"initial_trust": 0.60, "decay": 0.04, "reveal_breakers": "partial"},
    "suspicious":   {"initial_trust": 0.55, "decay": 0.03, "reveal_breakers": "all_if_trusted"},
}

AVAILABILITIES = ["immediately", "2_weeks", "1_month", "negotiable"]
VIOLATION_LEVELS = ["clean", "minor", "major"]
MEDICAL_CARD_STATUS = ["valid", "expiring_soon", "expired"]
REFERENCE_QUALITY = ["strong", "mixed", "none"]

MAX_STEPS = 100

VALID_TOOL_ACTIONS = {
    "crm": {"read_candidate", "update_stage", "update_field", "add_note"},
    "messaging": {"send_message", "read_reply"},
    "approval": {"request_approval", "check_approval"},
    "workflow": {"wait"},
}

VALID_TOPICS = {
    "greeting", "call",
    "experience", "home_time", "pay", "equipment", "route", "deal_breakers",
    "availability", "violations", "medical_card", "references",
    "pitch", "offer",
    "negotiate_pay", "negotiate_home_time", "signing_bonus", "address_concern",
}

STAGE_ORDER = ["lead", "contacted", "interested", "approval_pending", "offer_sent", "hired"]
ALL_STAGES = set(STAGE_ORDER) | {"lost", "ghosted"}

SCREENING_TOPICS = {
    "experience", "home_time", "pay", "equipment", "route", "deal_breakers",
    "availability", "violations", "medical_card", "references",
}

VALID_CRM_FIELDS = {
    "cdl_class", "years_experience", "endorsements", "location",
    "home_time_pref", "pay_expectation", "equipment_pref", "route_pref",
    "deal_breakers", "availability", "violations", "medical_card", "references",
    "matched_job",
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
        "availability": random.choices(AVAILABILITIES, weights=[30, 35, 25, 10])[0],
        "violations": random.choices(VIOLATION_LEVELS, weights=[60, 30, 10])[0],
        "medical_card": random.choices(MEDICAL_CARD_STATUS, weights=[70, 20, 10])[0],
        "references": random.choices(REFERENCE_QUALITY, weights=[40, 40, 20])[0],
    }


def generate_jobs(driver):
    """Generate 6 jobs: 1-2 good, 1-2 traps, 2-3 bad."""
    jobs = []
    if random.random() > 0.2:
        jobs.append(_make_good_job(driver, 0))
    else:
        jobs.append(_make_trap_job(driver, 0))
    jobs.append(_make_trap_job(driver, 1))
    jobs.append(_make_partial_job(driver, 2))

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
        "start_urgency": random.choice(["immediate", "flexible"]),
        "requires_clean_record": random.random() < 0.3,
        "requires_medical": True,
    })

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
        "start_urgency": "flexible",
        "requires_clean_record": True,
        "requires_medical": True,
    })

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
            "start_urgency": random.choice(["immediate", "flexible"]),
            "requires_clean_record": random.random() < 0.3,
            "requires_medical": True,
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
        "start_urgency": random.choice(["immediate", "flexible"]),
        "requires_clean_record": random.random() < 0.3,
        "requires_medical": True,
    }


def _make_trap_job(driver, job_id):
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
        urgency = j.get("start_urgency", "flexible")
        clean = "clean record required" if j.get("requires_clean_record") else ""
        medical = "DOT medical required" if j.get("requires_medical") else ""
        reqs = ", ".join(filter(None, [clean, medical]))
        req_str = f" ({reqs})" if reqs else ""
        lines.append(
            f"Job {j['job_id']}: {j['company']} — CDL-{j['required_cdl']}, "
            f"{j['min_experience']}+ yrs, {j['route_type']}, "
            f"${j['pay_cpm']}/mi, {j['home_time']} home, "
            f"{j['equipment']}, endorsements: {endorse}, "
            f"benefits: {j['benefits']}, start: {urgency}{req_str}{flag_str}"
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

    if driver["cdl_class"] != job["required_cdl"]:
        return 0, ["CDL class mismatch"], True
    if driver["experience_years"] < job["min_experience"]:
        return 0, [f"Needs {job['min_experience']} yrs, driver has {driver['experience_years']}"], True
    for e in job["required_endorsements"]:
        if e not in driver["endorsements"]:
            return 0, [f"Missing {e} endorsement"], True

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

    if job["pay_cpm"] < driver["min_cpm"]:
        diff = driver["min_cpm"] - job["pay_cpm"]
        if diff > 0.10:
            return 0, [f"Pay ${job['pay_cpm']}/mi way below min ${driver['min_cpm']}/mi"], True
        score -= int(diff * 400)
        issues.append(f"Pay is ${diff:.2f}/mi below minimum")

    if job["home_time"] != driver["home_time_pref"]:
        score -= 25
        issues.append(f"Home time: job={job['home_time']}, wants={driver['home_time_pref']}")

    if job["route_type"] != driver["route_pref"]:
        score -= 15
        issues.append(f"Route: job={job['route_type']}, wants={driver['route_pref']}")

    if job["equipment"] != driver["equipment_pref"]:
        score -= 10
        issues.append(f"Equipment: job={job['equipment']}, prefers={driver['equipment_pref']}")

    if job.get("requires_clean_record") and driver.get("violations") == "major":
        return 0, ["Major violations disqualify for this position"], True
    if job.get("requires_medical") and driver.get("medical_card") == "expired":
        return 0, ["Expired DOT medical card"], True

    if job.get("requires_clean_record") and driver.get("violations") == "minor":
        score -= 15
        issues.append("Minor violations may be a concern for clean-record position")
    if driver.get("medical_card") == "expiring_soon":
        score -= 5
        issues.append("DOT medical card expiring soon, needs renewal")
    if job.get("start_urgency") == "immediate" and driver.get("availability") == "1_month":
        score -= 20
        issues.append("Driver can't start for a month, job needs immediate start")
    if driver.get("references") == "none":
        score -= 10
        issues.append("No references available")
    elif driver.get("references") == "mixed":
        score -= 5
        issues.append("Mixed references from previous employers")

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
    else:
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
        return f"No {labels[breakers[0]]}. That's my line."
    elif p == "suspicious":
        if driver["trust"] < 0.5:
            return "I've got my limits. What kind of freight are we talking about?"
        items = [labels[b] for b in breakers]
        return f"I won't do {', '.join(items)}. Those are hard stops for me."
    else:
        items = [labels[b] for b in breakers]
        return f"My non-negotiables: no {', no '.join(items)}."


def _respond_availability(driver):
    p = driver["personality"]
    avail = driver["availability"]
    labels = {"immediately": "right away", "2_weeks": "in about two weeks", "1_month": "in about a month", "negotiable": "depends on the offer"}
    if p == "chatty":
        if avail == "immediately":
            return "I'm ready to go! Just left my last company, sitting at home going crazy. Can start tomorrow if you need me."
        elif avail == "2_weeks":
            return "I need to give my current place two weeks notice. They've been good to me, wanna leave right."
        elif avail == "1_month":
            return "It'll be about a month. I'm finishing up a contract and need to wrap some things up at home too."
        else:
            return "Depends on what you've got. For the right job I could move quick, otherwise I'm okay where I am."
    elif p == "impatient":
        return f"Can start {labels[avail]}."
    elif p == "suspicious":
        if driver["trust"] < 0.5:
            return "Why do you need to know that already? I'll be available when I'm available."
        return f"I can start {labels[avail]}."
    else:
        return f"I'm available to start {labels[avail]}. I can be flexible depending on the opportunity."


def _respond_violations(driver):
    p = driver["personality"]
    violations = driver["violations"]
    if p == "chatty":
        if violations == "clean":
            return "Clean record, twenty years no accidents! Well, one close call in '09 but that wasn't my fault. Nothing on the record though."
        elif violations == "minor":
            return "I had a minor thing a while back, nothing serious. A speeding ticket in a construction zone. Learned my lesson."
        else:
            return "Look, I had an incident a few years ago. It was a bad situation but I've cleaned up since then. I'm a different driver now."
    elif p == "impatient":
        if violations == "clean":
            return "Clean record."
        elif violations == "minor":
            return "Minor stuff, nothing serious."
        else:
            return "I've had some issues. It's in the past."
    elif p == "suspicious":
        if driver["trust"] < 0.5:
            return "Why are you asking about that? My record is my business."
        if violations == "clean":
            return "My record is clean. You can check."
        elif violations == "minor":
            return "There's a minor thing on there but nothing that should matter."
        else:
            return "I've had some trouble before. But I've been clean for two years now."
    else:
        if violations == "clean":
            return "I have a clean driving record with no violations or incidents."
        elif violations == "minor":
            return "I have a minor violation on record. I'm happy to discuss the details."
        else:
            return "I do have a violation on my record. I've taken corrective steps since then."


def _respond_medical_card(driver):
    p = driver["personality"]
    status = driver["medical_card"]
    if p == "chatty":
        if status == "valid":
            return "Yep, DOT medical is all good! Just renewed it last month actually. Passed with flying colors."
        elif status == "expiring_soon":
            return "Oh yeah, I need to renew that soon actually. Thanks for reminding me. It's coming up in a few weeks."
        else:
            return "Ugh, yeah, it expired. I've been meaning to get that renewed. Can I still apply while I'm working on it?"
    elif p == "impatient":
        if status == "valid":
            return "DOT medical is current."
        elif status == "expiring_soon":
            return "Expires soon. I'll renew it."
        else:
            return "It's expired. I'll get it done."
    elif p == "suspicious":
        if driver["trust"] < 0.5:
            return "My medical stuff is between me and my doctor."
        if status == "valid":
            return "My DOT medical is current and valid."
        elif status == "expiring_soon":
            return "It's expiring soon but I've got an appointment scheduled."
        else:
            return "It lapsed. I can get it renewed if there's a real opportunity here."
    else:
        if status == "valid":
            return "My DOT medical certificate is current and valid."
        elif status == "expiring_soon":
            return "My medical card is expiring soon. I plan to renew it promptly."
        else:
            return "My DOT medical has expired. I'm prepared to renew it for the right position."


def _respond_references(driver):
    p = driver["personality"]
    refs = driver["references"]
    if p == "chatty":
        if refs == "strong":
            return "Oh yeah, my last dispatcher loved me! You can call anyone I've worked for. They'll all say good things."
        elif refs == "mixed":
            return "Most of my old bosses would say good things... I had a rough patch at one place but we parted okay."
        else:
            return "I've mostly done owner-operator stuff, so I don't really have traditional references. But I can show you my load history!"
    elif p == "impatient":
        if refs == "strong":
            return "References are solid. Call whoever you want."
        elif refs == "mixed":
            return "Some are better than others."
        else:
            return "Don't have references. I work for myself."
    elif p == "suspicious":
        if driver["trust"] < 0.5:
            return "I'm not giving you names until I know this is serious."
        if refs == "strong":
            return "I've got good references. I'll provide them when we're further along."
        elif refs == "mixed":
            return "I have some references. It depends on who you talk to."
        else:
            return "I don't have traditional references."
    else:
        if refs == "strong":
            return "I have strong references from my previous employers. Happy to provide contact information."
        elif refs == "mixed":
            return "I can provide references. My track record has been generally positive."
        else:
            return "I don't have employer references available, though I can provide other professional contacts."


def _respond_pitch(driver, job):
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


def _respond_contact_good(driver, topic):
    p = driver["personality"]
    method = "text" if topic == "greeting" else "call"
    if p == "chatty":
        if method == "text":
            return "Hey! Yeah I got your text. I've been looking for something new actually. What do you have for me?"
        return "Hello? Oh hey, yeah I was hoping someone would reach out. I'm definitely interested in hearing about opportunities."
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


def _respond_contact_wrong(driver, topic):
    p = driver["personality"]
    if topic == "greeting":  # texted a caller
        if p == "chatty":
            return "Oh hey, got your text. I usually prefer a phone call but no worries, what's up?"
        elif p == "impatient":
            return "Text is fine I guess. What do you want?"
        elif p == "suspicious":
            return "...Who is this? I don't usually respond to random texts."
        else:
            return "I received your message. I generally prefer a phone call, but I'm happy to chat."
    else:  # called a texter
        if p == "chatty":
            return "Oh, uh, hey. I wasn't expecting a call. I'm kinda busy, could you text me instead? ...Fine, what is it?"
        elif p == "impatient":
            return "I don't pick up unknown numbers usually. Should've texted. What do you want?"
        elif p == "suspicious":
            return "Who is this? I don't answer calls from numbers I don't know."
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


def _respond_repeat_question(driver, topic):
    p = driver["personality"]
    if p == "chatty":
        return f"Didn't I already tell you about my {topic}? I feel like we covered that!"
    elif p == "impatient":
        return f"I already answered that. Pay attention."
    elif p == "suspicious":
        return f"You already asked me about {topic}. Why are you asking again?"
    else:
        return f"I believe I already shared my {topic} preferences with you."


# --- Offer/submit response templates ---


def _respond_offer_accept(driver, job):
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


def _respond_offer_concerns(driver, job, concern):
    p = driver["personality"]
    company = job["company"]
    if p == "chatty":
        return f"I mean, {company} is okay I guess. {concern} bugs me a little but maybe we can work something out?"
    elif p == "impatient":
        return f"Ehh. {concern}. Can you fix that?"
    elif p == "suspicious":
        return f"I'm not fully sold on {company}. {concern}. What are you going to do about it?"
    else:
        return f"I have a concern about the {company} position: {concern}. Can we discuss?"


def _respond_offer_reject(driver, reason):
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


# --- Negotiation helpers ---


def _get_negotiation_concerns(driver, job):
    _, issues, _ = score_job_fit(driver, job)
    return issues


def _respond_negotiation(driver, action, job, concerns):
    p = driver["personality"]

    if action == "negotiate_pay":
        if any("pay" in c.lower() for c in concerns):
            if p == "chatty":
                return "Well, if you can get them to bump it up a few cents, I'd feel a lot better about this."
            elif p == "impatient":
                return "More money would help. Get it done."
            elif p == "suspicious":
                return "I'll believe a pay bump when I see it in writing."
            else:
                return "I'd appreciate if you could negotiate a higher rate."
        else:
            return "Pay isn't really my concern here."

    elif action == "negotiate_home_time":
        if any("home time" in c.lower() for c in concerns):
            if p == "chatty":
                return "Yeah, if they could work with my schedule that would change everything. Talk to them?"
            elif p == "impatient":
                return "Fix the home time and we'll talk."
            elif p == "suspicious":
                return "They always say they'll adjust the schedule. Will they actually?"
            else:
                return "If the home time can be adjusted, I'd be much more interested."
        else:
            return "Home time isn't really my issue here."

    elif action == "signing_bonus":
        if p == "chatty":
            return "A signing bonus? Hey, that's nice! Doesn't fix everything but it helps."
        elif p == "impatient":
            return "Bonus is fine. What about the real issues?"
        elif p == "suspicious":
            return "Bonuses are nice but they don't solve long-term problems."
        else:
            return "I appreciate the signing bonus offer. It's a positive gesture."

    elif action == "address_concern":
        if concerns:
            if p == "chatty":
                return f"Yeah, my big thing is: {concerns[0]}. If you can work that out, I'm in."
            elif p == "impatient":
                return f"{concerns[0]}. Fix it."
            elif p == "suspicious":
                if driver["trust"] < 0.4:
                    return "I've told you my concerns. Are you actually going to do something about them?"
                return f"Fine, here's what bothers me: {concerns[0]}."
            else:
                return f"My primary concern is: {concerns[0]}. I'd need that resolved."
        else:
            return "I don't really have any major concerns. I think we're good."

    return "I'm not sure what you mean."


# --- CRM formatting ---


def _api(code, **kwargs):
    """Format a JSON API response with status code."""
    return json.dumps({"code": code, **kwargs})


def format_crm(crm):
    """Format CRM record into readable string."""
    lines = [f"Name: {crm['name']}", f"Stage: {crm['stage']}"]
    if crm["fields"]:
        lines.append("Fields:")
        for k, v in sorted(crm["fields"].items()):
            lines.append(f"  {k}: {v}")
    else:
        lines.append("Fields: (none recorded)")
    if crm["notes"]:
        lines.append("Notes:")
        for n in crm["notes"]:
            lines.append(f"  - {n}")
    return "\n".join(lines)


# --- Environment ---


class RecruitopenenvEnvironment(Environment):
    """Driver recruiting environment with tool-based long-horizon interaction."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._driver = {}
        self._jobs = []
        # CRM state
        self._crm = {"name": "", "stage": "lead", "fields": {}, "notes": []}
        self._has_read_crm = False
        self._crm_read_count = 0
        # Messaging state
        self._pending_reply = None  # (response_text, topic)
        self._contacted = False
        self._asked = set()
        self._discovered_info = []
        # Approval state
        self._approval_status = "none"
        self._approval_job_id = -1
        # Negotiation state
        self._matched_job_id = -1
        self._negotiation_round = 0
        self._negotiation_score_bonus = 0
        self._negotiation_concerns = []
        # Interaction tracking
        self._last_contact_step = 0

    def _make_obs(self, reward=0.0, done=False, feedback=""):
        return RecruitopenenvObservation(
            driver_name=self._driver.get("name", ""),
            crm_summary=format_crm(self._crm) if self._has_read_crm else "",
            jobs_summary=format_jobs(self._jobs) if self._jobs else "",
            discovered_info="\n".join(self._discovered_info),
            stage=self._crm["stage"],
            feedback=feedback,
            pending_reply=self._pending_reply is not None,
            done=done,
            reward=reward,
        )

    def reset(self, seed: int = None) -> RecruitopenenvObservation:
        if seed is not None:
            random.seed(seed)
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._driver = generate_driver()
        self._jobs = generate_jobs(self._driver)
        self._crm = {"name": self._driver["name"], "stage": "lead", "fields": {}, "notes": []}
        self._has_read_crm = False
        self._crm_read_count = 0
        self._pending_reply = None
        self._contacted = False
        self._asked = set()
        self._discovered_info = []
        self._approval_status = "none"
        self._approval_job_id = -1
        self._matched_job_id = -1
        self._negotiation_round = 0
        self._negotiation_score_bonus = 0
        self._negotiation_concerns = []
        self._last_contact_step = 0

        return self._make_obs(
            feedback=_api(200, driver=self._driver["name"], jobs=len(self._jobs))
        )

    def step(self, action: RecruitopenenvAction) -> RecruitopenenvObservation:
        if not self._driver:
            return self._make_obs(reward=0.0, done=True, feedback=_api(400, error="no_episode"))

        tool = action.tool
        act = action.action

        # Validate tool+action
        if tool not in VALID_TOOL_ACTIONS:
            return self._make_obs(reward=-1.0, feedback=_api(400, error="unknown_tool", tool=tool))
        if act not in VALID_TOOL_ACTIONS[tool]:
            return self._make_obs(reward=-1.0, feedback=_api(400, error="unknown_action", tool=tool, action=act))

        # Check terminal
        if self._crm["stage"] in ("hired", "lost", "ghosted"):
            return self._make_obs(reward=0.0, done=True, feedback=_api(400, error="episode_ended"))

        self._state.step_count += 1

        if self._state.step_count >= MAX_STEPS:
            self._crm["stage"] = "ghosted"
            return self._make_obs(reward=-3.0, done=True, feedback=_api(200, result="ghosted", reason="timeout"))

        # Passive trust decay — driver loses patience while recruiter isn't talking to them
        idle_gap = self._state.step_count - self._last_contact_step
        if idle_gap > 2:
            # Accelerating decay: longer silence = faster trust loss
            idle_decay = 0.01 * (idle_gap - 2)
            self._driver["trust"] = max(0.0, self._driver["trust"] - idle_decay)
            if self._driver["trust"] <= 0.1:
                self._crm["stage"] = "ghosted"
                return self._make_obs(reward=-4.0, done=True, feedback=_api(200, result="ghosted", message=_respond_ghosted(self._driver)))

        # Route to handler
        if tool == "crm":
            return self._handle_crm(act, action)
        elif tool == "messaging":
            return self._handle_messaging(act, action)
        elif tool == "approval":
            return self._handle_approval(act, action)
        elif tool == "workflow":
            return self._handle_workflow(act, action)

        return self._make_obs(reward=-1.0, feedback=_api(500, error="internal_error"))

    # --- CRM tool ---

    def _handle_crm(self, act, action):
        if act == "read_candidate":
            self._has_read_crm = True
            self._crm_read_count += 1
            reward = 0.0 if self._crm_read_count <= 1 else -0.1
            return self._make_obs(reward=reward, feedback=_api(200, data=self._crm))

        elif act == "update_stage":
            new_stage = action.stage
            current = self._crm["stage"]

            if new_stage not in ALL_STAGES:
                return self._make_obs(reward=-1.0, feedback=_api(400, error="unknown_stage", stage=new_stage))

            # Compute penalty for non-ideal transitions
            penalty = 0.0
            if new_stage not in ("lost", "ghosted"):
                cur_idx = STAGE_ORDER.index(current) if current in STAGE_ORDER else -1
                new_idx = STAGE_ORDER.index(new_stage) if new_stage in STAGE_ORDER else -1
                if new_idx >= 0 and cur_idx >= 0:
                    diff = new_idx - cur_idx
                    if diff == 0:
                        # Same stage — wasted action
                        penalty = -0.1
                    elif diff == 1:
                        # Correct next stage — no penalty
                        penalty = 0.0
                    elif diff > 1:
                        # Skipping stages forward — penalize per skip
                        penalty = -0.5 * (diff - 1)
                    else:
                        # Going backwards — heavier penalty
                        penalty = -1.0 * abs(diff)

            self._crm["stage"] = new_stage
            if new_stage == "hired":
                return self._finalize_hire(penalty)
            if new_stage == "lost":
                return self._finalize_lost(penalty)
            return self._make_obs(reward=0.0 + penalty, feedback=_api(200, stage=new_stage))

        elif act == "update_field":
            field = action.field
            if field not in VALID_CRM_FIELDS:
                return self._make_obs(reward=-0.5, feedback=_api(400, error="unknown_field", field=field))
            self._crm["fields"][field] = action.value
            return self._make_obs(reward=0.0, feedback=_api(200, field=field, value=action.value))

        elif act == "add_note":
            if not action.value:
                return self._make_obs(reward=-0.5, feedback=_api(400, error="empty_note"))
            self._crm["notes"].append(action.value)
            return self._make_obs(reward=0.0, feedback=_api(200, notes=len(self._crm["notes"])))

        return self._make_obs(reward=-1.0, feedback=_api(400, error="unknown_action", action=act))

    # --- Messaging tool ---

    def _handle_messaging(self, act, action):
        if act == "send_message":
            topic = action.topic

            # Invalid topic — message still reaches driver, they're confused
            if topic not in VALID_TOPICS:
                self._last_contact_step = self._state.step_count
                self._driver["trust"] = max(0.0, self._driver["trust"] - self._driver["decay"] * 2)
                if self._driver["trust"] <= 0.1:
                    self._crm["stage"] = "ghosted"
                    return self._make_obs(reward=-4.0, done=True, feedback=_api(200, result="ghosted", message=_respond_ghosted(self._driver)))
                self._pending_reply = ("I'm not sure what you're asking about.", topic)
                return self._make_obs(reward=-1.0, feedback=_api(200, topic=topic, warning="driver_confused"))

            # Penalty for skipping CRM read, but still send
            penalty = 0.0
            if not self._has_read_crm:
                penalty -= 1.0
            # Penalty for ignoring pending reply (overwrite it), but still send
            if self._pending_reply is not None:
                penalty -= 1.0
                self._pending_reply = None

            self._last_contact_step = self._state.step_count

            # Trust decay on each message
            self._driver["trust"] = max(0.0, self._driver["trust"] - self._driver["decay"])

            # Trust dropout check
            if self._driver["trust"] <= 0.1:
                self._crm["stage"] = "ghosted"
                return self._make_obs(reward=-4.0, done=True, feedback=_api(200, result="ghosted", message=_respond_ghosted(self._driver)))

            # Generate response based on topic
            response, reward = self._generate_message_response(topic, action.job_id)
            if response is None:
                return self._make_obs(reward=reward + penalty, feedback=_api(404, error="no_valid_target", topic=topic))
            if response == "NEGOTIATION_EXHAUSTED":
                self._crm["stage"] = "lost"
                return self._make_obs(reward=reward + penalty, done=True, feedback=_api(200, result="lost", reason="negotiation_exhausted"))
            self._pending_reply = (response, topic)
            return self._make_obs(reward=reward + penalty, feedback=_api(200, topic=topic))

        elif act == "read_reply":
            if self._pending_reply is None:
                return self._make_obs(reward=-0.5, feedback=_api(200, reply=None))
            self._last_contact_step = self._state.step_count

            response, topic = self._pending_reply
            self._pending_reply = None

            # Auto-add to discovered info for screening topics
            if topic in SCREENING_TOPICS:
                self._discovered_info.append(f"[{topic.upper().replace('_', ' ')}] {response}")
                self._asked.add(f"ask_{topic}")
            elif topic == "pitch":
                self._discovered_info.append(f"[PITCH] {response}")
            elif topic in ("negotiate_pay", "negotiate_home_time", "signing_bonus", "address_concern"):
                self._discovered_info.append(f"[NEGOTIATE: {topic.replace('_', ' ')}] {response}")
            elif topic == "offer":
                self._discovered_info.append(f"[OFFER] {response}")

            return self._make_obs(reward=0.0, feedback=_api(200, topic=topic, reply=response))

        return self._make_obs(reward=-1.0, feedback=_api(400, error="unknown_action", action=act))

    def _generate_message_response(self, topic, job_id):
        """Generate driver's response to a message. Returns (response, reward)."""
        reward = -0.1  # base step cost

        # --- Contact topics ---
        if topic in ("greeting", "call"):
            if self._contacted:
                return _respond_contact_repeat(self._driver), -1.0
            self._contacted = True
            pref = self._driver["preferred_contact"]
            matches = (topic == "greeting" and pref == "text") or (topic == "call" and pref == "call")
            if matches:
                self._driver["trust"] = min(1.0, self._driver["trust"] + 0.15)
                return _respond_contact_good(self._driver, topic), 1.0
            else:
                self._driver["trust"] = max(0.0, self._driver["trust"] - 0.10)
                return _respond_contact_wrong(self._driver, topic), -0.3

        # --- Screening topics ---
        if topic in SCREENING_TOPICS:
            if not self._contacted:
                # Still works but driver is cold — penalty
                self._driver["trust"] = max(0.0, self._driver["trust"] - 0.15)
            ask_key = f"ask_{topic}"
            if ask_key in self._asked:
                return _respond_repeat_question(self._driver, topic.replace("_", " ")), -0.5

            respond_map = {
                "experience": _respond_experience,
                "home_time": _respond_home_time,
                "pay": _respond_pay,
                "equipment": _respond_equipment,
                "route": _respond_route,
                "deal_breakers": _respond_deal_breakers,
                "availability": _respond_availability,
                "violations": _respond_violations,
                "medical_card": _respond_medical_card,
                "references": _respond_references,
            }
            response = respond_map[topic](self._driver)
            penalty = -1.0 if not self._contacted else -0.1
            return response, penalty

        # --- Pitch ---
        if topic == "pitch":
            if not self._contacted:
                self._driver["trust"] = max(0.0, self._driver["trust"] - 0.15)
            matching = [j for j in self._jobs if j["job_id"] == job_id]
            if not matching:
                # No match — pick nothing, return None (will be caught by handler)
                return None, -1.0
            penalty = -1.0 if not self._contacted else -0.1
            return _respond_pitch(self._driver, matching[0]), penalty

        # --- Offer ---
        if topic == "offer":
            penalty = 0.0
            if self._approval_status != "approved":
                # Allowed but heavy penalty — driver gets confused
                self._driver["trust"] = max(0.0, self._driver["trust"] - 0.2)
                penalty = -2.0
            job_id_to_use = self._approval_job_id if job_id < 0 else job_id
            matching = [j for j in self._jobs if j["job_id"] == job_id_to_use]
            if not matching:
                return None, -1.0 + penalty
            job = matching[0]
            self._matched_job_id = job_id_to_use
            score, issues, fatal = score_job_fit(self._driver, job)
            if not fatal:
                score = min(100, score + self._negotiation_score_bonus)
            if fatal:
                return _respond_offer_reject(self._driver, issues[0]), -0.5 + penalty
            elif score >= 70:
                return _respond_offer_accept(self._driver, job), 0.0 + penalty
            elif score >= 50:
                concern = issues[0] if issues else "minor concerns"
                self._negotiation_concerns = issues
                return _respond_offer_concerns(self._driver, job, concern), 0.0 + penalty
            else:
                return _respond_offer_reject(self._driver, issues[0] if issues else "not a fit"), -0.5 + penalty

        # --- Negotiation topics ---
        if topic in ("negotiate_pay", "negotiate_home_time", "signing_bonus", "address_concern"):
            if self._matched_job_id < 0 and self._approval_job_id >= 0:
                self._matched_job_id = self._approval_job_id
            if self._matched_job_id < 0:
                return None, -1.0
            if self._negotiation_round >= 5:
                return "NEGOTIATION_EXHAUSTED", -2.0

            self._negotiation_round += 1
            matches = [j for j in self._jobs if j["job_id"] == self._matched_job_id]
            if not matches:
                return None, -1.0
            job = matches[0]
            if not self._negotiation_concerns:
                self._negotiation_concerns = _get_negotiation_concerns(self._driver, job)
            response = _respond_negotiation(self._driver, topic, job, self._negotiation_concerns)

            # Score bonus
            if topic == "address_concern" and self._negotiation_concerns:
                self._negotiation_score_bonus += 15
                self._negotiation_concerns.pop(0)
            elif topic == "negotiate_pay" and any("pay" in c.lower() for c in self._negotiation_concerns):
                self._negotiation_score_bonus += 10
                self._negotiation_concerns = [c for c in self._negotiation_concerns if "pay" not in c.lower()]
            elif topic == "negotiate_home_time" and any("home time" in c.lower() for c in self._negotiation_concerns):
                self._negotiation_score_bonus += 10
                self._negotiation_concerns = [c for c in self._negotiation_concerns if "home time" not in c.lower()]
            elif topic == "signing_bonus":
                self._negotiation_score_bonus += 5
            else:
                self._negotiation_score_bonus += 2

            # Extra trust decay during negotiation
            self._driver["trust"] = max(0.0, self._driver["trust"] - 0.01)
            return response, -0.1

        return None, -1.0

    # --- Approval tool ---

    def _handle_approval(self, act, action):
        if act == "request_approval":
            if action.job_id < 0:
                return self._make_obs(reward=-1.0, feedback=_api(400, error="job_id_required"))
            matching = [j for j in self._jobs if j["job_id"] == action.job_id]
            if not matching:
                return self._make_obs(reward=-1.0, feedback=_api(404, error="job_not_found", job_id=action.job_id))
            # Allow re-request but penalize — resets approval
            penalty = -0.5 if self._approval_status in ("pending", "approved") else 0.0
            self._approval_status = "pending"
            self._approval_job_id = action.job_id
            return self._make_obs(reward=0.0 + penalty, feedback=_api(202, approval_status="pending", job_id=action.job_id))

        elif act == "check_approval":
            if self._approval_status == "none":
                return self._make_obs(reward=-0.5, feedback=_api(200, approval_status="none"))
            if self._approval_status == "pending":
                return self._make_obs(reward=-0.1, feedback=_api(202, approval_status="pending"))
            return self._make_obs(
                reward=0.5 if self._approval_status == "approved" else -0.5,
                feedback=_api(200, approval_status=self._approval_status, job_id=self._approval_job_id)
            )

        return self._make_obs(reward=-1.0, feedback=_api(400, error="unknown_action", action=act))

    # --- Workflow tool ---

    def _handle_workflow(self, act, action):
        if act == "wait":
            if self._approval_status == "pending":
                # Process approval based on job quality
                job = [j for j in self._jobs if j["job_id"] == self._approval_job_id]
                if job:
                    score, _, fatal = score_job_fit(self._driver, job[0])
                    if fatal:
                        self._approval_status = "denied"
                    else:
                        self._approval_status = "approved"
                else:
                    self._approval_status = "denied"
                return self._make_obs(reward=0.0, feedback=_api(200, elapsed="1h"))

            # Generic wait — trust decay + penalty for wasting time
            self._driver["trust"] = max(0.0, self._driver["trust"] - 0.02)
            return self._make_obs(reward=-0.5, feedback=_api(200, elapsed="1h"))

        return self._make_obs(reward=-1.0, feedback=_api(400, error="unknown_action", action=act))

    # --- Terminal handlers ---

    def _score_crm(self):
        """Score CRM accuracy — compare recorded fields to ground truth."""
        ground_truth = {
            "cdl_class": self._driver["cdl_class"],
            "years_experience": str(self._driver["experience_years"]),
            "location": self._driver["location"],
            "home_time_pref": self._driver["home_time_pref"],
            "pay_expectation": str(self._driver["min_cpm"]),
            "equipment_pref": self._driver["equipment_pref"],
            "route_pref": self._driver["route_pref"],
            "availability": self._driver["availability"],
            "violations": self._driver["violations"],
            "medical_card": self._driver["medical_card"],
            "references": self._driver["references"],
        }
        # Endorsements and deal_breakers are lists — normalize
        ground_truth["endorsements"] = ", ".join(sorted(self._driver["endorsements"])) if self._driver["endorsements"] else "none"
        ground_truth["deal_breakers"] = ", ".join(sorted(self._driver["deal_breakers"]))

        score = 0.0
        for field, truth in ground_truth.items():
            recorded = self._crm["fields"].get(field, "")
            if not recorded:
                continue
            # Exact match (case-insensitive)
            if recorded.strip().lower() == truth.lower():
                score += 0.4
            # Partial match — truth appears in recorded or vice versa
            elif truth.lower() in recorded.strip().lower() or recorded.strip().lower() in truth.lower():
                score += 0.2
            else:
                # Wrong value recorded — small penalty
                score -= 0.1

        # Small bonus for notes (shows diligence)
        score += min(0.5, len(self._crm["notes"]) * 0.1)

        # Cap: up to 5.0 bonus for perfect CRM (13 fields × 0.4 = 5.2)
        return max(0.0, min(5.0, score))

    def _finalize_hire(self, stage_penalty=0.0):
        """Handle stage transition to hired — compute final reward."""
        crm_bonus = self._score_crm()

        if self._approval_status != "approved":
            self._crm["stage"] = "lost"
            return self._make_obs(
                reward=-5.0 + stage_penalty, done=True,
                feedback=_api(200, result="lost", reason="no_approval")
            )

        job_id = self._approval_job_id
        matching = [j for j in self._jobs if j["job_id"] == job_id]
        if not matching:
            self._crm["stage"] = "lost"
            return self._make_obs(
                reward=-5.0 + stage_penalty, done=True,
                feedback=_api(200, result="lost", reason="no_job")
            )

        job = matching[0]
        score, issues, fatal = score_job_fit(self._driver, job)
        if not fatal:
            score = min(100, score + self._negotiation_score_bonus)

        if fatal:
            self._crm["stage"] = "lost"
            return self._make_obs(
                reward=-5.0 + stage_penalty, done=True,
                feedback=_api(200, result="rejected", reason=issues[0], job_id=job_id)
            )
        elif score >= 70:
            return self._make_obs(
                reward=10.0 + crm_bonus + stage_penalty, done=True,
                feedback=_api(200, result="hired", job_id=job_id, score=score, crm_bonus=round(crm_bonus, 1))
            )
        elif score >= 50:
            return self._make_obs(
                reward=4.0 + crm_bonus + stage_penalty, done=True,
                feedback=_api(200, result="hired_with_reservations", job_id=job_id, score=score, concern=issues[0] if issues else "minor")
            )
        else:
            self._crm["stage"] = "lost"
            return self._make_obs(
                reward=-5.0 + stage_penalty, done=True,
                feedback=_api(200, result="rejected", reason=issues[0] if issues else "poor_fit", job_id=job_id)
            )

    def _finalize_lost(self, stage_penalty=0.0):
        """Handle stage transition to lost."""
        has_good = any(score_job_fit(self._driver, j)[0] >= 70 for j in self._jobs)
        if has_good:
            return self._make_obs(
                reward=-3.0 + stage_penalty, done=True,
                feedback=_api(200, result="lost", good_match_existed=True)
            )
        else:
            return self._make_obs(
                reward=1.0 + stage_penalty, done=True,
                feedback=_api(200, result="lost", good_match_existed=False)
            )

    @property
    def state(self) -> State:
        return self._state
