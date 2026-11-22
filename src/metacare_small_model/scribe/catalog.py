"""Made up visit specs, each with its right answers (MET-1929).

Every visit is drawn from a template with random values, so the structured right answers are
known before any script is written: no model labels anything. Everything is fictional.

A spec has:
  speakers  letter -> who they are (no names; the script may invent first names)
  items     the right answers, each {kind, fhir, say, check}:
              kind   what it is (medication, observation, condition, order, device, ...)
              fhir   the FHIR resource it becomes (MedicationRequest, Observation, ...)
              fields the structured values
              say    how the visit must state it, in plain words
              check  values that must appear in the script (numbers accept digits or words)
  style     how the conversation goes
"""

from __future__ import annotations

import random

AGES = [(22, 35), (36, 50), (51, 65), (66, 80), (81, 92)]
STYLES = [
    "calm and routine",
    "the patient is anxious and asks many questions",
    "the patient talks fast and rambles",
    "the patient is hard of hearing and asks for things to be repeated",
    "the patient pushes back on starting a new medicine",
    "the clinician is in a hurry but thorough",
    "the patient is tired and answers briefly",
    "friendly, with some small talk at the start",
]
FAMILY = ["his daughter", "her son", "a spouse", "an adult grandchild", "a home health aide"]


def med(rng, name, dose, unit, freq, action="start", route="by mouth", why=None, duration=None):
    words = {"start": "start", "increase": "increase to", "decrease": "lower to", "stop": "stop", "continue": "continue"}[action]
    say = (
        f"{words} {name}"
        + ("" if action == "stop" else f" {dose} {unit} {freq}")
        + (f" for {duration}" if duration else "")
        + (f" for {why}" if why else "")
    )
    check = [name] + ([] if action == "stop" else [dose])
    return dict(
        kind="medication",
        fhir="MedicationRequest" if action != "continue" else "MedicationStatement",
        fields=dict(name=name, dose=dose, unit=unit, frequency=freq, route=route, action=action, duration=duration, reason=why),
        say=say,
        check=check,
    )


def obs(name, value, unit="", say=None):
    return dict(
        kind="observation",
        fhir="Observation",
        fields=dict(name=name, value=value, unit=unit),
        say=say or f"{name} {value}{(' ' + unit) if unit else ''}",
        check=[str(value)],
    )


def cond(name, say=None, check=None):
    return dict(kind="condition", fhir="Condition", fields=dict(name=name), say=say or name, check=check or [name])


def order(name, say=None, check=None, fhir="ServiceRequest"):
    return dict(kind="order", fhir=fhir, fields=dict(name=name), say=say or name, check=check or [name.split()[0]])


def device(name, say=None, check=None):
    return dict(kind="device", fhir="DeviceRequest", fields=dict(name=name), say=say or name, check=check or [name.split()[-1]])


def advice(text, check):
    return dict(kind="advice", fhir="CarePlan", fields=dict(text=text), say=text, check=check)


def followup(when, check):
    return dict(kind="follow up", fhir="Appointment", fields=dict(when=when), say=f"come back in {when}", check=check)


def pick(rng, items, k):
    return rng.sample(items, k=min(k, len(items)))


# --- templates: each returns (clinician, reason, items) ---------------------------------------


def hypertension(rng):
    s, d = rng.randint(142, 172), rng.randint(86, 102)
    m = rng.choice(
        [
            ("amlodipine", "5", "mg", "once a day"),
            ("lisinopril", "10", "mg", "once a day"),
            ("losartan", "50", "mg", "once a day"),
            ("chlorthalidone", "12.5", "mg", "once a day"),
        ]
    )
    items = [obs("blood pressure", f"{s} over {d}", "", f"blood pressure today {s} over {d}"), med(rng, *m, why="blood pressure")]
    items[0]["check"] = [str(s), str(d)]
    items += pick(
        rng,
        [
            order("basic metabolic panel", check=["metabolic panel"]),
            order("lipid panel", check=["lipid"]),
            device("home blood pressure cuff", check=["cuff"]),
            advice("cut back on salt", ["salt"]),
        ],
        2,
    )
    items.append(followup(rng.choice(["two weeks", "four weeks", "one month"]), []))
    return "internist", "high blood pressure", items


def diabetes(rng):
    a1c = round(rng.uniform(7.6, 10.4), 1)
    prev = round(a1c - rng.uniform(0.6, 1.8), 1)
    items = [
        obs("hemoglobin A1C", a1c, "percent"),
        obs("previous A1C", prev, "percent"),
        med(rng, "metformin", "1000", "mg", "twice a day", action="increase"),
    ]
    items.append(
        rng.choice(
            [
                med(rng, "empagliflozin", "10", "mg", "every morning"),
                med(rng, "semaglutide", "0.25", "mg", "once a week", route="by injection"),
                med(rng, "glipizide", "5", "mg", "before breakfast"),
            ]
        )
    )
    items += pick(
        rng,
        [
            order("diabetic eye exam with ophthalmology", check=["eye"]),
            order("podiatry referral", check=["podiatr"]),
            order("urine microalbumin test", check=["urine"]),
            device("glucose meter and test strips", check=["meter"]),
        ],
        2,
    )
    return "family doctor", "diabetes follow up", items


def heart_failure(rng):
    ef, lb = rng.choice([25, 30, 35, 40]), rng.randint(3, 6)
    items = [
        obs("weight gain", f"{lb} pounds in three days", "", f"gained {lb} pounds in three days"),
        obs("ejection fraction", ef, "percent"),
        med(rng, "furosemide", "40", "mg", "twice a day", action="increase", duration="one week"),
        med(rng, "metoprolol succinate", "25", "mg", "once a day", action="continue"),
        advice("keep sodium under 2000 milligrams a day", ["2,?000|two thousand"]),
        advice("call if weight goes up more than 3 pounds in a day", ["3 pounds|three pounds"]),
    ]
    items[0]["check"] = [str(lb)]
    items += pick(
        rng,
        [
            order("basic metabolic panel in one week", check=["metabolic panel"]),
            order("cardiac rehabilitation referral", check=["rehab"]),
            device("bathroom scale", check=["scale"]),
        ],
        2,
    )
    return "cardiologist", "swelling and shortness of breath", items


def copd(rng):
    sat = rng.randint(84, 88)
    items = [
        obs("oxygen saturation", sat, "percent at rest"),
        med(rng, "prednisone", "40", "mg", "once a day", duration="5 days"),
        med(rng, "tiotropium", "18", "mcg", "once a day", route="inhaled", action="continue"),
        device("home oxygen concentrator at 2 liters per minute", check=["oxygen", "2 liters|two liters"]),
        advice("no smoking anywhere near the oxygen", ["smok"]),
    ]
    items += pick(rng, [order("chest x ray", check=["x.?ray"]), order("pulmonary rehabilitation", check=["rehab"])], 1)
    return "pulmonologist", "worse breathing", items


def asthma_child(rng):
    pf = rng.choice([60, 65, 70, 75])
    items = [
        obs("peak flow", pf, "percent of personal best"),
        med(rng, "fluticasone", "44", "mcg", "two puffs twice a day", route="inhaled"),
        med(rng, "albuterol", "2", "puffs", "every 4 hours as needed", route="inhaled", action="continue"),
        advice("rinse the mouth after the steroid inhaler", ["rinse"]),
        device("spacer for the inhaler", check=["spacer"]),
        followup("four weeks", ["four weeks|4 weeks"]),
    ]
    return "pediatrician", "night time coughing", items


def ear_infection(rng):
    t, kg = rng.choice([101.8, 102.4, 103.1]), rng.randint(14, 24)
    ml = {14: "5", 15: "5.5", 16: "6", 17: "6.5", 18: "7", 19: "7", 20: "7.5", 21: "8", 22: "8", 23: "8.5", 24: "9"}[kg]
    items = [
        obs("temperature", t, "degrees"),
        obs("weight", kg, "kilograms"),
        cond("acute otitis media of the right ear", check=["otitis|ear infection"]),
        med(rng, "amoxicillin", ml, "mL", "twice a day", duration="10 days"),
        med(rng, "ibuprofen", "100", "mg per 5 mL", "every 6 hours as needed"),
        advice("come back if the fever lasts more than 48 hours on the antibiotic", ["48|forty.eight"]),
    ]
    return "pediatrician", "ear pain and fever", items


def uti(rng):
    items = [
        obs("urine dipstick nitrites", "positive", "", "urine dipstick positive for nitrites"),
        cond("urinary tract infection", check=["urinary|bladder infection|uti"]),
        med(rng, "nitrofurantoin", "100", "mg", "twice a day", duration="5 days"),
        order("urine culture", check=["culture"]),
        advice("call if fever or back pain starts", ["fever"]),
    ]
    items[0]["check"] = ["nitrite"]
    return "nurse practitioner", "burning with urination", items


def knee_post_op(rng):
    day, rom = rng.choice([10, 12, 14]), rng.choice([85, 90, 95, 100])
    items = [
        obs("range of motion", rom, "degrees"),
        med(rng, "oxycodone", "5", "mg", "every 6 hours as needed", action="decrease"),
        med(rng, "enoxaparin", "40", "mg", "once a day", route="by injection", action="continue", duration="until day 14"),
        med(rng, "aspirin", "81", "mg", "twice a day", duration="four weeks"),
        order("physical therapy three times a week", check=["physical therapy"]),
        device("front wheeled walker", check=["walker"]),
        advice("call for calf pain or swelling", ["calf"]),
    ]
    return "orthopedic surgeon", f"{day} days after a knee replacement", items


def depression(rng):
    phq, prev = rng.randint(12, 18), rng.randint(19, 24)
    items = [
        obs("PHQ 9 score", phq, "", f"PHQ 9 score {phq}, down from {prev}"),
        med(rng, "sertraline", "100", "mg", "every morning", action="increase"),
        order("referral for cognitive behavioral therapy", check=["therap"]),
        advice("call or text 988 if thoughts of self harm get stronger", ["988|nine.eight.eight"]),
        followup("four weeks", ["four weeks|4 weeks"]),
    ]
    items[0]["check"] = [str(phq)]
    return "primary care doctor", "depression follow up", items


def migraine(rng):
    days = rng.randint(8, 14)
    items = [
        obs("headache days per month", days),
        med(rng, "topiramate", "25", "mg", "at night"),
        med(rng, "sumatriptan", "50", "mg", "at the start of a migraine, at most twice a week"),
        advice("keep a headache diary", ["diary"]),
    ]
    items += pick(rng, [order("MRI of the brain", check=["mri"]), order("eye exam", check=["eye"])], 1)
    return "neurologist", "frequent headaches", items


def thyroid(rng):
    tsh = round(rng.uniform(5.4, 9.8), 1)
    old, new = rng.choice([("75", "88"), ("88", "100"), ("100", "112")])
    items = [
        obs("TSH", tsh),
        med(rng, "levothyroxine", new, "mcg", "every morning on an empty stomach", action="increase"),
        advice("take it thirty minutes before breakfast, away from calcium", ["breakfast"]),
        order("TSH again in six weeks", check=["six weeks|6 weeks"]),
    ]
    items[1]["say"] += f", up from {old} mcg"
    return "endocrinologist", "tiredness and feeling cold", items


def osteoporosis(rng):
    t = rng.choice(["minus 2.6", "minus 2.8", "minus 3.1"])
    items = [
        obs("DEXA T score at the hip", t, "", f"DEXA T score {t} at the hip"),
        cond("osteoporosis"),
        med(rng, "alendronate", "70", "mg", "once a week"),
        med(rng, "vitamin D", "2000", "units", "once a day"),
        advice("stay upright for thirty minutes after the alendronate", ["upright"]),
    ]
    items[0]["check"] = [t.split()[-1]]
    items += pick(rng, [order("mammogram", check=["mammogram"]), order("fall risk physical therapy", check=["therapy"])], 1)
    return "internist", "bone density results", items


def prenatal(rng):
    wk, g = rng.randint(26, 30), rng.randint(141, 168)
    items = [
        obs("gestational age", f"{wk} weeks", "", f"{wk} weeks pregnant"),
        obs("one hour glucose test", g),
        med(rng, "iron sulfate", "325", "mg", "every other day"),
        order("three hour glucose tolerance test", check=["three.hour|3.hour"]),
        dict(kind="immunization", fhir="Immunization", fields=dict(name="Tdap"), say="a Tdap vaccine today", check=["tdap"]),
    ]
    items[0]["check"] = [str(wk)]
    return "obstetrician", "routine prenatal visit", items


def back_pain(rng):
    items = [
        cond("acute low back strain", check=["back"]),
        med(rng, "naproxen", "500", "mg", "twice a day with food", duration="7 days"),
        med(rng, "cyclobenzaprine", "5", "mg", "at bedtime as needed"),
        order("physical therapy referral", check=["physical therapy"]),
        advice("go to the emergency room for numbness in the groin or trouble controlling the bladder", ["bladder"]),
    ]
    return "family doctor", "low back pain after lifting", items


def afib(rng):
    hr = rng.randint(96, 128)
    items = [
        obs("heart rate", hr, "beats per minute"),
        cond("atrial fibrillation", check=["atrial fibrillation|a.?fib"]),
        med(rng, "apixaban", "5", "mg", "twice a day", why="stroke prevention"),
        med(rng, "diltiazem", "120", "mg", "once a day"),
        order("echocardiogram", check=["echo"]),
        device("heart rhythm monitor patch for 14 days", check=["monitor"]),
    ]
    return "cardiologist", "a racing heartbeat", items


def cholesterol_stop(rng):
    ldl = rng.randint(150, 190)
    items = [
        obs("LDL cholesterol", ldl),
        med(rng, "simvastatin", "20", "mg", "at night", action="stop"),
        med(rng, "atorvastatin", "40", "mg", "at night"),
        order("liver tests and lipid panel in eight weeks", check=["eight weeks|8 weeks"]),
    ]
    items[1]["say"] = "stop the simvastatin because of muscle aches"
    return "internist", "cholesterol and muscle aches", items


TEMPLATES = [
    hypertension,
    diabetes,
    heart_failure,
    copd,
    asthma_child,
    ear_infection,
    uti,
    knee_post_op,
    depression,
    migraine,
    thyroid,
    osteoporosis,
    prenatal,
    back_pain,
    afib,
    cholesterol_stop,
]
CHILD = {asthma_child, ear_infection}


def make_spec(index: int, seed: int = 1929) -> dict:
    rng = random.Random(seed * 100003 + index)
    template = TEMPLATES[index % len(TEMPLATES)]
    clinician, reason, items = template(rng)
    if template in CHILD:
        speakers = {"D": clinician, "P": f"parent of a {rng.randint(3, 10)} year old"}
        if rng.random() < 0.4:
            speakers["C"] = "the child"
    else:
        lo, hi = rng.choice(AGES)
        speakers = {"D": clinician, "P": f"patient, {rng.randint(lo, hi)}"}
        if rng.random() < 0.3:
            speakers["F"] = rng.choice(FAMILY)
    rng.shuffle(items)
    return dict(
        id=f"v{index:05d}", template=template.__name__, reason=reason, speakers=speakers, style=rng.choice(STYLES), items=items
    )
