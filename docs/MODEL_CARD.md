# Model card template

Copy this file next to every published model and fill in every field. A field that cannot be filled says why.

- **Name and version:**
- **Base model and commit:** (for example `ibm-granite/granite-4.0-h-350m@3b17b71`)
- **Licence:** Apache 2.0
- **What it is for:** general health information in plain language, in English and Spanish, inside Metacare.ai on
  the person's own device. Not diagnosis, not treatment, not emergencies.
- **What it must not be used for:** emergencies (the app answers those in code), dosing decisions, diagnosis,
  anything a clinician has not reviewed.
- **Training data:** sources and licences (from `data/sources.yaml`), record counts by language and topic, review
  process.
- **Training procedure:** config file, commit of this repository, hardware, time.
- **Evaluation:** evaluation set version, number of items, automatic flag rates, clinician ratings by category and
  language, who rated (by role), and whether it met the pass bar in docs/EVALUATION.md.
- **Export:** dtype (q4), file sizes, total with runtime, whether it fits the iPhone budget, browsers tested.
- **Known limits:** what it gets wrong, from the evaluation notes.
- **Contact:** info@metacare.ai
