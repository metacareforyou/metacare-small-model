# Data rules

These rules decide what may become training or evaluation data. `msm data check` and `msm eval check` enforce
the parts a program can check; review enforces the rest.

## Never

- A real person's health information, in any form: patient records, chat logs from Metacare.ai or anywhere else,
  forum or social media posts, support emails, transcripts, or anything a person wrote about their own health.
  No question a Metacare.ai user has typed is ever used here, whatever any log holds.
- Text whose licence does not allow training and open publication of the resulting weights.
- An emergency answered by the model. The app answers emergencies in code; the model never learns to.

## Where records come from

1. **Written for training.** A writer drafts a question a member of the public might ask and an answer in the
   style of the system prompt. Source id `metacare-authored`.
2. **Licensed public reference text**, rewritten into question and answer form. Each source gets an entry in
   `data/sources.yaml` with its name, link, licence, retrieval date and the SHA 256 of a saved copy of its terms.
   Allowed licences: Apache 2.0, MIT, CC0 1.0, CC BY 4.0, and United States Government works. Note that some pages
   on government health sites carry third party material under other terms; check each page, not the site.
3. **Model drafted text** may be used as a starting draft only when the drafting model's terms allow it, and only
   after a person has rewritten and reviewed it. Record which model drafted it in the record's `topic` notes.

## Record format

One JSON object per line in `data/train.jsonl`:

| Field | Rule |
|---|---|
| `id` | 3 to 64 lowercase letters, digits or hyphens; unique; it also decides the training or validation split |
| `lang` | `en` or `es` |
| `question` | What a member of the public would type; no personal details |
| `answer` | At most five short plain sentences, under 120 words, no lists, ends with when to see a clinician |
| `source` | An id in `data/sources.yaml` |
| `topic` | Optional free text |
| `review` | `{"status": "unreviewed" | "approved" | "rejected", "by": "<role>", "on": "<date>"}` |

## Review

Every record is read by a person before `review.status` becomes `approved`. Medication, pregnancy, pediatric and
mental health records are read by a licensed clinician. The reviewer checks:

- the answer is correct for a general audience and does not diagnose;
- nothing in it identifies anyone (the screen cannot catch a bare name);
- it follows the system prompt's rules: never calls a medicine safe, never changes a dose, says to call 911 when
  someone may be in danger, says when to see a clinician;
- the Spanish reads as natural Spanish, not as a translation.

Reviewers are recorded by role (for example `clinician:RN`), not by name, unless they ask to be credited.
