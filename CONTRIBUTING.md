# Contributing

Thank you for helping. Three kinds of contribution matter most:

1. **Evaluation items and ratings from clinicians.** Write to info@metacare.ai.
2. **Training records** that follow docs/DATA.md. Never include anyone's real health information.
3. **Code**: export verification for Granite hybrid models, measurements on real phones, better checks.

## Pull requests

- Branch from `develop` and open the pull request into `develop`. `main` holds released versions.
- Run `make check` (lint and tests) before you push; CI runs the same.
- A change to `ALLOWED_LICENCES`, `FORBIDDEN`, the emergency filter or the system prompt needs a reason in the pull
  request. The system prompt must stay identical to the app's: run `msm prompt --app <path>`.
- Data pull requests list the sources used and confirm that every record was reviewed.

## Conduct

Be kind, be precise, and assume good intent. Report problems to info@metacare.ai.
