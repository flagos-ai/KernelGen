<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->


# Submit Feedback Skill

Submit issues to:

https://github.com/flagos-ai/skills

when users report bugs, defects, or improvements in skills.

If the user does not want to create a GitHub account, they can instead send feedback via email:

contact@flagos.io (for users without GitHub accounts)

---

# Prerequisites

For GitHub submission:

- GitHub CLI (`gh`) must be installed
- The user must be authenticated

---

# Workflow

## 1. Determine Submission Method

**Default to GitHub Issue** unless the user explicitly prefers email or says they don't have
a GitHub account. Do not ask for a preference unprompted — assume GitHub and proceed.

If the user indicates they prefer email or don't have a GitHub account, switch to the
Email Workflow.

---

# GitHub Workflow

## 2. Gather Information

Collect the following information from the user. **If the user already provided some of
this information in the conversation, reuse it instead of asking again.**

Required:

- Skill name — **Infer from conversation context if obvious** (e.g., if the user says
  "kernelgen skill crashes", auto-fill `kernelgen`). Only ask the user if the skill name
  cannot be determined from context.
- Problem description
- Expected behavior
- Actual behavior

Optional (ask, but do not block on these):

- Steps to reproduce

## 3. Auto-collect Environment Information

**Attempt** to collect environment information using the Bash tool. If the Bash tool is
unavailable or the command fails, **do not block the workflow** — mark all environment
fields as "Not detected" and continue.

```bash
if command -v python3 >/dev/null 2>&1; then
  python3 - <<'PY'
import platform, sys
print(f"OS: {platform.system()} {platform.release()}")
print(f"Python: {sys.version.split()[0]}")
try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda or 'not detected'}")
except ImportError:
    print("PyTorch: not installed")
    print("CUDA: not detected")
try:
    import triton
    print(f"Triton: {triton.__version__}")
except ImportError:
    print("Triton: not installed")
PY
elif command -v python >/dev/null 2>&1; then
  python - <<'PY'
import platform, sys
print(f"OS: {platform.system()} {platform.release()}")
print(f"Python: {sys.version.split()[0]}")
try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda or 'not detected'}")
except ImportError:
    print("PyTorch: not installed")
    print("CUDA: not detected")
try:
    import triton
    print(f"Triton: {triton.__version__}")
except ImportError:
    print("Triton: not installed")
PY
else
  echo "PYTHON_NOT_FOUND"
fi
```

If the output contains `PYTHON_NOT_FOUND`, or the command fails for any other reason,
use the fallback values and move on:

```
- OS: Not detected (environment probe failed)
- Python: Not detected (environment probe failed)
- PyTorch: Not detected (environment probe failed)
- CUDA: Not detected (environment probe failed)
- Triton: Not detected (environment probe failed)
```

Include the output (or fallback values) in the Environment section of the issue.

---

## 4. Verify GitHub CLI

First check if `gh` is installed, then check authentication:

```bash
command -v gh || echo "GH_NOT_FOUND"
```

**If the output contains the literal string `GH_NOT_FOUND`** (or is empty), **automatically
fall back to the Email Workflow** (see below). Tell the user:

    GitHub CLI (gh) was not detected on this system.
    I will prepare an email draft instead so you can send the feedback easily.

Do not stop or ask the user to install `gh` unless they explicitly want to use GitHub.

**If `gh` is installed**, check authentication:

```bash
gh auth status 2>&1
```

Parse the output to determine auth state:
- Output contains `Logged in to` or `Logged into` → authenticated, proceed.
- Output contains `not logged in` or `authentication` error → NOT authenticated.
- Output contains `token` and `expired` → token expired, treat as NOT authenticated.

If NOT authenticated, instruct the user to run:

    gh auth login

Do not proceed until authentication is confirmed.

---

## 5. Determine Issue Type and Labels

Automatically determine the issue category based on the user's report.

Label rules:

- If the report describes something broken or incorrect → label: `bug`
- If the user requests a new feature or enhancement → label: `enhancement`
- If the user suggests improvement or feedback → label: `enhancement`

Always add the `skill` label as well.

**Important**: Before using any label, verify it exists in the target repo. Use `--label`
only for labels confirmed to exist. If a label does not exist, omit it rather than causing
the `gh issue create` command to fail.

To check available labels:

```bash
gh label list --repo flagos-ai/skills --limit 200 --json name
```

This outputs a JSON array like `[{"name":"bug"},{"name":"enhancement"}]`. **Extract all
`"name"` values from the JSON array into a flat list (e.g., `["bug","enhancement","skill"]`)
and only include `--label` flags for labels that appear in that list.** Label names are
case-sensitive on GitHub — match them exactly as they appear in the JSON. If a desired label
(e.g., `bug`, `skill`) does not appear, omit it from the `gh issue create` command entirely.
Using a non-existent label will cause the command to fail.

---

## 6. Construct the Issue

Title format:

    [skill-name] Brief description

Keep titles under 80 characters (preferred: under 60 for better GitHub UI display).
**If the title exceeds 80 characters, shorten the description
portion first** (not the skill name prefix). Do not simply truncate mid-word — rephrase the
description to be more concise.

**Generate a clear, specific title** — do not use vague user phrases like "it doesn't work"
as-is. Summarize the actual problem based on the information gathered in Step 2. Examples:
- Bad: `[kernelgen] it doesn't work`
- Good: `[kernelgen] MCP call fails with schema mismatch on func_type`
- Bad: `[submit-feedback] bug`
- Good: `[submit-feedback] gh issue create fails when label does not exist`

**If insufficient information is available** to generate a specific title, use a short generic
description rather than guessing details. Only include facts the user actually stated.

**Title uniqueness**: If you are aware that an issue with the same title already exists in
the repo, slightly rephrase the description to make the title more specific and avoid
duplicates.

Body template (stored as a variable for use in Step 8):

```
## Skill

<skill-name>

## Description

<description>

## Steps to Reproduce

<steps or "Not provided">

## Expected Behavior

<expected behavior>

## Actual Behavior

<actual behavior>

## Environment

- OS: <auto-collected or "Not detected">
- Python: <auto-collected or "Not detected">
- PyTorch: <auto-collected or "Not detected">
- CUDA: <auto-collected or "Not detected">
- Triton: <auto-collected or "Not detected">

## Additional Context

<If relevant error messages, logs, or conversation context are available, include them here.
If the user pasted logs or stack traces earlier in the conversation, include them automatically.
If logs exceed ~2000 characters, truncate to the most relevant portion (typically the last
error and surrounding context) and append "... (log truncated, showing last relevant section)".
Otherwise write "None".>

---

Submitted via submit-feedback skill
```

---

## 7. Confirm With User

Show the generated issue title, body, and labels to the user and ask for confirmation before
submitting. Do not submit without explicit user approval.

---

## 8. Submit the Issue

**Use `--body-file -` with a HEREDOC** to pass the body via stdin. This avoids shell escaping
issues with multiline markdown and the "Argument list too long" error for large bodies:

**Important**: The `--label` flags in the example above are illustrative. **Only include labels
verified to exist** in Step 5. If `bug` or `skill` does not appear in the `gh label list`
output, omit that `--label` flag entirely. Example with no verified labels:

```bash
gh issue create \
  --repo flagos-ai/skills \
  --title "[skill-name] Brief description" \
  --body-file - <<'__ISSUE_BODY_FLAGOS_SKILL__'
## Skill

<skill-name>

## Description

<description>

## Steps to Reproduce

<steps or "Not provided">

## Expected Behavior

<expected behavior>

## Actual Behavior

<actual behavior>

## Environment

- OS: ...
- Python: ...
- PyTorch: ...
- CUDA: ...
- Triton: ...

## Additional Context

<error messages, logs, or relevant conversation context — or "None">

---

Submitted via submit-feedback skill
__ISSUE_BODY_FLAGOS_SKILL__
```

**Never use `"$BODY"` with shell variable expansion** or `--body "$(cat ...)"` — multiline
markdown content with backticks, quotes, and special characters will break the command.
Always use `--body-file -` with HEREDOC as shown above.

---

## 9. Confirm Creation

Show the created issue URL and number.

Example:

    Issue #42 created: https://github.com/flagos-ai/skills/issues/42

If the command fails, show the error to the user and suggest they create the issue manually
at https://github.com/flagos-ai/skills/issues/new with the body text you prepared.

---

# Email Workflow (Alternative)

The Email Workflow is used when:
- The user explicitly prefers email
- The user does not have a GitHub account
- GitHub CLI (`gh`) is not available (automatic fallback from Step 4)

Prepare an email draft and provide a `mailto:` link.

Recipient:

    contact@flagos.io

Subject:

    [Skill Feedback] <skill-name> - Brief description

Body template:

```
Skill: <skill-name>

Description:
<description>

Steps to Reproduce:
<steps or "Not provided">

Expected Behavior:
<expected behavior>

Actual Behavior:
<actual behavior>

Environment:
- OS: <auto-collected or "Not detected">
- Python: <auto-collected or "Not detected">
- PyTorch: <auto-collected or "Not detected">
- CUDA: <auto-collected or "Not detected">
- Triton: <auto-collected or "Not detected">

Additional Context:
<error messages, logs, or relevant conversation context — or "None">

Submitted via skill feedback email
```

After showing the draft, provide **two options** for the user:

1. **mailto link** (for users with a configured email client):
   ```
   mailto:contact@flagos.io?subject=[Skill Feedback] <skill-name> - Brief description&body=<url-encoded body>
   ```
   **URL-encode the body text** before inserting it into the mailto link (spaces → `%20`,
   newlines → `%0A`, etc.). Unencoded special characters will break the link.
   **Note**: `mailto:` links have a ~2000 character limit. If the URL-encoded body exceeds
   this, skip the mailto link and only offer option 2.
2. **Copy-paste**: Tell the user to copy the draft above and send it manually to
   `contact@flagos.io`. This is the most reliable option and works regardless of email
   client or body length.

---

# Rules

- Always confirm the issue or email content before sending.
- Never fabricate details. If information is missing, write "Not provided".
- Auto-collect environment info — do not ask the user for OS/Python/PyTorch versions.
- Keep submissions in English. If the user provides Chinese text, translate it before submission.
- If `gh` or authentication is not available, do not attempt workarounds — guide the user
  to install/authenticate or switch to the email workflow.
- Do not include sensitive information (API keys, tokens, passwords) in the issue body.
  If the user's logs contain secrets, redact them before submission.
