---
name: mach6-review
description: "Review a PR for code quality, errors, tests, completeness, and simplicity. Post findings as a PR comment, then do an independent assessment to separate genuine issues from nitpicks and false positives. Usage: mach6-review 42 [aspects]"
argument-hint: "<pr-number> [code|errors|tests|completeness|simplify]"
---

# mach6-review — Multi-Aspect PR Review

**User input:** $ARGUMENTS

## Global Rules

1. **GitHub as shared memory** — Reviews and assessments are posted as PR comments so any future session can pick up context.
2. **HTML markers** — Use `<!-- mach6-review -->` and `<!-- mach6-assessment -->` as the first line of comment bodies.
3. **No `#N` in comment bodies** — Use "finding 3", "item 3", "stage 2" etc. instead.

**Important: Do NOT fix any issues in this session. Fixes happen via `/skill:mach6-implement`.**

## Step 1: Parse input

Extract:
- **PR number** (required)
- **Review aspects** (optional) — if specified, only run matching review aspects

## Step 2: Prepare

```bash
gh pr checkout <pr-number>
git pull
```

Mark the PR as ready for review (it was opened as a draft by mach6-plan):
```bash
gh pr ready <pr-number>
```

Gather PR context — read ALL comments, not just specific markers:
```bash
gh pr view <pr-number> --json title,body,comments,files
gh pr diff <pr-number>
```

Read the PR description, ALL comments (plans, progress updates, prior reviews, discussion), and the linked issue. This full context must inform your review.

## Step 3: Select and run review aspects

**Available review aspects:**

| Aspect | Question | When to run |
|---|---|---|
| `code` | "Does this code do what it should, correctly and idiomatically?" | Always |
| `errors` | "What can go wrong silently at runtime?" | If error handling / fallback logic touched |
| `tests` | "What behaviors are untested or poorly tested?" | If test files changed or testable code added |
| `completeness` | "Does this PR deliver everything the linked issue requires?" | If PR links to an issue |
| `simplify` | "Can this be expressed more clearly without changing behavior?" | Always (runs last, after others) |

**Targeted review:** If the user specified aspects, only review matching aspects:
- `code` → code quality review
- `errors` → error handling review
- `tests` → test coverage review
- `completeness` → completeness review
- `simplify` → simplification review

For each aspect you run:
- Read the actual changed files for full context
- Look at surrounding code, not just the diff lines
- Be specific about what you find
- Use confidence scoring (0-100, only report findings ≥ 80)

## Step 4: Post review findings

Compile all findings into a single structured comment:

```bash
gh pr comment <pr-number> --body "<!-- mach6-review -->
## Code Review

### Critical
<findings with severity: critical, if any>

### Important
<findings with severity: high, if any>

### Suggestions
<findings with severity: medium or low, if any>

### Strengths
<notable positive observations>

**Aspects reviewed:** <list of aspects>

---
*Reviewed by mach6*"
```

## Step 5: Independent assessment

Do your own independent assessment of the findings:

- Read the actual code for each finding and verify independently
- Classify each finding as:
  - **Genuine issue** — Real problem, should fix before merge. Explain why.
  - **Nitpick** — Stylistic, doesn't affect correctness. Explain why it doesn't matter.
  - **False positive** — Not actually an issue. Explain why the code is correct.
  - **Deferred** — Real issue but out of scope. Should track separately.

If a finding was already addressed in prior commits or PR discussion, classify as false positive with a note.

**Important guidance on "deferred" classifications:** Test coverage gaps should NOT be automatically deferred. If a PR adds new testable code, tests should ship with it — even if that means adding test infrastructure to a package that lacks it. Only defer tests when the gap is truly unrelated to the PR's changes (e.g., pre-existing untested code that the PR happens to touch). When tests are deferred, note whether a tracking issue exists or needs to be created.

After classifying all findings, produce an **action plan** listing what to fix, in what order.

## Step 6: Post assessment

```bash
gh pr comment <pr-number> --body "<!-- mach6-assessment -->
## Review Assessment

### Classifications

| Finding | Classification | Reasoning |
|---|---|---|
| <summary> | genuine/nitpick/false-positive/deferred | <1-2 sentences> |

### Action Plan

<numbered list of what to fix, ordered by priority>

---
*Assessment by mach6*"
```

## Step 7: CLI summary

Present to the user:
- Per-finding breakdown: summary, classification, reasoning
- Counts: genuine, nitpicks, false positives, deferred
- Action plan

If any findings were classified as **deferred**, ask the user if they want to create issues for them:
```bash
gh issue create --title "<title>" --body "<body referencing PR and finding>"
```

Suggest next step:
- If genuine issues: `/skill:mach6-implement <pr-number> <finding-numbers>`
- If all clear: `/skill:mach6-publish <pr-number>`
