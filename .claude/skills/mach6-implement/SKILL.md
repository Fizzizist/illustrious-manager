---
name: mach6-implement
description: "Implement a plan from a PR, or fix review findings / CI failures. Usage: mach6-implement 42 [finding-numbers] or mach6-implement 42 ci"
argument-hint: "<pr-number> [finding-numbers | ci]"
---

# mach6-implement — Implement Plans, Fix Findings, or Fix CI

**User input:** $ARGUMENTS

This skill has two modes:
- **Implement mode** (just a PR number): reads the plan comment and implements it
- **Fix mode** (with finding numbers or `ci`): fixes specific review findings or CI failures

## Global Rules

1. **GitHub as shared memory** — Plans, reviews, and assessments are on the PR as comments with HTML markers.
2. **No `#N` in comment bodies** — Use "finding 3", "item 3" etc. instead.
3. **Safe git** — Never use `git add -A` or `git add .`. Stage files by name. Never stage secrets.

## Step 1: Parse input

Extract:
- **PR number** (required)
- **Finding numbers** to fix (optional — e.g., `1,2,3`)
- **`ci`** flag (optional — fix CI failures instead of review findings)

If only a PR number is given → **implement mode**.
If finding numbers or `ci` → **fix mode**.

## Step 2: Checkout

```bash
gh pr checkout <pr-number>
git pull
```

---

## Implement Mode (PR number only)

### Step 3i: Read the plan and full PR context

Read ALL PR comments and the PR body to get complete context:
```bash
gh pr view <pr-number> --json title,body,comments
```

Find the plan comment (contains `<!-- mach6-plan -->` marker) from the comments. If no plan comment exists, tell the user and suggest running `/skill:mach6-plan` first.

Also read any progress updates, prior review findings, assessments, and discussion — all of this context informs implementation.

### Step 4i: Read the codebase

Read all files mentioned in the plan. Understand the existing code before making changes.

### Step 5i: Implement

Implement each deliverable directly using the available tools (read, edit, write, bash, search, grep, find, ls).

**For each deliverable in the plan**:
- Identify the specific files to modify and what changes are needed
- Read existing code for understanding
- Make the changes using edit_file, write_file, and bash tools
- Run tests and linting after making changes
- **If the plan includes tests for this deliverable, tests MUST be written as part of the implementation — not deferred**

**Test coverage is part of the deliverable, not an afterthought.** If the plan specifies tests for a deliverable, implement them directly. If the target package lacks test infrastructure, add it.

**Dependency ordering:** If deliverables are independent (don't modify the same files), you may work on them in any order. If they have dependencies, implement them sequentially — later features may depend on earlier ones.

**Small plans (1-2 simple deliverables):** These should be straightforward to implement directly.

### Step 6i: Verify

After implementing all deliverables:
- Run the project's test suite
- Run any linting/formatting tools
- Build the project if applicable
- Verify each deliverable from the plan is addressed
- If any issues remain, address the gaps

Suggest next step: `/skill:mach6-push` then `/skill:mach6-review <pr-number>` for review.

---

## Fix Mode (finding numbers or `ci`)

### Step 3f: Gather context

#### If `ci` was specified:

```bash
gh pr checks <pr-number>
gh run view <run-id> --log-failed
```

Read the failed CI logs and identify issues. Extract test failures, stack traces, error messages. If all checks pass, report this and stop.

#### If finding numbers were specified:

Read ALL PR comments to get full context:
```bash
gh pr view <pr-number> --json title,body,comments
```

Find the review (`<!-- mach6-review -->`) and assessment (`<!-- mach6-assessment -->`) comments, then extract the specific findings to fix. Prior progress comments and discussion may also provide useful context.

#### If no finding numbers and not `ci`:

Read ALL PR comments, find review/assessment comments, present genuine findings, and ask which to fix.

### Step 4f: Batch sizing

- **Simple fixes** (typos, naming, imports): ~10 per batch
- **Moderate fixes** (logic changes, refactors): ~6 per batch
- **Complex fixes** (architecture, new features): ~3 per batch

If more than batch size, fix first batch and tell user to re-run.

### Step 5f: Implement fixes

Implement fixes directly using the available tools (read, edit, write, bash, search, grep, find, ls).

**For each finding** (or batch of related findings):
- Understand the finding description and the assessment's classification/reasoning
- Identify the specific files and code locations involved
- Make the fixes using edit_file, write_file, and bash tools
- Run tests after fixing

**Simple fixes** (typos, naming, one-line changes): Fix these directly.

Defer out-of-scope items to new issues.

### Step 6f: Verify

After implementing all fixes:
- Run tests and linting
- Verify each fix addresses its finding
- If any issues remain, address the gaps

Suggest next step: `/skill:mach6-push` then `/skill:mach6-review <pr-number>` for re-review.
