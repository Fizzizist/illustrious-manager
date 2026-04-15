---
name: mach6-issue
description: "Assess an existing GitHub issue (explore codebase, identify scope/risks/ambiguities, post assessment) or create a new structured issue. Usage: mach6-issue 42 (assess) or mach6-issue (create) or mach6-issue <description> (create with context)"
argument-hint: "[issue-number | description]"
---

# mach6-issue — Assess or Create Issue

**User input:** $ARGUMENTS

## Global Rules

1. **GitHub as shared memory** — Plans, reviews, assessments, and progress are posted as PR/issue comments so any future session can pick up context.
2. **HTML markers** — Use `<!-- mach6-assessment -->`, `<!-- mach6-plan -->`, `<!-- mach6-review -->`, `<!-- mach6-progress -->` as the first line of comment bodies for reliable discovery.
3. **No `#N` in comment bodies** — GitHub auto-links `#N` to issues/PRs. Use "finding 3", "item 3", "stage 2" etc. instead.
4. **Safe git** — Never use `git add -A` or `git add .`. Stage files by name. Never stage secrets (.env, credentials, tokens, keys).
5. **Project conventions** — Check for CLAUDE.md, AGENTS.md, .dreb/CONTEXT.md, and CONTRIBUTING.md before planning or implementing.

## Determine Mode

If the input is a number, run **ASSESS** mode. Otherwise, run **CREATE** mode.

---

## ASSESS Mode

**Assess an existing GitHub issue — explore the codebase, identify scope/risks/ambiguities, post assessment.**

### Step 1: Read the issue

```bash
gh issue view <number>
gh issue view <number> --comments
```

Parse: problem statement, constraints, requirements, acceptance criteria, prior discussion, linked PRs.

### Step 2: Explore the codebase

Explore the codebase directly using the available tools (search, grep, find, read, ls) to understand:
- **Relevant code**: Find existing code related to the issue, trace implementation patterns
- **Architecture**: Map relevant architecture layers, abstractions, data flow
- **Prior work**: Check for related branches, PRs, or commits

Identify 5-10 key files and read them thoroughly.

### Step 3: Assess

Present to the user:
1. **Summary**: The issue in your own words
2. **Current state**: What exists today that's relevant
3. **Gaps**: What's missing, broken, or unclear
4. **Ambiguities**: Underspecified aspects or open questions
5. **Scope**: Size and complexity estimate
6. **Risks**: Pitfalls, edge cases, architectural concerns

### Step 4: Post assessment

Post as an issue comment:

```bash
gh issue comment <number> --body "<!-- mach6-assessment -->
## Issue Assessment

<assessment content>

---
*Automated assessment by mach6*"
```

Suggest next step: `/skill:mach6-plan <number>`

---

## CREATE Mode

**Create a new structured GitHub issue from context or description.**

### Step 1: Gather context

If a description was provided, use it as the starting point. Otherwise, ask the user what they want to create an issue for.

Check if the repository has issue templates:
```bash
ls .github/ISSUE_TEMPLATE/ 2>/dev/null
```
If templates exist, read them and select the most appropriate one.

Explore the codebase if needed to understand the relevant area.

### Step 2: Draft the issue

Create a structured issue with:
- **Title**: Clear, concise, action-oriented (under 80 chars, imperative form)
- **Summary**: 2-3 sentences describing the problem or feature
- **Current Behavior** (for bugs/improvements): What happens now
- **Proposed Behavior**: What should happen
- **Acceptance Criteria**: Bullet list of verifiable conditions that define "done"
- **Context**: Links to related PRs, issues, or discussions
- **Technical Notes**: Implementation hints, relevant files, architectural considerations
- **Labels**: Suggest appropriate labels based on the issue type

Present the draft to the user for approval.

### Step 3: Create the issue

```bash
gh issue create --title "<title>" --body "<body>" [--label "<labels>"]
```

Report the issue number and URL. Suggest next step: `/skill:mach6-plan <number>`
