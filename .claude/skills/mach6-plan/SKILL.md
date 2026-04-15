---
name: mach6-plan
description: "Explore codebase, create implementation plan, create feature branch with dummy commit, open draft PR, post plan as PR comment. Everything lives on the PR from this point forward. Usage: mach6-plan 42"
argument-hint: "<issue-number>"
---

# mach6-plan — Plan, Branch, and Open PR

**User input:** $ARGUMENTS

This command is strictly for **planning**. Do NOT implement any code changes — no file edits, no file writes.

## Global Rules

1. **GitHub as shared memory** — Plans, reviews, assessments, and progress are posted as PR/issue comments so any future session can pick up context.
2. **HTML markers** — Use `<!-- mach6-plan -->` as the first line of plan comment bodies for reliable discovery.
3. **No `#N` in comment bodies** — GitHub auto-links `#N` to issues/PRs. Use "finding 3", "item 3", "stage 2" etc. instead.
4. **Safe git** — Never use `git add -A` or `git add .`. Stage files by name. Never stage secrets.
5. **Project conventions** — Check for CLAUDE.md, AGENTS.md, .dreb/CONTEXT.md, and CONTRIBUTING.md before planning.

## Step 1: Read the issue

```bash
gh issue view <number>
gh issue view <number> --comments
```

Parse everything: problem statement, constraints, requirements, acceptance criteria, prior discussion, any existing assessment comments (look for `<!-- mach6-assessment -->`).

## Step 2: Read project conventions

Check for and read (first found):
- CONTRIBUTING.md, DEVELOPMENT.md, .github/CONTRIBUTING.md
- CLAUDE.md, AGENTS.md, .dreb/CONTEXT.md

Extract planning-relevant guidance: project layers, testing expectations, coding conventions.

## Step 3: Explore the codebase

Explore the codebase directly using the available tools (search, grep, find, read, ls) to understand:
- **Similar features**: Find existing code that solves related problems, trace implementation patterns
- **Architecture**: Map relevant architecture layers, abstractions, data flow
- **Integration points**: Identify where new code connects to existing systems

Include project conventions in your exploration. Identify 5-10 key files and read them thoroughly.

## Step 4: Draft the plan

Create an implementation plan with:
- Clear analysis of the problem
- **Deliverables**: What will be produced (be specific)
- **Acceptance criteria**: How to verify the work is done
- **Files to create or modify**: List each with what changes
- **Testing approach**: What tests to write, what to verify
- **Risks and open questions**: Anything that might derail implementation

The plan should be **high-level on implementation details** (avoid cascading spec errors from over-specifying) but **specific on deliverables and acceptance criteria**.

**Project-layer coverage:** Cross-check the plan against discovered project layers. Every affected layer should be addressed.

**Test coverage is mandatory, not optional.** Every new behavior, command handler, formatting function, or event wiring must include tests in the plan. If the target package lacks test infrastructure, the plan must include setting it up as a deliverable — this cannot be deferred. The testing approach should specify:
- Which test files to create or modify
- What behaviors to verify (happy paths, error paths, edge cases)
- What test infrastructure/helpers are needed (mocks, factories, fixtures)

Present the plan to the user. Discuss and revise if they have feedback.

## Step 5: Create branch and draft PR

```bash
# Derive branch name from issue
# Format: feature/issue-<N>-<slug> (slug = 3-5 words from title, lowercase, hyphens)
git checkout -b feature/issue-<N>-<slug>

# Create an empty commit so the PR can be opened
git commit --allow-empty -m "chore: open PR for issue <N>"

git push -u origin feature/issue-<N>-<slug>

# Open draft PR
gh pr create --draft --title "<title>" --body "Closes #<N>

<brief description>

Implementation plan posted as a comment below."
```

## Step 6: Post plan to PR

```bash
gh pr comment <pr-number> --body "<!-- mach6-plan -->
## Implementation Plan

<full plan content>

---
*Plan created by mach6*"
```

Suggest next step: implement the plan, then `/skill:mach6-push` when ready.
