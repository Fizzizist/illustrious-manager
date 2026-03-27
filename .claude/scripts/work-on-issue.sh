#!/usr/bin/env bash
set -euo pipefail

ISSUE_NUMBER="${1:?Usage: work-on-issue.sh <issue-number>}"
BASE_BRANCH="${MOE_BRANCH:-trunk}"
LOCK_DIR="/tmp/illustrious-manager-code-review.lock"
LOG_FILE="/tmp/work-on-issue-${ISSUE_NUMBER}.log"

# Generate a deterministic session ID upfront so we don't need to parse it from output
SESSION_ID=$(uuidgen | tr '[:upper:]' '[:lower:]')
REPO_ROOT="$(git rev-parse --show-toplevel)"

# State variables (populated during execution)
PR_NUMBER=""
WORKTREE_PATH=""
HOLDS_LOCK=false
HUMAN_CHOICE=""
SPINNER_PID=""

# --- Spinner ---

start_spinner() {
  local msg="${1:-Claude is thinking}"
  (
    local chars='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
    local i=0
    while true; do
      printf "\r  %s %s... " "${chars:i%${#chars}:1}" "$msg" >&2
      i=$((i + 1))
      sleep 0.1
    done
  ) &
  SPINNER_PID=$!
}

stop_spinner() {
  if [ -n "$SPINNER_PID" ] && kill -0 "$SPINNER_PID" 2>/dev/null; then
    kill "$SPINNER_PID" 2>/dev/null
    wait "$SPINNER_PID" 2>/dev/null || true
    printf "\r\033[K" >&2
    SPINNER_PID=""
  fi
}

# --- Cleanup ---

cleanup() {
  stop_spinner
  echo "=== Cleanup ==="
  if [ "$HOLDS_LOCK" = true ]; then
    echo "Releasing code review lock..."
    rm -f "${LOCK_DIR}/pid"
    rmdir "$LOCK_DIR" 2>/dev/null || true
    HOLDS_LOCK=false
  fi
  if [ -n "$WORKTREE_PATH" ] && [ -d "$WORKTREE_PATH" ]; then
    echo "Removing worktree at ${WORKTREE_PATH}..."
    git -C "$REPO_ROOT" worktree remove "$WORKTREE_PATH" --force 2>/dev/null || true
    git -C "$REPO_ROOT" branch -D "agent/issue-${ISSUE_NUMBER}" 2>/dev/null || true
  fi
  echo "=== Finished: $(date) ==="
}

trap cleanup EXIT INT TERM

# --- Lock Helpers ---

acquire_lock() {
  echo "Acquiring code review lock..."
  while ! mkdir "$LOCK_DIR" 2>/dev/null; do
    # Check for stale lock
    if [ -f "${LOCK_DIR}/pid" ]; then
      local lock_pid
      lock_pid=$(cat "${LOCK_DIR}/pid")
      if ! kill -0 "$lock_pid" 2>/dev/null; then
        echo "Removing stale lock from dead process ${lock_pid}..."
        rm -f "${LOCK_DIR}/pid"
        rmdir "$LOCK_DIR" 2>/dev/null || true
        continue
      fi
    fi
    echo "Lock held by another process, waiting 10s..."
    sleep 10
  done
  echo $$ > "${LOCK_DIR}/pid"
  HOLDS_LOCK=true
  echo "Lock acquired."
}

release_lock() {
  if [ "$HOLDS_LOCK" = true ]; then
    echo "Releasing code review lock..."
    rm -f "${LOCK_DIR}/pid"
    rmdir "$LOCK_DIR" 2>/dev/null || true
    HOLDS_LOCK=false
    echo "Lock released."
  fi
}

# --- Setup ---

if [ ! -f ".claude/.env" ]; then
  echo "ERROR: .claude/.env file not found in $(pwd). Please make a copy of `.claude/.env.example` and configure it with your own tokens before running .claude/scripts/work-on-issue.sh." >&2
  exit 1
fi
source .claude/.env

# Logging: tee all output to log file
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== work-on-issue: Issue #${ISSUE_NUMBER} ==="
echo "=== Started: $(date) ==="

# Validate prerequisites
command -v claude >/dev/null 2>&1 || { echo "ERROR: claude CLI not found"; exit 1; }
command -v gh >/dev/null 2>&1 || { echo "ERROR: gh CLI not found"; exit 1; }
command -v jq >/dev/null 2>&1 || { echo "ERROR: jq not found (required for parsing JSON output)"; exit 1; }
command -v uuidgen >/dev/null 2>&1 || { echo "ERROR: uuidgen not found (required for generating session IDs)"; exit 1; }
gh issue view "$ISSUE_NUMBER" --json number >/dev/null 2>&1 || { echo "ERROR: Issue #${ISSUE_NUMBER} not found"; exit 1; }

echo "Prerequisites validated."

# --- Phase Functions ---

phase_triage() {
  echo "=== Phase 1: Triage ==="
  local issue_body
  issue_body=$(gh issue view "$ISSUE_NUMBER" --json body,title --jq '"Title: \(.title)\n\nBody:\n\(.body)"')

  local triage_result
  start_spinner "Triaging issue #${ISSUE_NUMBER}"
  triage_result=$(claude -p --dangerously-skip-permissions --model opus \
    --output-format json \
    --json-schema '{"type":"object","properties":{"requires_code_changes":{"type":"boolean"},"reasoning":{"type":"string"}},"required":["requires_code_changes","reasoning"]}' \
    "You are triaging a GitHub issue. Determine whether this issue requires code changes to the repository, or if it is a non-code task (e.g., a manual task to perform, a question to answer, an issue to close, etc.).

Issue #${ISSUE_NUMBER}:
${issue_body}

Respond with whether code changes are required and your reasoning." 2>/dev/null) || true
  stop_spinner

  if [ -z "$triage_result" ]; then
    echo "ERROR: Claude returned empty output during triage"
    exit 1
  fi

  local requires_code
  requires_code=$(echo "$triage_result" | jq -r '(.structured_output // .result // .) | if type == "string" then fromjson else . end | .requires_code_changes' 2>/dev/null)

  if [ -z "$requires_code" ] || [ "$requires_code" = "null" ]; then
    echo "ERROR: Failed to parse triage result from Claude output"
    exit 1
  fi

  echo "Triage result: requires_code_changes=${requires_code}"
  echo "Reasoning: $(echo "$triage_result" | jq -r '(.structured_output // .result // .) | if type == "string" then fromjson else . end | .reasoning' 2>/dev/null)"

  if [ "$requires_code" != "true" ]; then
    echo "No code changes needed. Attempting to perform the task..."
    start_spinner "Handling non-code task"
    claude -p --dangerously-skip-permissions --model opus \
      "This GitHub issue does not require code changes. Perform whatever non-code task is appropriate (close the issue, post a comment, update docs, etc.). If you cannot perform the task, explain why.

Issue #${ISSUE_NUMBER}:
${issue_body}" || true
    stop_spinner

    echo "Non-code task complete. Exiting."
    exit 0
  fi

  echo "Code changes required. Proceeding to implementation."
}

phase_implement() {
  echo "=== Phase 2: Prepare & Implement ==="

  local branch_name="agent/issue-${ISSUE_NUMBER}"

  echo "Checking out ${BASE_BRANCH} and pulling latest..."
  git checkout "$BASE_BRANCH"
  git pull

  # Create worktree outside the repo to avoid polluting the working tree
  WORKTREE_PATH="${HOME}/.config/illustrious-manager/worktrees/issue-${ISSUE_NUMBER}"
  mkdir -p "$(dirname "$WORKTREE_PATH")"
  echo "Creating worktree at ${WORKTREE_PATH} on branch ${branch_name}..."
  git worktree add -b "$branch_name" "$WORKTREE_PATH" "$BASE_BRANCH"

  local issue_body
  issue_body=$(gh issue view "$ISSUE_NUMBER" --json body,title --jq '"Title: \(.title)\n\nBody:\n\(.body)"')

  echo "Launching Claude in worktree (session: ${SESSION_ID})..."
  start_spinner "Implementing issue #${ISSUE_NUMBER}"
  cd "$WORKTREE_PATH"

  local impl_output
  impl_output=$(claude -p --dangerously-skip-permissions --model sonnet \
    --session-id "$SESSION_ID" \
    "You are working on GitHub issue #${ISSUE_NUMBER}. Here is the issue:

${issue_body}

Instructions:
1. Use the /test-driven-development skill to guide your implementation approach.
2. Implement the required changes to resolve this issue.
3. Commit your changes with clear commit messages.
4. Push your branch and open a pull request against the '${BASE_BRANCH}' branch.
5. In the PR description, reference the issue with 'Closes #${ISSUE_NUMBER}'.

IMPORTANT: If you have questions or need clarification before proceeding, state your questions clearly and then STOP. Do not attempt to create a PR until you have the answers you need." 2>&1) || true
  stop_spinner

  echo "$impl_output"

  # Q&A loop: if Claude asked questions instead of creating a PR, let the user respond
  local max_qa_rounds=5
  local qa_round=0
  while [ "$qa_round" -lt "$max_qa_rounds" ]; do
    # Check if a PR was created
    PR_NUMBER=$(gh pr list --head "$branch_name" --json number --jq '.[0].number' 2>/dev/null || true)
    if [ -n "$PR_NUMBER" ]; then
      break
    fi

    # No PR yet — Claude likely has questions. Prompt the user.
    echo "" >&2
    echo "=====================================" >&2
    echo "  Claude has not yet created a PR." >&2
    echo "  It may have questions (see output above)." >&2
    echo "=====================================" >&2
    echo "  Enter your response (or 'abort' to stop):" >&2

    local user_response
    read -rp "> " user_response </dev/tty

    if [ "$user_response" = "abort" ]; then
      echo "Aborted by user."
      exit 1
    fi

    qa_round=$((qa_round + 1))
    echo "Resuming session with your response (Q&A round ${qa_round})..."
    start_spinner "Continuing implementation"
    impl_output=$(claude -p --dangerously-skip-permissions --model sonnet \
      --resume "$SESSION_ID" \
      "${user_response}

Continue implementing the issue. When done, commit, push, and open a PR against '${BASE_BRANCH}' with 'Closes #${ISSUE_NUMBER}' in the description. If you still have questions, state them clearly and STOP." 2>&1) || true
    stop_spinner

    echo "$impl_output"
  done

  # Final PR check with retries for API lag
  if [ -z "$PR_NUMBER" ]; then
    local retries=0
    while [ -z "$PR_NUMBER" ] && [ "$retries" -lt 5 ]; do
      PR_NUMBER=$(gh pr list --head "$branch_name" --json number --jq '.[0].number')
      if [ -z "$PR_NUMBER" ]; then
        retries=$((retries + 1))
        echo "PR not found yet, retrying in 5s... (attempt ${retries}/5)"
        sleep 5
      fi
    done
  fi

  if [ -z "$PR_NUMBER" ]; then
    echo "ERROR: No PR found for branch ${branch_name} after implementation"
    exit 1
  fi
  echo "PR #${PR_NUMBER} created."
}

phase_code_review() {
  echo "=== Phase 3: Code Review ==="

  acquire_lock

  echo "Launching code review for PR #${PR_NUMBER}..."
  start_spinner "Reviewing PR #${PR_NUMBER}"
  claude -p --dangerously-skip-permissions --model opus \
    "Review pull request #${PR_NUMBER} in this repository.

Use the /code-review skill to perform the review. Leave your feedback as review comments on the PR using the gh CLI." || true
  stop_spinner

  release_lock
}

phase_fix() {
  echo "=== Phase 4: Fix Based on Review ==="

  echo "Resuming session to address review comments..."
  start_spinner "Fixing review comments on PR #${PR_NUMBER}"
  claude -p --dangerously-skip-permissions --model sonnet \
    --resume "$SESSION_ID" \
    "Look at the review comments on PR #${PR_NUMBER}. Address all feedback, push your fixes, and leave a reply on each comment explaining what you changed."
  stop_spinner
}

phase_ci_loop() {
  echo "=== Phase 5: CI Pipeline Loop ==="
  local max_iterations="${CI_MAX_ITERATIONS:-5}"
  local poll_interval="${CI_POLL_INTERVAL:-30}"
  local max_wait_cycles="${CI_MAX_WAIT_CYCLES:-60}"  # max_wait_cycles * poll_interval = max wait time
  local iteration=0

  while [ "$iteration" -lt "$max_iterations" ]; do
    # Wait for all in-progress checks to settle
    echo "Waiting for CI checks to complete..."
    local wait_count=0
    while [ "$wait_count" -lt "$max_wait_cycles" ]; do
      local pending
      pending=$(gh pr checks "$PR_NUMBER" --json state \
        --jq '[.[] | select(.state != "completed")] | length' 2>/dev/null || echo "0")
      if [ "$pending" = "0" ]; then
        break
      fi
      echo "  ${pending} check(s) still running, waiting ${poll_interval}s..."
      sleep "$poll_interval"
      wait_count=$((wait_count + 1))
    done

    if [ "$wait_count" -ge "$max_wait_cycles" ]; then
      echo "WARNING: Timed out waiting for CI checks after $((max_wait_cycles * poll_interval))s. Proceeding anyway."
      return
    fi

    # Count failed checks
    local failed
    failed=$(gh pr checks "$PR_NUMBER" --json state,conclusion \
      --jq '[.[] | select(.state == "completed" and (.conclusion == "failure" or .conclusion == "timed_out" or .conclusion == "action_required"))] | length' \
      2>/dev/null || echo "0")

    if [ "$failed" = "0" ]; then
      echo "All CI checks passed."
      return
    fi

    iteration=$((iteration + 1))
    echo "${failed} CI check(s) failed. Asking Claude to fix (iteration ${iteration}/${max_iterations})..."
    start_spinner "Fixing CI failures (iteration ${iteration})"
    claude -p --dangerously-skip-permissions --model sonnet \
      --resume "$SESSION_ID" \
      "The CI pipeline for PR #${PR_NUMBER} has failing checks.

1. Run 'gh pr checks ${PR_NUMBER}' to identify which checks failed.
2. Investigate the failure logs (use 'gh run view <run-id> --log-failed' as needed).
3. Fix the underlying issues.
4. Commit your changes with a clear message explaining what you fixed.
5. Push the fixes.

Do not open a new PR — push to the existing branch." || true
    stop_spinner
  done

  echo "WARNING: CI did not fully pass after ${max_iterations} fix iteration(s). Proceeding to human review."
}

phase_abandon() {
  echo "=== Phase 6: Abandon & Rewrite Issue ==="

  start_spinner "Abandoning and documenting learnings"
  claude -p --dangerously-skip-permissions --model opus \
    --resume "$SESSION_ID" \
    "The implementation attempt for issue #${ISSUE_NUMBER} is being abandoned.

1. Close PR #${PR_NUMBER} using the gh CLI.
2. Update GitHub issue #${ISSUE_NUMBER} by appending a '## Learnings' section to the existing body. Include: what was tried, what worked, what didn't, and what the next approach should be. Do not remove the original issue text. Use the gh CLI to update the issue body."
  stop_spinner
}

prompt_human() {
  local round="$1"
  local options
  if [ "$round" -eq 1 ]; then
    options="[2] Another round of fixes"
  else
    options="[2] Abandon - close PR and rewrite the issue"
  fi

  echo "" >&2
  echo "=====================================" >&2
  echo "  Round ${round} complete." >&2
  echo "  PR: $(gh pr view "$PR_NUMBER" --json url --jq '.url')" >&2
  echo "=====================================" >&2
  echo "  [1] Cleanup (accept the PR as-is)" >&2
  echo "  ${options}" >&2
  echo "=====================================" >&2

  # Ensure we have a TTY before prompting the user.
  if [ ! -t 0 ] && [ ! -t 1 ] && [ ! -t 2 ]; then
    echo "Error: No TTY available. This script must be run interactively to choose how to proceed with the PR." >&2
    exit 1
  fi
  while true; do
    read -rp "Choose [1/2]: " HUMAN_CHOICE </dev/tty
    case "$HUMAN_CHOICE" in
      1|2) break ;;
      *) echo "Invalid choice. Enter 1 or 2." >&2 ;;
    esac
  done
}

phase_human_loop() {
  # Round 1
  prompt_human 1

  if [ "$HUMAN_CHOICE" = "1" ]; then
    echo "Accepted. Cleaning up."
    return
  fi

  # Second round of fixes
  echo "=== Round 2: Fixing again ==="
  start_spinner "Fixing review comments (round 2)"
  claude -p --dangerously-skip-permissions --model sonnet \
    --resume "$SESSION_ID" \
    "Look at the latest review comments on PR #${PR_NUMBER}. Address all remaining feedback, push your fixes, and leave a reply on each comment explaining what you changed."
  stop_spinner

  prompt_human 2

  if [ "$HUMAN_CHOICE" = "1" ]; then
    echo "Accepted. Cleaning up."
    return
  fi

  # Abandon
  phase_abandon
}

# --- Main Flow ---
phase_triage
phase_implement
phase_code_review
phase_fix
phase_ci_loop
phase_human_loop

echo "=== Done ==="
