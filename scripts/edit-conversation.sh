#!/usr/bin/env bash
# Dev script: edit the "text" field of a conversation row's content JSON.
#
# Usage: scripts/edit-conversation.sh <db_path> <id> <new_text>
#
# Opens the SQLite session DB at <db_path> using `tursodb`, finds the row in
# the `conversation` table with the given <id>, and replaces the value of the
# "text" key inside the first text block of the `content` JSON array with
# <new_text>. The surrounding JSON structure (block type, tool_use blocks,
# etc.) is preserved.

set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 <db_path> <id> <new_text>" >&2
    exit 1
fi

DB_PATH="$1"
ROW_ID="$2"
NEW_TEXT="$3"

if [[ ! -f "$DB_PATH" ]]; then
    echo "Error: db file not found: $DB_PATH" >&2
    exit 1
fi

if ! [[ "$ROW_ID" =~ ^[0-9]+$ ]]; then
    echo "Error: id must be an integer, got: $ROW_ID" >&2
    exit 1
fi

for cmd in tursodb jq; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "Error: required command not found: $cmd" >&2
        exit 1
    fi
done

CURRENT_CONTENT=$(tursodb -q -m list "$DB_PATH" \
    "SELECT content FROM conversation WHERE id = $ROW_ID;")

if [[ -z "$CURRENT_CONTENT" ]]; then
    echo "Error: no conversation row with id=$ROW_ID" >&2
    exit 1
fi

# Replace the "text" field of the first block whose type is "text".
NEW_CONTENT=$(printf '%s' "$CURRENT_CONTENT" | jq -c \
    --arg new "$NEW_TEXT" '
    (
        first(
            .[]
            | select(.type == "text")
        )
    ) |= (.text = $new)
    ')

if [[ -z "$NEW_CONTENT" ]]; then
    echo "Error: failed to transform content JSON" >&2
    exit 1
fi

# SQL-escape single quotes by doubling them.
ESCAPED_CONTENT=${NEW_CONTENT//\'/\'\'}

tursodb -q "$DB_PATH" \
    "UPDATE conversation SET content = '$ESCAPED_CONTENT' WHERE id = $ROW_ID;"

echo "Updated row id=$ROW_ID in $DB_PATH"
echo "New content: $NEW_CONTENT"
