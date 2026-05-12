use crate::types::{ContentBlock, Message, Role};

/// Count the number of turns in a message slice.
///
/// A "turn" starts with a user message and includes all immediately following
/// assistant messages. Orphan assistant messages at the start form their own turn.
pub fn count_turns(messages: &[Message]) -> usize {
    if messages.is_empty() {
        return 0;
    }
    let mut turns = 0;
    let mut i = 0;
    while i < messages.len() {
        turns += 1;
        i += 1;
        while i < messages.len() && messages[i].role == Role::Assistant {
            i += 1;
        }
    }
    turns
}

/// Find the start index of the last `n` turns in the message slice.
///
/// Returns `0` if there are fewer than `n` turns (i.e., all messages are retained)
/// or if `n` is 0.
pub fn retained_start_index(messages: &[Message], n: usize) -> usize {
    if n == 0 || messages.is_empty() {
        return 0;
    }
    let total = count_turns(messages);
    let skip = total.saturating_sub(n);
    if skip == 0 {
        return 0;
    }
    let mut turn_start = 0;
    let mut turns_seen = 0;
    let mut i = 0;
    while i < messages.len() && turns_seen < skip {
        turn_start = i + 1;
        turns_seen += 1;
        i += 1;
        while i < messages.len() && messages[i].role == Role::Assistant {
            turn_start = i + 1;
            i += 1;
        }
    }
    turn_start
}

/// Strip non-text blocks from a message, returning `None` if the result is empty.
///
/// Tool-use, tool-result, thinking, and redacted-thinking blocks are removed.
/// Only `Text` blocks are preserved. Messages that become empty after stripping
/// are filtered out entirely.
pub fn strip_non_text(msg: &Message) -> Option<Message> {
    let stripped: Vec<ContentBlock> = msg
        .content
        .iter()
        .filter(|b| matches!(b, ContentBlock::Text(_)))
        .cloned()
        .collect();
    if stripped.is_empty() {
        None
    } else {
        Some(Message {
            role: msg.role.clone(),
            content: stripped,
        })
    }
}

/// Extract the last `n` turns from the history, returning the messages
/// that belong to those turns with non-text blocks stripped.
pub fn retained_turns(messages: &[Message], n: usize) -> Vec<Message> {
    let start = retained_start_index(messages, n);
    if start >= messages.len() {
        return Vec::new();
    }
    messages[start..]
        .iter()
        .filter_map(strip_non_text)
        .collect()
}
