use crate::types::{ContentBlock, Message, Role};

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
