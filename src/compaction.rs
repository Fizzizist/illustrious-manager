use crate::types::{ContentBlock, Message, Role};

/// Result of computing a compaction plan.
#[derive(Debug, Clone, PartialEq)]
pub struct CompactionPlan {
    /// DB row IDs of messages to deactivate (soft-delete).
    pub ids_to_deactivate: Vec<i64>,
    /// A summary message to insert in place of the deactivated messages.
    pub summary_message: Message,
    /// How many messages were kept verbatim (the recent window).
    pub messages_kept: usize,
}

/// Compute a compaction plan for the given history.
///
/// The plan identifies which messages to deactivate (soft-delete in the DB)
/// and produces a single summary message to replace them.
///
/// # Arguments
/// * `history` — Full conversation history including context prefix.
/// * `context_prefix_len` — Number of messages at the front that are the
///   non-persisted context prefix. These are never compacted.
/// * `keep_recent` — Number of messages at the tail to keep verbatim.
///
/// # Invariants
/// * Context prefix is never touched.
/// * Tool-use pairs (ToolUse + matching ToolResult) are never split across
///   the compaction boundary.
/// * If the history is too short to compact, returns `None`.
pub fn compute_compaction_plan(
    history: &[(i64, Message)],
    context_prefix_len: usize,
    keep_recent: usize,
) -> Option<CompactionPlan> {
    let persisted_len = history.len().saturating_sub(context_prefix_len);
    if persisted_len <= keep_recent {
        return None;
    }

    let max_compactable = persisted_len.saturating_sub(keep_recent);
    if max_compactable == 0 {
        return None;
    }

    // Find a safe cut point in the persisted range (after context prefix).
    // We must not split a tool-use pair: a ToolUse block in an assistant message
    // has a matching ToolResult block in the next user message.
    let cut = find_safe_cut(history, context_prefix_len, max_compactable)?;

    let ids_to_deactivate: Vec<i64> = history[context_prefix_len..cut]
        .iter()
        .map(|(id, _)| *id)
        .collect();

    if ids_to_deactivate.is_empty() {
        return None;
    }

    let messages_removed = ids_to_deactivate.len();
    let messages_kept = history.len().saturating_sub(context_prefix_len) - messages_removed;

    let summary_text = format!(
        "[CONTEXT SUMMARY] The following conversation history was compacted. \
         {} older messages were removed and replaced with this summary. \
         {} recent messages are retained verbatim below.",
        messages_removed, messages_kept,
    );

    let summary_message = Message::text(Role::User, summary_text);

    Some(CompactionPlan {
        ids_to_deactivate,
        summary_message,
        messages_kept,
    })
}

/// Find a safe index to cut the history so that no tool-use pair is split.
///
/// A tool-use pair spans two messages: an assistant message with ToolUse block(s)
/// followed by a user message with ToolResult block(s). We must not cut between
/// these two messages.
///
/// The cut is between messages `i-1` and `i`. It is unsafe if:
/// - Message `i-1` contains ToolUse blocks (the matching ToolResult is in message `i`), OR
/// - Message `i` contains ToolResult blocks (matching ToolUse in message `i-1`).
fn find_safe_cut(
    history: &[(i64, Message)],
    context_prefix_len: usize,
    max_cut: usize,
) -> Option<usize> {
    let cut_end = context_prefix_len + max_cut;

    // Walk backward from cut_end looking for a safe boundary.
    for i in (context_prefix_len + 1..=cut_end).rev() {
        if i >= history.len() {
            continue;
        }
        let prev_msg = &history[i - 1].1;
        let next_msg = if i < history.len() {
            &history[i].1
        } else {
            break;
        };

        let prev_has_tool_use = prev_msg
            .content
            .iter()
            .any(|b| matches!(b, ContentBlock::ToolUse { .. }));
        let next_has_tool_result = next_msg
            .content
            .iter()
            .any(|b| matches!(b, ContentBlock::ToolResult { .. }));

        if !prev_has_tool_use && !next_has_tool_result {
            return Some(i);
        }
    }

    // Fallback: cut at context_prefix_len (compact everything possible).
    // This only happens if the entire persisted range is one giant tool-use chain.
    if cut_end > context_prefix_len {
        Some(context_prefix_len)
    } else {
        None
    }
}

/// Truncation fallback: instead of summarising, just drop the oldest messages
/// while respecting tool-use pair integrity and the context prefix.
///
/// Returns the IDs to deactivate and the number of messages kept.
pub fn compute_truncation_plan(
    history: &[(i64, Message)],
    context_prefix_len: usize,
    keep_recent: usize,
) -> Option<(Vec<i64>, usize)> {
    let persisted_len = history.len().saturating_sub(context_prefix_len);
    if persisted_len <= keep_recent {
        return None;
    }

    let max_compactable = persisted_len.saturating_sub(keep_recent);
    if max_compactable == 0 {
        return None;
    }

    let cut = find_safe_cut(history, context_prefix_len, max_compactable)?;
    let ids_to_deactivate: Vec<i64> = history[context_prefix_len..cut]
        .iter()
        .map(|(id, _)| *id)
        .collect();

    if ids_to_deactivate.is_empty() {
        return None;
    }

    let messages_kept = persisted_len - ids_to_deactivate.len();

    Some((ids_to_deactivate, messages_kept))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_history(messages: Vec<(String, Vec<ContentBlock>)>) -> Vec<(i64, Message)> {
        messages
            .into_iter()
            .enumerate()
            .map(|(i, (role_str, content))| {
                let role = if role_str == "user" {
                    Role::User
                } else {
                    Role::Assistant
                };
                ((i + 1) as i64, Message { role, content })
            })
            .collect()
    }

    fn text_msg(role: &str, text: &str) -> (String, Vec<ContentBlock>) {
        (role.to_string(), vec![ContentBlock::Text(text.to_string())])
    }

    fn tool_use_msg(id: &str, name: &str) -> (String, Vec<ContentBlock>) {
        (
            "assistant".to_string(),
            vec![
                ContentBlock::Text("using tool".to_string()),
                ContentBlock::ToolUse {
                    id: id.to_string(),
                    name: name.to_string(),
                    input: serde_json::json!({}),
                },
            ],
        )
    }

    fn tool_result_msg(id: &str, content: &str) -> (String, Vec<ContentBlock>) {
        (
            "user".to_string(),
            vec![ContentBlock::ToolResult {
                tool_use_id: id.to_string(),
                content: content.to_string(),
                is_error: false,
            }],
        )
    }

    #[test]
    fn compact_empty_history_returns_none() {
        let history: Vec<(i64, Message)> = vec![];
        assert!(compute_compaction_plan(&history, 0, 6).is_none());
    }

    #[test]
    fn compact_below_keep_recent_returns_none() {
        let history = make_history(vec![text_msg("user", "hello"), text_msg("assistant", "hi")]);
        // keep_recent = 2 means we can't compact anything
        assert!(compute_compaction_plan(&history, 0, 2).is_none());
    }

    #[test]
    fn compact_with_context_prefix_preserves_prefix() {
        // 2 context prefix messages + 4 persisted messages
        let history = make_history(vec![
            text_msg("user", "context prefix 1"),
            text_msg("user", "context prefix 2"),
            text_msg("user", "hello"),
            text_msg("assistant", "hi"),
            text_msg("user", "how are you"),
            text_msg("assistant", "fine"),
        ]);

        let plan = compute_compaction_plan(&history, 2, 2).expect("should compact");
        // Should deactivate messages after prefix but before the keep_recent window
        // Messages 3,4 (ids 3,4) should be deactivated; 5,6 kept
        assert!(plan.ids_to_deactivate.contains(&3));
        assert!(plan.ids_to_deactivate.contains(&4));
        assert!(!plan.ids_to_deactivate.contains(&1));
        assert!(!plan.ids_to_deactivate.contains(&2));
        assert_eq!(plan.messages_kept, 2);
    }

    #[test]
    fn compact_preserves_recent_messages() {
        let history = make_history(vec![
            text_msg("user", "msg1"),
            text_msg("assistant", "msg2"),
            text_msg("user", "msg3"),
            text_msg("assistant", "msg4"),
            text_msg("user", "msg5"),
            text_msg("assistant", "msg6"),
        ]);

        let plan = compute_compaction_plan(&history, 0, 4).expect("should compact");
        // keep_recent=4 means only messages 1,2 can be compacted
        assert!(plan.ids_to_deactivate.contains(&1));
        assert!(plan.ids_to_deactivate.contains(&2));
        assert_eq!(plan.messages_kept, 4);
        assert!(!plan.ids_to_deactivate.contains(&3));
        assert!(!plan.ids_to_deactivate.contains(&4));
        assert!(!plan.ids_to_deactivate.contains(&5));
        assert!(!plan.ids_to_deactivate.contains(&6));
    }

    #[test]
    fn compact_maintains_tool_use_pairs() {
        // tool_use in msg3, tool_result in msg4 — pair must not be split
        let history = make_history(vec![
            text_msg("user", "msg1"),
            text_msg("assistant", "msg2"),
            tool_use_msg("t1", "bash"),      // id 3 — assistant
            tool_result_msg("t1", "output"), // id 4 — user
            text_msg("user", "msg5"),
            text_msg("assistant", "msg6"),
        ]);

        let plan = compute_compaction_plan(&history, 0, 2).expect("should compact");
        // The safe cut should be at index 4 (before msg5, after tool_result),
        // so ids 1,2,3,4 are deactivated and 5,6 are kept.
        // OR it could be at index 2 (before tool_use, after msg2),
        // so ids 1,2 are deactivated and 3,4,5,6 are kept.
        // Either way, 3 and 4 must be in the same group.
        let ids = &plan.ids_to_deactivate;
        let has_3 = ids.contains(&3);
        let has_4 = ids.contains(&4);
        assert_eq!(
            has_3, has_4,
            "tool-use pair (ids 3,4) must not be split: ids={ids:?}"
        );
    }

    #[test]
    fn compact_with_single_summary_message() {
        let history = make_history(vec![
            text_msg("user", "msg1"),
            text_msg("assistant", "msg2"),
            text_msg("user", "msg3"),
            text_msg("assistant", "msg4"),
        ]);

        let plan = compute_compaction_plan(&history, 0, 2).expect("should compact");
        assert!(!plan.ids_to_deactivate.is_empty());
        assert_eq!(plan.summary_message.role, Role::User);
        match &plan.summary_message.content[0] {
            ContentBlock::Text(t) => {
                assert!(t.starts_with("[CONTEXT SUMMARY]"));
            }
            _ => panic!("expected text block"),
        }
    }

    #[test]
    fn compact_configurable_keep_recent() {
        let history = make_history(vec![
            text_msg("user", "msg1"),
            text_msg("assistant", "msg2"),
            text_msg("user", "msg3"),
            text_msg("assistant", "msg4"),
            text_msg("user", "msg5"),
            text_msg("assistant", "msg6"),
        ]);

        let plan = compute_compaction_plan(&history, 0, 2).expect("should compact");
        assert_eq!(plan.ids_to_deactivate.len(), 4);
        assert_eq!(plan.messages_kept, 2);

        let plan2 = compute_compaction_plan(&history, 0, 4).expect("should compact");
        assert_eq!(plan2.ids_to_deactivate.len(), 2);
        assert_eq!(plan2.messages_kept, 4);
    }

    #[test]
    fn compact_context_prefix_only_returns_none() {
        let history = make_history(vec![
            text_msg("user", "prefix1"),
            text_msg("user", "prefix2"),
        ]);
        // All messages are prefix — nothing to compact
        assert!(compute_compaction_plan(&history, 2, 0).is_none());
    }

    #[test]
    fn truncation_fallback_basic() {
        let history = make_history(vec![
            text_msg("user", "msg1"),
            text_msg("assistant", "msg2"),
            text_msg("user", "msg3"),
            text_msg("assistant", "msg4"),
        ]);

        let (ids, kept) = compute_truncation_plan(&history, 0, 2).expect("should truncate");
        assert!(ids.contains(&1));
        assert!(ids.contains(&2));
        assert_eq!(kept, 2);
    }

    #[test]
    fn truncation_respects_tool_pairs() {
        let history = make_history(vec![
            text_msg("user", "msg1"),
            text_msg("assistant", "msg2"),
            tool_use_msg("t1", "bash"),
            tool_result_msg("t1", "output"),
            text_msg("user", "msg5"),
            text_msg("assistant", "msg6"),
        ]);

        let (ids, _) = compute_truncation_plan(&history, 0, 2).expect("should truncate");
        let has_3 = ids.contains(&3);
        let has_4 = ids.contains(&4);
        assert_eq!(
            has_3, has_4,
            "tool-use pair must not be split in truncation"
        );
    }

    #[test]
    fn compact_with_consecutive_tool_pairs() {
        // Two consecutive tool-use pairs back-to-back
        let history = make_history(vec![
            text_msg("user", "msg1"),
            tool_use_msg("t1", "bash"),
            tool_result_msg("t1", "out1"),
            tool_use_msg("t2", "bash"),
            tool_result_msg("t2", "out2"),
            text_msg("user", "msg6"),
        ]);

        let plan = compute_compaction_plan(&history, 0, 2).expect("should compact");
        // Each pair must be internally consistent (both deactivated or both kept).
        // The two consecutive pairs may be in different groups — the cut may
        // fall between the result of one pair and the use of the next.
        let ids = &plan.ids_to_deactivate;
        let has_2 = ids.contains(&2);
        let has_3 = ids.contains(&3);
        let has_4 = ids.contains(&4);
        let has_5 = ids.contains(&5);
        assert_eq!(has_2, has_3, "pair t1 must be together");
        assert_eq!(has_4, has_5, "pair t2 must be together");
    }

    #[test]
    fn compact_keeps_at_least_keep_recent_even_with_tool_pairs() {
        // Edge case: the keep_recent window starts in the middle of a tool pair.
        // The cut point should move backward to before the pair.
        let history = make_history(vec![
            text_msg("user", "msg1"),
            text_msg("assistant", "msg2"),
            tool_use_msg("t1", "bash"),
            tool_result_msg("t1", "output"),
            text_msg("user", "msg5"),
            text_msg("assistant", "msg6"),
        ]);

        // keep_recent=3 would want to keep messages 4,5,6 but msg4 is a tool result
        // paired with msg3. The cut should move to before msg3.
        let plan = compute_compaction_plan(&history, 0, 3).expect("should compact");
        assert!(!plan.ids_to_deactivate.contains(&3));
        assert!(!plan.ids_to_deactivate.contains(&4));
        // At minimum messages 1,2 should be compacted
        assert!(plan.ids_to_deactivate.contains(&1));
        assert!(plan.ids_to_deactivate.contains(&2));
    }
}
