#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryType {
    Identifier,
    NaturalLanguage,
    PathLike,
}

fn has_camel_case(s: &str) -> bool {
    let bytes = s.as_bytes();
    for i in 1..bytes.len() {
        if bytes[i].is_ascii_uppercase() && bytes[i - 1].is_ascii_lowercase() {
            return true;
        }
    }
    false
}

fn has_snake_case(s: &str) -> bool {
    s.contains('_') && s.chars().any(|c| c.is_alphanumeric())
}

fn is_screaming_snake(s: &str) -> bool {
    if !s.contains('_') {
        return false;
    }
    s.chars()
        .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
}

pub fn classify_query(query: &str) -> QueryType {
    let trimmed = query.trim();
    if trimmed.is_empty() {
        return QueryType::NaturalLanguage;
    }

    if trimmed.contains('/') || trimmed.contains('\\') {
        return QueryType::PathLike;
    }

    if has_dotted_path(trimmed) {
        return QueryType::PathLike;
    }

    let words: Vec<&str> = trimmed.split_whitespace().collect();
    if words.len() == 1 && has_file_extension(trimmed) {
        return QueryType::PathLike;
    }

    if words.len() == 1 {
        return QueryType::Identifier;
    }

    if words.len() <= 3 {
        let looks_like_code = words
            .iter()
            .any(|w| has_camel_case(w) || has_snake_case(w) || is_screaming_snake(w));
        if looks_like_code {
            return QueryType::Identifier;
        }
    }

    QueryType::NaturalLanguage
}

fn has_dotted_path(s: &str) -> bool {
    let parts: Vec<&str> = s.split('.').collect();
    parts.len() >= 3
        && parts
            .iter()
            .all(|p| !p.is_empty() && p.chars().all(|c| c.is_alphanumeric() || c == '_'))
}

fn has_file_extension(s: &str) -> bool {
    if !s.contains('.') {
        return false;
    }
    let ext = s.rsplit('.').next().expect("should have part after dot");
    !ext.is_empty() && ext.len() <= 5 && ext.chars().all(|c| c.is_alphanumeric())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_query_classified_as_natural_language() {
        assert_eq!(classify_query(""), QueryType::NaturalLanguage);
    }

    #[test]
    fn whitespace_only_query_classified_as_natural_language() {
        assert_eq!(classify_query("   "), QueryType::NaturalLanguage);
    }

    #[test]
    fn single_camel_case_word_is_identifier() {
        assert_eq!(classify_query("myFunction"), QueryType::Identifier);
    }

    #[test]
    fn single_snake_case_word_is_identifier() {
        assert_eq!(classify_query("my_function"), QueryType::Identifier);
    }

    #[test]
    fn single_lowercase_word_is_identifier() {
        assert_eq!(classify_query("search"), QueryType::Identifier);
    }

    #[test]
    fn single_uppercase_word_is_identifier() {
        assert_eq!(classify_query("Search"), QueryType::Identifier);
    }

    #[test]
    fn path_with_slash_is_path_like() {
        assert_eq!(classify_query("src/tools/mod.rs"), QueryType::PathLike);
    }

    #[test]
    fn path_with_backslash_is_path_like() {
        assert_eq!(classify_query("src\\tools\\mod.rs"), QueryType::PathLike);
    }

    #[test]
    fn dotted_path_three_segments_is_path_like() {
        assert_eq!(classify_query("foo.bar.baz"), QueryType::PathLike);
    }

    #[test]
    fn filename_with_extension_is_path_like() {
        assert_eq!(classify_query("config.yaml"), QueryType::PathLike);
    }

    #[test]
    fn screaming_snake_case_is_identifier() {
        assert_eq!(classify_query("MAX_SIZE DEFAULT"), QueryType::Identifier);
    }

    #[test]
    fn short_camel_case_phrase_is_identifier() {
        assert_eq!(
            classify_query("myFunction handles input"),
            QueryType::Identifier
        );
    }

    #[test]
    fn natural_language_sentence_is_natural_language() {
        assert_eq!(
            classify_query("find all places where database connections are established"),
            QueryType::NaturalLanguage
        );
    }

    #[test]
    fn short_mixed_words_without_code_patterns_is_natural_language() {
        assert_eq!(
            classify_query("how does this work"),
            QueryType::NaturalLanguage
        );
    }

    #[test]
    fn four_words_without_code_patterns_is_natural_language() {
        assert_eq!(
            classify_query("find the error handler function"),
            QueryType::NaturalLanguage
        );
    }
}
