pub fn tokenize(text: &str) -> Vec<String> {
    // Split on common delimiters after camelCase expansion
    let camel_expanded = expand_camel_case(text);

    let parts: Vec<&str> = camel_expanded
        .split(|c: char| {
            c.is_whitespace() || c == '/' || c == '\\' || c == '.' || c == '-' || c == '_'
        })
        .collect();

    let mut seen = std::collections::HashSet::new();
    let mut tokens = Vec::new();
    for part in parts {
        let lower = part.to_lowercase();
        if lower.len() >= 2 && seen.insert(lower.clone()) {
            tokens.push(lower);
        }
    }
    tokens
}

fn expand_camel_case(text: &str) -> String {
    let mut result = String::with_capacity(text.len() * 2);
    let mut prev_lower = false;
    for c in text.chars() {
        if c.is_ascii_uppercase() && prev_lower {
            result.push(' ');
        }
        prev_lower = c.is_ascii_lowercase();
        result.push(c);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_snake_case() {
        let tokens = tokenize("my_function_name");
        assert!(tokens.contains(&"my".to_string()));
        assert!(tokens.contains(&"function".to_string()));
        assert!(tokens.contains(&"name".to_string()));
    }

    #[test]
    fn splits_camel_case() {
        let tokens = tokenize("myFunctionName");
        assert!(tokens.contains(&"my".to_string()));
        assert!(tokens.contains(&"function".to_string()));
        assert!(tokens.contains(&"name".to_string()));
    }

    #[test]
    fn splits_path_separators() {
        let tokens = tokenize("src/tools/search/mod.rs");
        assert!(tokens.contains(&"src".to_string()));
        assert!(tokens.contains(&"tools".to_string()));
        assert!(tokens.contains(&"search".to_string()));
    }

    #[test]
    fn normalizes_to_lowercase() {
        let tokens = tokenize("SearchEngine");
        assert!(tokens.iter().any(|t| t == "search"));
    }

    #[test]
    fn filters_short_tokens() {
        let tokens = tokenize("a b cd");
        assert!(!tokens.iter().any(|t| t == "a" || t == "b"));
        assert!(tokens.contains(&"cd".to_string()));
    }

    #[test]
    fn deduplicates() {
        let tokens = tokenize("foo foo foo");
        let count = tokens.iter().filter(|t| **t == "foo").count();
        assert_eq!(count, 1);
    }

    #[test]
    fn empty_input_returns_empty() {
        assert!(tokenize("").is_empty());
    }

    #[test]
    fn splits_on_dots() {
        let tokens = tokenize("config.yaml");
        assert!(tokens.contains(&"config".to_string()));
        assert!(tokens.contains(&"yaml".to_string()));
    }
}
