use illustrious_manager::types::Role;

#[test]
fn test_role_serializes_as_lowercase() {
    let user_role = Role::User;
    let assistant_role = Role::Assistant;

    let user_json = serde_json::to_string(&user_role).unwrap();
    let assistant_json = serde_json::to_string(&assistant_role).unwrap();

    assert_eq!(user_json, r#""user""#);
    assert_eq!(assistant_json, r#""assistant""#);
}
