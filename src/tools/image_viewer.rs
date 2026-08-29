use crate::tools::sandbox::SandboxPolicy;
use crate::tools::{Tool, ToolError, ToolResult};
use crate::types::ContentBlock;
use async_trait::async_trait;
use base64::Engine;
use serde_json::Value;
use std::path::Path;

const MAX_IMAGE_BYTES: usize = 5 * 1024 * 1024;

#[derive(Debug)]
pub struct ImageViewer {
    sandbox: SandboxPolicy,
    schema: Value,
}

impl ImageViewer {
    pub fn new(sandbox: SandboxPolicy) -> Self {
        Self {
            sandbox,
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the image file (relative to sandbox root or absolute within an allowed root). Supported formats: png, jpeg, gif, webp."
                    }
                },
                "required": ["path"]
            }),
        }
    }
}

fn mime_from_extension(ext: &str) -> Option<String> {
    match ext.to_lowercase().as_str() {
        "png" => Some("image/png".to_string()),
        "jpg" | "jpeg" => Some("image/jpeg".to_string()),
        "gif" => Some("image/gif".to_string()),
        "webp" => Some("image/webp".to_string()),
        _ => None,
    }
}

fn size_limit_error(bytes: u64) -> ToolError {
    ToolError::Execution {
        tool_name: "image_viewer".to_string(),
        message: format!(
            "Image is {bytes} bytes; maximum allowed is {} bytes (5MB)",
            MAX_IMAGE_BYTES
        ),
    }
}

fn validate_image_header(data: &[u8], media_type: &str) -> bool {
    match media_type {
        "image/png" => {
            data.len() >= 8 && data[..8] == [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]
        }
        "image/jpeg" => data.len() >= 3 && data[..3] == [0xFF, 0xD8, 0xFF],
        "image/gif" => data.len() >= 6 && matches!(&data[..6], b"GIF87a" | b"GIF89a"),
        "image/webp" => data.len() >= 12 && &data[..4] == b"RIFF" && &data[8..12] == b"WEBP",
        _ => false,
    }
}

#[async_trait]
impl Tool for ImageViewer {
    fn name(&self) -> &str {
        "image_viewer"
    }

    fn description(&self) -> &str {
        "Read an image file from disk and return it to the LLM as image content. Supports PNG, JPEG, GIF, and WebP up to 5MB."
    }

    fn input_schema(&self) -> &Value {
        &self.schema
    }

    async fn execute(&self, input: Value) -> Result<ToolResult, ToolError> {
        let path_str =
            input
                .get("path")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ToolError::InvalidInput {
                    message: "Missing required 'path' field".to_string(),
                })?;

        let validated = self
            .sandbox
            .validate_path(Path::new(path_str))
            .map_err(|e| ToolError::Execution {
                tool_name: "image_viewer".to_string(),
                message: e.to_string(),
            })?;

        let media_type = validated
            .extension()
            .and_then(|e| e.to_str())
            .and_then(mime_from_extension)
            .ok_or_else(|| ToolError::Execution {
                tool_name: "image_viewer".to_string(),
                message: format!(
                    "Unsupported file extension for path {path_str}; supported: png, jpg/jpeg, gif, webp"
                ),
            })?;

        let file_size = std::fs::metadata(&validated)
            .map_err(|e| ToolError::Execution {
                tool_name: "image_viewer".to_string(),
                message: format!("Failed to stat {path_str}: {e}"),
            })?
            .len();
        if file_size > MAX_IMAGE_BYTES as u64 {
            return Err(size_limit_error(file_size));
        }

        let data = std::fs::read(&validated).map_err(|e| ToolError::Execution {
            tool_name: "image_viewer".to_string(),
            message: format!("Failed to read {path_str}: {e}"),
        })?;

        if data.len() > MAX_IMAGE_BYTES {
            return Err(size_limit_error(data.len() as u64));
        }

        if !validate_image_header(&data, &media_type) {
            return Err(ToolError::Execution {
                tool_name: "image_viewer".to_string(),
                message: format!(
                    "File header does not match expected {media_type} signature; the file may be corrupt or misnamed"
                ),
            });
        }

        let encoded = base64::engine::general_purpose::STANDARD.encode(&data);
        let display_name = validated
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(path_str);

        Ok(ToolResult {
            content: vec![
                ContentBlock::Image {
                    media_type: media_type.clone(),
                    data: encoded,
                },
                ContentBlock::Text(format!(
                    "Image loaded: {display_name} ({}, {} bytes)",
                    media_type,
                    data.len()
                )),
            ],
            is_error: false,
            agent_events: vec![],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_policy(dir: &TempDir) -> SandboxPolicy {
        SandboxPolicy::new(dir.path())
    }

    #[tokio::test]
    async fn happy_path_png() {
        let dir = TempDir::new().expect("temp dir");
        let policy = make_policy(&dir);
        let tool = ImageViewer::new(policy);
        let png = dir.path().join("test.png");
        let header = [0x89u8, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        std::fs::write(&png, header).expect("write");
        let result = tool
            .execute(serde_json::json!({"path": "test.png"}))
            .await
            .expect("ok");
        assert!(!result.is_error);
        assert_eq!(result.content.len(), 2);
        match &result.content[0] {
            ContentBlock::Image { media_type, .. } => assert_eq!(media_type, "image/png"),
            _ => panic!("expected Image block"),
        }
        match &result.content[1] {
            ContentBlock::Text(t) => assert!(t.contains("image/png")),
            _ => panic!("expected Text companion"),
        }
    }

    #[tokio::test]
    async fn happy_path_jpeg() {
        let dir = TempDir::new().expect("temp dir");
        let policy = make_policy(&dir);
        let tool = ImageViewer::new(policy);
        let jpg = dir.path().join("photo.jpg");
        std::fs::write(&jpg, [0xFFu8, 0xD8, 0xFF, 0xE0]).expect("write");
        let result = tool
            .execute(serde_json::json!({"path": "photo.jpg"}))
            .await
            .expect("ok");
        assert!(!result.is_error);
        match &result.content[0] {
            ContentBlock::Image { media_type, .. } => assert_eq!(media_type, "image/jpeg"),
            _ => panic!("expected Image block"),
        }
    }

    #[tokio::test]
    async fn happy_path_gif() {
        let dir = TempDir::new().expect("temp dir");
        let policy = make_policy(&dir);
        let tool = ImageViewer::new(policy);
        let gif = dir.path().join("anim.gif");
        std::fs::write(&gif, b"GIF89aextra").expect("write");
        let result = tool
            .execute(serde_json::json!({"path": "anim.gif"}))
            .await
            .expect("ok");
        assert!(!result.is_error);
        match &result.content[0] {
            ContentBlock::Image { media_type, .. } => assert_eq!(media_type, "image/gif"),
            _ => panic!("expected Image block"),
        }
    }

    #[tokio::test]
    async fn happy_path_webp() {
        let dir = TempDir::new().expect("temp dir");
        let policy = make_policy(&dir);
        let tool = ImageViewer::new(policy);
        let webp = dir.path().join("pic.webp");
        let mut data = b"RIFF".to_vec();
        data.extend_from_slice(&[0u8, 0, 0, 0]);
        data.extend_from_slice(b"WEBP");
        std::fs::write(&webp, &data).expect("write");
        let result = tool
            .execute(serde_json::json!({"path": "pic.webp"}))
            .await
            .expect("ok");
        assert!(!result.is_error);
        match &result.content[0] {
            ContentBlock::Image { media_type, .. } => assert_eq!(media_type, "image/webp"),
            _ => panic!("expected Image block"),
        }
    }

    #[tokio::test]
    async fn oversized_image_rejected() {
        let dir = TempDir::new().expect("temp dir");
        let policy = make_policy(&dir);
        let tool = ImageViewer::new(policy);
        let big = dir.path().join("big.png");
        let mut data = vec![0x89u8, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        data.extend(std::iter::repeat_n(0u8, MAX_IMAGE_BYTES + 1));
        std::fs::write(&big, &data).expect("write");
        let err = tool
            .execute(serde_json::json!({"path": "big.png"}))
            .await
            .expect_err("should reject");
        match err {
            ToolError::Execution { message, .. } => assert!(message.contains("maximum")),
            _ => panic!("expected Execution error"),
        }
    }

    #[tokio::test]
    async fn unknown_extension_rejected() {
        let dir = TempDir::new().expect("temp dir");
        let policy = make_policy(&dir);
        let tool = ImageViewer::new(policy);
        let file = dir.path().join("file.bmp");
        std::fs::write(&file, b"BM").expect("write");
        let err = tool
            .execute(serde_json::json!({"path": "file.bmp"}))
            .await
            .expect_err("should reject");
        match err {
            ToolError::Execution { message, .. } => assert!(message.contains("Unsupported")),
            _ => panic!("expected Execution error"),
        }
    }

    #[tokio::test]
    async fn nonexistent_path_rejected() {
        let dir = TempDir::new().expect("temp dir");
        let policy = make_policy(&dir);
        let tool = ImageViewer::new(policy);
        let err = tool
            .execute(serde_json::json!({"path": "nope.png"}))
            .await
            .expect_err("should reject");
        match err {
            ToolError::Execution { message, .. } => assert!(message.contains("does not exist")),
            _ => panic!("expected Execution error"),
        }
    }

    #[tokio::test]
    async fn out_of_sandbox_rejected() {
        let dir = TempDir::new().expect("temp dir");
        let outside = TempDir::new().expect("outside dir");
        let policy = make_policy(&dir);
        let tool = ImageViewer::new(policy);
        let outside_file = outside.path().join("escape.png");
        std::fs::write(
            &outside_file,
            [0x89u8, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A],
        )
        .expect("write");
        let err = tool
            .execute(serde_json::json!({"path": outside_file.to_str().expect("path")}))
            .await
            .expect_err("should reject");
        match err {
            ToolError::Execution { message, .. } => assert!(message.contains("outside sandbox")),
            _ => panic!("expected Execution error"),
        }
    }

    #[tokio::test]
    async fn corrupted_png_rejected() {
        let dir = TempDir::new().expect("temp dir");
        let policy = make_policy(&dir);
        let tool = ImageViewer::new(policy);
        let fake = dir.path().join("fake.png");
        std::fs::write(&fake, b"not a png at all").expect("write");
        let err = tool
            .execute(serde_json::json!({"path": "fake.png"}))
            .await
            .expect_err("should reject");
        match err {
            ToolError::Execution { message, .. } => assert!(message.contains("header")),
            _ => panic!("expected Execution error"),
        }
    }

    #[test]
    fn is_write_tool_is_false() {
        let dir = TempDir::new().expect("temp dir");
        let tool = ImageViewer::new(make_policy(&dir));
        assert!(!tool.is_write_tool());
    }
}
