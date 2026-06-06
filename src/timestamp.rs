use time::OffsetDateTime;

pub fn format_timestamp(unix_secs: f64) -> String {
    let secs = unix_secs as i64;
    let utc_dt = OffsetDateTime::from_unix_timestamp(secs).unwrap_or(OffsetDateTime::UNIX_EPOCH);
    let local_dt = match time::UtcOffset::current_local_offset() {
        Ok(local_offset) => utc_dt.to_offset(local_offset),
        Err(_) => utc_dt,
    };
    format!(
        "[{}{:02}{:02}-{:02}:{:02}]",
        local_dt.year(),
        local_dt.month() as u8,
        local_dt.day(),
        local_dt.hour(),
        local_dt.minute()
    )
}

pub fn now_timestamp() -> f64 {
    OffsetDateTime::now_utc().unix_timestamp() as f64
}

pub fn format_now_timestamp() -> String {
    format_timestamp(now_timestamp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_timestamp_epoch() {
        let result = format_timestamp(0.0);
        assert_eq!(result, "[19700101-00:00]");
    }

    #[test]
    fn format_timestamp_known_date() {
        let result = format_timestamp(1704348000.0);
        assert_eq!(result, "[20240104-06:00]");
    }

    #[test]
    fn format_timestamp_truncates_fractional_seconds() {
        let result = format_timestamp(1704348000.999);
        assert_eq!(result, "[20240104-06:00]");
    }

    #[test]
    fn format_now_timestamp_produces_bracketed_format() {
        let result = format_now_timestamp();
        assert!(result.starts_with('['), "should start with '[': {result}");
        assert!(result.ends_with(']'), "should end with ']': {result}");
    }

    #[test]
    fn format_timestamp_negative_timestamp() {
        let result = format_timestamp(-1.0);
        assert!(result.starts_with('['), "should start with '[': {result}");
    }
}
