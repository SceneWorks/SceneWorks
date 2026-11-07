//! Shared case-insensitive Parquet field extraction, with explicit coercion policy.
use parquet::record::{Field, Row};

pub(crate) fn find_field<'a>(row: &'a Row, aliases: &[&str]) -> Option<&'a Field> {
    row.get_column_iter()
        .find(|(name, _)| {
            aliases
                .iter()
                .any(|requested| name.eq_ignore_ascii_case(requested))
        })
        .map(|(_, field)| field)
}

pub(crate) fn row_string_alias(row: &Row, aliases: &[&str]) -> Option<String> {
    match find_field(row, aliases)? {
        Field::Str(value) => Some(value.clone()),
        Field::Bytes(value) => std::str::from_utf8(value.data()).ok().map(str::to_owned),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum MetadataField<T> {
    Missing,
    Invalid,
    Value(T),
}

pub(crate) fn row_u32_alias(row: &Row, aliases: &[&str]) -> MetadataField<u32> {
    row_u32_with_policy(row, aliases, true)
}

/// Integer-only import columns and coercible catalog metadata intentionally remain distinct.
pub(crate) fn row_u32_with_policy(
    row: &Row,
    aliases: &[&str],
    coerce_metadata: bool,
) -> MetadataField<u32> {
    let Some(field) = find_field(row, aliases) else {
        return MetadataField::Missing;
    };
    let value = match field {
        Field::Byte(value) => u32::try_from(*value).ok(),
        Field::Short(value) => u32::try_from(*value).ok(),
        Field::Int(value) => u32::try_from(*value).ok(),
        Field::Long(value) => u32::try_from(*value).ok(),
        Field::UByte(value) => Some((*value).into()),
        Field::UShort(value) => Some((*value).into()),
        Field::UInt(value) => Some(*value),
        Field::ULong(value) => u32::try_from(*value).ok(),
        Field::Float(value) if coerce_metadata => exact_u32((*value).into()),
        Field::Double(value) if coerce_metadata => exact_u32(*value),
        Field::Str(value) if coerce_metadata => value.trim().parse().ok(),
        Field::Bytes(value) if coerce_metadata => std::str::from_utf8(value.data())
            .ok()
            .and_then(|value| value.trim().parse().ok()),
        _ => None,
    };
    match value {
        Some(value) => MetadataField::Value(value),
        None => MetadataField::Invalid,
    }
}

pub(crate) fn row_f64_alias(row: &Row, aliases: &[&str]) -> MetadataField<f64> {
    let Some(field) = find_field(row, aliases) else {
        return MetadataField::Missing;
    };
    let value = match field {
        Field::Byte(value) => Some((*value).into()),
        Field::Short(value) => Some((*value).into()),
        Field::Int(value) => Some((*value).into()),
        Field::Long(value) => Some(*value as f64),
        Field::UByte(value) => Some((*value).into()),
        Field::UShort(value) => Some((*value).into()),
        Field::UInt(value) => Some((*value).into()),
        Field::ULong(value) => Some(*value as f64),
        Field::Float(value) if value.is_finite() => Some((*value).into()),
        Field::Double(value) if value.is_finite() => Some(*value),
        Field::Str(value) => parse_metadata_score(value),
        Field::Bytes(value) => std::str::from_utf8(value.data())
            .ok()
            .and_then(parse_metadata_score),
        _ => None,
    };
    match value {
        Some(value) if value.is_finite() => MetadataField::Value(value),
        _ => MetadataField::Invalid,
    }
}

pub(crate) fn row_probability_alias(row: &Row, aliases: &[&str]) -> MetadataField<f64> {
    match row_f64_alias(row, aliases) {
        MetadataField::Value(value) if (0.0..=1.0).contains(&value) => MetadataField::Value(value),
        MetadataField::Value(_) => MetadataField::Invalid,
        other => other,
    }
}

fn exact_u32(value: f64) -> Option<u32> {
    (value.is_finite() && value >= 0.0 && value <= u32::MAX as f64 && value.fract() == 0.0)
        .then_some(value as u32)
}

fn parse_metadata_score(value: &str) -> Option<f64> {
    let normalized = value.trim().to_ascii_lowercase();
    normalized.parse().ok().or(match normalized.as_str() {
        "safe" | "sfw" | "false" => Some(0.0),
        "unlikely" => Some(0.25),
        "unsure" => Some(0.5),
        "unsafe" | "nsfw" | "true" | "porn" => Some(1.0),
        _ => None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn integer_import_and_catalog_coercion_remain_distinct() {
        for (field, expected) in [
            (Field::Byte(-1), None),
            (Field::Long(42), Some(42)),
            (Field::ULong(u64::MAX), None),
            (Field::Double(42.0), Some(42)),
            (Field::Double(42.5), None),
            (Field::Double(f64::NAN), None),
            (Field::Str(" 42 ".into()), Some(42)),
        ] {
            let integer = matches!(field, Field::Long(_));
            let row = Row::new(vec![("WiDtH".into(), field)]);
            assert_eq!(
                row_u32_alias(&row, &["width"]),
                expected
                    .map(MetadataField::Value)
                    .unwrap_or(MetadataField::Invalid)
            );
            assert_eq!(
                row_u32_with_policy(&row, &["WIDTH"], false),
                if integer {
                    MetadataField::Value(42)
                } else {
                    MetadataField::Invalid
                }
            );
            assert_eq!(row_u32_alias(&row, &["absent"]), MetadataField::Missing);
        }
    }
    #[test]
    fn first_column_alias_precedence_and_scores_are_preserved() {
        let row = Row::new(vec![
            ("text".into(), Field::Str("first".into())),
            ("caption".into(), Field::Str("second".into())),
            ("nsfw".into(), Field::Str("unsafe".into())),
            ("bad".into(), Field::Double(f64::INFINITY)),
        ]);
        assert_eq!(
            row_string_alias(&row, &["caption", "TEXT"]).as_deref(),
            Some("first")
        );
        assert_eq!(
            row_probability_alias(&row, &["NSFW"]),
            MetadataField::Value(1.0)
        );
        assert_eq!(row_f64_alias(&row, &["bad"]), MetadataField::Invalid);
    }
}
