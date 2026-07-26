use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::path::Path;

use url::{Host, Url};

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum LoraUrlError {
    Invalid,
    UnsupportedScheme,
    MissingFilename,
    UnsafeFilename,
    BlockedHost,
}

impl LoraUrlError {
    pub fn message(&self) -> &'static str {
        match self {
            Self::Invalid => "LoRA sourceUrl must be a valid URL",
            Self::UnsupportedScheme => "LoRA sourceUrl must use http or https",
            Self::MissingFilename => "LoRA sourceUrl must include a filename",
            Self::UnsafeFilename => {
                "LoRA sourceUrl filename must use letters, numbers, dots, dashes, or underscores"
            }
            Self::BlockedHost => "LoRA sourceUrl host is not allowed",
        }
    }
}

pub fn parse_lora_source_url(source_url: &str) -> Result<Url, LoraUrlError> {
    parse_lora_source_url_with_private(source_url, false)
}

pub fn parse_lora_source_url_with_private(
    source_url: &str,
    allow_private_hosts: bool,
) -> Result<Url, LoraUrlError> {
    let url = Url::parse(source_url).map_err(|_| LoraUrlError::Invalid)?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err(LoraUrlError::UnsupportedScheme);
    }
    if !allow_private_hosts {
        validate_lora_url_host(&url)?;
    }
    lora_source_url_file_name(source_url)?;
    Ok(url)
}

pub fn lora_source_url_file_name(source_url: &str) -> Result<String, LoraUrlError> {
    let url = Url::parse(source_url).map_err(|_| LoraUrlError::Invalid)?;
    let file_name = url
        .path_segments()
        .and_then(Iterator::last)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or(LoraUrlError::MissingFilename)?;
    if !safe_lora_filename(file_name) {
        return Err(LoraUrlError::UnsafeFilename);
    }
    Ok(file_name.to_owned())
}

pub fn lora_source_url_file_stem(source_url: &str) -> Result<String, LoraUrlError> {
    let file_name = lora_source_url_file_name(source_url)?;
    Path::new(&file_name)
        .file_stem()
        .and_then(|value| value.to_str())
        .map(str::to_owned)
        .filter(|value| !value.trim().is_empty())
        .ok_or(LoraUrlError::MissingFilename)
}

pub fn safe_lora_filename(file_name: &str) -> bool {
    let trimmed = file_name.trim();
    !trimmed.is_empty()
        && trimmed.len() <= 160
        && trimmed.chars().all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '.' | '-' | '_')
        })
}

pub fn validate_lora_url_host(url: &Url) -> Result<(), LoraUrlError> {
    let Some(host) = url.host() else {
        return Err(LoraUrlError::BlockedHost);
    };
    match host {
        Host::Ipv4(address) => validate_public_ip(IpAddr::V4(address)),
        Host::Ipv6(address) => validate_public_ip(IpAddr::V6(address)),
        Host::Domain(domain) => {
            let domain = domain.trim_end_matches('.').to_ascii_lowercase();
            if domain == "localhost" || domain.ends_with(".localhost") || domain.ends_with(".local")
            {
                return Err(LoraUrlError::BlockedHost);
            }
            Ok(())
        }
    }
}

pub fn validate_public_ip(address: IpAddr) -> Result<(), LoraUrlError> {
    let blocked = match address {
        IpAddr::V4(address) => blocked_ipv4(address),
        IpAddr::V6(address) => match address.to_ipv4_mapped().or_else(|| address.to_ipv4()) {
            // An IPv4-mapped (`::ffff:a.b.c.d`) or IPv4-compatible (`::a.b.c.d`)
            // IPv6 address reaches the very same IPv4 endpoint, so re-run the v4
            // blocklist on the embedded address instead of the native-v6 checks. Without
            // this, `http://[::ffff:127.0.0.1]/x.safetensors` (or a DNS AAAA of
            // `::ffff:10.0.0.1`) slips past both the URL-time host check and the
            // connect-time re-validation (sc-4210 / F-CORE-6 SSRF bypass).
            Some(embedded) => blocked_ipv4(embedded),
            None => blocked_ipv6(address),
        },
    };
    if blocked {
        Err(LoraUrlError::BlockedHost)
    } else {
        Ok(())
    }
}

fn blocked_ipv4(address: Ipv4Addr) -> bool {
    let [a, b, c, _] = address.octets();
    address.is_private()
        || address.is_loopback()
        || address.is_link_local()
        || address.is_broadcast()
        || address.is_documentation()
        || a == 0
        || a >= 224
        // RFC 6598 shared address space.
        || (a == 100 && (64..=127).contains(&b))
        // IETF protocol assignments and the deprecated 6to4 relay anycast.
        || (a == 192 && b == 0 && c == 0)
        || (a == 192 && b == 88 && c == 99)
        // RFC 2544 benchmark networks.
        || (a == 198 && (18..=19).contains(&b))
}

fn blocked_ipv6(address: Ipv6Addr) -> bool {
    let segments = address.segments();
    let first = segments[0];

    // The well-known NAT64 prefix is globally reachable even though it sits
    // outside today's 2000::/3 global-unicast allocation. Preserve it only
    // when its embedded IPv4 destination is itself public.
    if segments[..6] == [0x0064, 0xff9b, 0, 0, 0, 0] {
        let embedded = Ipv4Addr::new(
            (segments[6] >> 8) as u8,
            segments[6] as u8,
            (segments[7] >> 8) as u8,
            segments[7] as u8,
        );
        return blocked_ipv4(embedded);
    }

    // Native globally routable unicast currently occupies 2000::/3. This
    // explicit allocation check rejects unspecified/loopback, multicast,
    // link-local, unique-local, deprecated site-local (fec0::/10), discard,
    // local-use translation, and other reserved space without relying on the
    // unstable std `is_global` classification.
    if (first & 0xe000) != 0x2000 {
        return true;
    }

    // 2001::/23 is IANA's protocol-assignment block and is non-global unless
    // a more-specific allocation says otherwise. Keep the globally usable
    // assignments explicit so unallocated/reserved holes cannot become SSRF
    // bypasses.
    if first == 0x2001 && segments[1] <= 0x01ff {
        let globally_usable_assignment = segments[1] == 0
            // PCP, TURN, and DNS-SD anycast singleton addresses.
            || (segments[1] == 1
                && segments[2..7] == [0, 0, 0, 0, 0]
                && matches!(segments[7], 1..=3))
            // AMT.
            || segments[1] == 3
            // AS112-v6.
            || (segments[1] == 4 && segments[2] == 0x0112)
            // ORCHIDv2 and Drone Remote ID DETs are marked globally
            // reachable in the IANA special-purpose registry.
            || (segments[1] & 0xfff0) == 0x0020
            || (segments[1] & 0xfff0) == 0x0030;
        if !globally_usable_assignment {
            return true;
        }
    }

    // Special-purpose ranges carved out of 2000::/3 that are not globally
    // routable unicast.
    // Documentation prefixes.
    (first == 0x2001 && segments[1] == 0x0db8)
        || (first == 0x3fff && (segments[1] & 0xf000) == 0)
        // Returned/deprecated 6bone space.
        || first == 0x3ffe
        // 6to4 embeds an IPv4 destination in segments 1-2. A private or
        // otherwise special embedded address must not tunnel around the v4
        // policy; genuinely public 6to4 destinations remain allowed.
        || (first == 0x2002
            && blocked_ipv4(Ipv4Addr::new(
                (segments[1] >> 8) as u8,
                segments[1] as u8,
                (segments[2] >> 8) as u8,
                segments[2] as u8,
            )))
}

#[cfg(test)]
mod tests {
    use super::{
        lora_source_url_file_name, parse_lora_source_url, validate_public_ip, LoraUrlError,
    };
    use std::net::{IpAddr, Ipv6Addr};

    fn v6(literal: &str) -> IpAddr {
        IpAddr::V6(literal.parse::<Ipv6Addr>().expect("valid IPv6 literal"))
    }

    /// sc-4210 / F-CORE-6: an IPv4-mapped/compatible IPv6 address reaches the
    /// embedded IPv4 endpoint, so the SSRF guard must unmap and re-run the v4
    /// blocklist. Before the fix these slipped past `validate_public_ip` and the
    /// connect-time re-validation, letting a crafted URL reach loopback/RFC1918.
    #[test]
    fn validate_public_ip_blocks_ipv4_mapped_private_targets() {
        // IPv4-mapped (`::ffff:a.b.c.d`) loopback / RFC1918 / link-local.
        for blocked in ["::ffff:127.0.0.1", "::ffff:10.0.0.1", "::ffff:169.254.0.1"] {
            assert_eq!(
                validate_public_ip(v6(blocked)),
                Err(LoraUrlError::BlockedHost),
                "{blocked} must be blocked"
            );
        }
        // IPv4-compatible (`::a.b.c.d`) loopback.
        assert_eq!(
            validate_public_ip(v6("::127.0.0.1")),
            Err(LoraUrlError::BlockedHost)
        );

        // A mapped *public* IPv4 and a genuine public IPv6 must still pass — the
        // fix blocks the bypass, not all v4-mapped traffic.
        assert!(validate_public_ip(v6("::ffff:1.1.1.1")).is_ok());
        assert!(validate_public_ip(v6("2606:4700:4700::1111")).is_ok());
    }

    /// End-to-end: the bracketed IPv4-mapped literal is rejected by URL parsing,
    /// mirroring the existing literal-`127.0.0.1` case.
    #[test]
    fn parse_lora_source_url_blocks_ipv4_mapped_loopback_host() {
        assert_eq!(
            parse_lora_source_url("http://[::ffff:127.0.0.1]/style.safetensors").unwrap_err(),
            LoraUrlError::BlockedHost
        );
    }

    #[test]
    fn ipv6_literal_parser_blocks_non_global_ranges_and_preserves_global_unicast() {
        for blocked in [
            "::",
            "::1",
            "100::",
            "64:ff9b:1::1",
            "fc00::1",
            "fdff:ffff:ffff:ffff:ffff:ffff:ffff:ffff",
            "fe7f:ffff:ffff:ffff:ffff:ffff:ffff:ffff",
            "fe80::1",
            "febf:ffff:ffff:ffff:ffff:ffff:ffff:ffff",
            "fec0::",
            "feff:ffff:ffff:ffff:ffff:ffff:ffff:ffff",
            "ff00::",
            "2001:2::",
            "2001:1::4",
            "2001:5::",
            "2001:10::",
            "2001:db8::1",
            "2002:0a00:0001::",
            "3ffe::1",
            "3fff::1",
            "4000::1",
        ] {
            assert_eq!(
                parse_lora_source_url(&format!("https://[{blocked}]/style.safetensors")),
                Err(LoraUrlError::BlockedHost),
                "{blocked} must be blocked"
            );
        }

        for public in [
            "::ffff:1.1.1.1",
            "64:ff9b::101:101",
            "2001:1::1",
            "2001:3::1",
            "2001:4:112::1",
            "2001:20::1",
            "2001:30::1",
            "2001:4860:4860::8888",
            "2002:0101:0101::",
            "2606:4700:4700::1111",
        ] {
            assert!(
                parse_lora_source_url(&format!("https://[{public}]/style.safetensors")).is_ok(),
                "{public} must remain allowed"
            );
        }
    }

    #[test]
    fn validate_public_ip_blocks_site_local_fec0_boundaries() {
        for blocked in [
            "febf:ffff:ffff:ffff:ffff:ffff:ffff:ffff",
            "fec0::",
            "feff:ffff:ffff:ffff:ffff:ffff:ffff:ffff",
            "ff00::",
        ] {
            assert_eq!(
                validate_public_ip(v6(blocked)),
                Err(LoraUrlError::BlockedHost),
                "{blocked} must be blocked"
            );
        }
        for public in ["2001:4860:4860::8888", "2606:4700:4700::1111"] {
            assert!(
                validate_public_ip(v6(public)).is_ok(),
                "{public} must remain allowed"
            );
        }
    }

    #[test]
    fn lora_source_urls_validate_scheme_host_and_filename() {
        assert_eq!(
            lora_source_url_file_name("https://example.com/models/style.safetensors").unwrap(),
            "style.safetensors"
        );
        assert_eq!(
            parse_lora_source_url("file:///tmp/style.safetensors").unwrap_err(),
            LoraUrlError::UnsupportedScheme
        );
        assert_eq!(
            parse_lora_source_url("https://example.com/").unwrap_err(),
            LoraUrlError::MissingFilename
        );
        assert_eq!(
            parse_lora_source_url("http://127.0.0.1/style.safetensors").unwrap_err(),
            LoraUrlError::BlockedHost
        );
        assert_eq!(
            parse_lora_source_url("https://example.com/style.safetensors%00.txt").unwrap_err(),
            LoraUrlError::UnsafeFilename
        );
    }

    #[test]
    fn validate_public_ip_blocks_reserved_ipv4_ranges() {
        for blocked in [
            "100.64.0.1",
            "100.127.255.254",
            "192.0.0.1",
            "192.88.99.1",
            "198.18.0.1",
            "198.19.255.254",
        ] {
            assert_eq!(
                validate_public_ip(blocked.parse().expect("valid IP")),
                Err(LoraUrlError::BlockedHost),
                "{blocked} must be blocked"
            );
        }
        for public in ["100.63.255.254", "100.128.0.1", "192.0.1.1", "198.20.0.1"] {
            assert!(
                validate_public_ip(public.parse().expect("valid IP")).is_ok(),
                "{public} must remain allowed"
            );
        }
    }
}
