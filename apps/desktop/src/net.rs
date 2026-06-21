//! Likely-LAN-address discovery for the remote-access URL (epic 4484, story 5).
//!
//! When the user enables LAN remote access the Settings UI shows the host URL
//! (`http://<ip>:<port>`) so another device can reach it. This module enumerates the
//! host's network interfaces and picks a best-guess private IPv4 to put in that URL,
//! returning the full candidate list too so the UI can offer a picker if the guess is
//! wrong. Cross-platform via `if-addrs` (lists adapters with their IPs on macOS AND
//! Windows); the private-IPv4 heuristic is name-agnostic, so it doesn't assume macOS
//! `en0`-style naming.

use std::net::{IpAddr, Ipv4Addr};

use serde::Serialize;

/// Best-guess LAN IPv4 plus all private candidates for the remote-access URL.
#[derive(Serialize, Default, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct LanAddresses {
    /// Best-guess private LAN IPv4 to show in the remote URL, or `None` if the host
    /// has no private IPv4 (e.g. offline / loopback-only). `candidates.first()`.
    pub primary: Option<String>,
    /// Every private LAN IPv4 found, best-first, so the UI can offer a picker when the
    /// primary guess is wrong (multi-homed: Wi-Fi + Ethernet + VPN).
    pub candidates: Vec<String>,
}

/// Whether an address is a usable private LAN IPv4. `Ipv4Addr::is_private` already
/// excludes loopback (127/8) and link-local (169.254/16) — they aren't in the
/// 10/8, 172.16/12, 192.168/16 private ranges — but the extra guards make the intent
/// explicit and survive any future loosening of `is_private`.
fn is_private_lan_ipv4(ip: &Ipv4Addr) -> bool {
    ip.is_private() && !ip.is_loopback() && !ip.is_link_local() && !ip.is_unspecified()
}

/// Preference rank for a private IPv4 (lower = more likely the real LAN address on a
/// typical home/office Wi-Fi/Ethernet setup): 192.168/16 first, then 10/8, then
/// 172.16-31/12 (often Docker/VM bridges), then anything else.
fn rank(ip: &Ipv4Addr) -> u8 {
    match ip.octets() {
        [192, 168, ..] => 0,
        [10, ..] => 1,
        [172, b, ..] if (16..=31).contains(&b) => 2,
        _ => 3,
    }
}

/// Order + de-duplicate discovered private IPv4s into a [`LanAddresses`]. Pure so the
/// ranking is unit-tested without touching real interfaces.
fn rank_addresses(mut found: Vec<Ipv4Addr>) -> LanAddresses {
    found.sort_by_key(|ip| (rank(ip), ip.octets()));
    found.dedup();
    let candidates: Vec<String> = found.iter().map(Ipv4Addr::to_string).collect();
    LanAddresses {
        primary: candidates.first().cloned(),
        candidates,
    }
}

/// Enumerate the host's private LAN IPv4 addresses, best guess first. Skips
/// loopback/link-local/down interfaces (a down interface exposes no address, so
/// `if-addrs` omits it). Returns an empty [`LanAddresses`] (the UI shows "unknown")
/// when none is found rather than erroring.
pub fn lan_addresses() -> LanAddresses {
    let mut found: Vec<Ipv4Addr> = Vec::new();
    if let Ok(interfaces) = if_addrs::get_if_addrs() {
        for interface in interfaces {
            if interface.is_loopback() {
                continue;
            }
            if let IpAddr::V4(v4) = interface.ip() {
                if is_private_lan_ipv4(&v4) {
                    found.push(v4);
                }
            }
        }
    }
    rank_addresses(found)
}

/// Tauri command: the likely LAN address(es) for the remote-access URL (story 4
/// combines `primary` with the configured port).
#[tauri::command]
pub fn get_lan_address() -> LanAddresses {
    lan_addresses()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_non_private_and_loopback_and_link_local() {
        assert!(!is_private_lan_ipv4(&Ipv4Addr::new(127, 0, 0, 1))); // loopback
        assert!(!is_private_lan_ipv4(&Ipv4Addr::new(169, 254, 1, 1))); // link-local
        assert!(!is_private_lan_ipv4(&Ipv4Addr::new(8, 8, 8, 8))); // public
        assert!(!is_private_lan_ipv4(&Ipv4Addr::new(0, 0, 0, 0))); // unspecified
        assert!(is_private_lan_ipv4(&Ipv4Addr::new(192, 168, 1, 50)));
        assert!(is_private_lan_ipv4(&Ipv4Addr::new(10, 0, 0, 4)));
        assert!(is_private_lan_ipv4(&Ipv4Addr::new(172, 16, 5, 9)));
        // 172.32 is outside the private 172.16-31 block.
        assert!(!is_private_lan_ipv4(&Ipv4Addr::new(172, 32, 0, 1)));
    }

    #[test]
    fn ranks_192_168_first_then_10_then_172() {
        let result = rank_addresses(vec![
            Ipv4Addr::new(172, 17, 0, 1),
            Ipv4Addr::new(10, 1, 2, 3),
            Ipv4Addr::new(192, 168, 0, 42),
        ]);
        assert_eq!(result.primary.as_deref(), Some("192.168.0.42"));
        assert_eq!(
            result.candidates,
            vec!["192.168.0.42", "10.1.2.3", "172.17.0.1"]
        );
    }

    #[test]
    fn empty_when_no_private_address() {
        let result = rank_addresses(Vec::new());
        assert_eq!(result.primary, None);
        assert!(result.candidates.is_empty());
    }

    #[test]
    fn dedupes_repeated_addresses() {
        let result = rank_addresses(vec![
            Ipv4Addr::new(192, 168, 1, 5),
            Ipv4Addr::new(192, 168, 1, 5),
        ]);
        assert_eq!(result.candidates, vec!["192.168.1.5"]);
    }
}
