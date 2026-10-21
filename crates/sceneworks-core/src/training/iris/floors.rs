//! Units and provenance of the Iris training memory floors (sc-25690).
//!
//! **Every floor is GiB** (2^30 bytes). The worker gates compare a floor to capacities that are GiB:
//! an NVIDIA card's `memory.total` MiB ÷ 1024 (a "96 GB" RTX PRO 6000 reports 97,887 MiB = 95.59
//! GiB) and a Mac's `hw.memsize` bytes ÷ 2^30. Probes report peaks in other units — decimal GB,
//! MiB, bytes — so a measured row is entered as the [`Peak`] its probe printed and converted here,
//! once: `GiB = decimal GB ÷ 1.073741824`, plus the lane's allocator reserve, rounded UP.
//!
//! **Provenance is per row** ([`FloorProvenance`]): a table mixes rows measured at their own shape,
//! rows extrapolated from measured ones, rows declared above a measured lower bound, and rows that
//! are plain declarations. The published `memoryFloors.provenance` map, the `basis` text and every
//! refusal message read the row's own provenance, never a table-wide flag.

use serde::Serialize;
use serde_json::{json, Map, Value};

/// The unit of every Iris training floor and of the capacity it is judged against.
pub const FLOOR_UNIT: &str = "GiB";

/// Decimal GB per GiB (2^30 / 10^9).
pub const DECIMAL_GB_PER_GIB: f64 = 1.073_741_824;

/// The allocator reserve (GiB) a Candle floor adds on top of a measured device-VRAM peak.
pub const CANDLE_RESERVE_GIB: u32 = 2;

/// The margin (GiB) an MLX floor adds on top of a measured ACTIVE peak. The MLX peaks are taken
/// with the buffer cache disabled (what the run needs, not what the allocator holds); the process
/// footprint in the same runs sat within 0.5 GB of the active peak, and the rest covers allocator
/// granularity and run-to-run variation, as the Candle reserve does. (The macOS/app reserve is
/// separate: the gate holds it out of the machine's unified memory, not out of the floor.)
pub const MLX_RESERVE_GIB: u32 = 2;

/// The card the epic-25678 Candle measurements ran on: an RTX PRO 6000 Blackwell ("96 GB"),
/// `nvidia-smi` `memory.total` 97,887 MiB = 95.59 GiB. A row measured on it is held to it
/// ([`IrisMemoryFloor::held_on`]): the card that demonstrably ran a shape is never refused for it.
pub const CAMPAIGN_CUDA_CARD: Peak = Peak::Mib(97_887.0);

/// A measured peak, in the unit its probe reported.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Peak {
    /// Decimal gigabytes (10^9 bytes) — what the `train_memory_probe` summaries print.
    DecimalGb(f64),
    /// Gibibytes (2^30 bytes).
    Gib(f64),
    /// Mebibytes (2^20 bytes) — NVML / `nvidia-smi` memory figures.
    Mib(f64),
    /// Bytes — `proc_pid_rusage` `ri_phys_footprint` / `ri_lifetime_max_phys_footprint`.
    Bytes(u64),
}

impl Peak {
    /// The peak in GiB.
    pub const fn gib(self) -> f64 {
        match self {
            Self::DecimalGb(gb) => gb / DECIMAL_GB_PER_GIB,
            Self::Gib(gib) => gib,
            Self::Mib(mib) => mib / 1024.0,
            Self::Bytes(bytes) => bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        }
    }
}

/// `gib` rounded UP to a whole GiB (a floor never rounds a need down).
pub const fn ceil_gib(gib: f64) -> u32 {
    assert!(
        gib >= 0.0 && gib < 1_000_000.0,
        "a floor is a finite, non-negative GiB count"
    );
    let whole = gib as u32;
    if (whole as f64) < gib {
        whole + 1
    } else {
        whole
    }
}

/// What a floor row rests on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FloorProvenance {
    /// The peak of a real run at this row's own shape (+ the lane's reserve).
    Measured,
    /// The need of a real run at this row's own shape that did NOT fit the device it was measured
    /// on: the run completed only by spilling to system memory (far slower). The row is the true
    /// need (+ reserve) — which refuses that device — and never claims the device held it.
    MeasuredSpill,
    /// Derived from measured rows at other shapes (a slope, a cross-lane gap, or the bound a
    /// measured larger run places on a smaller one); no run at this shape backs it.
    Extrapolated,
    /// A real run at this shape was stopped (guard, out of memory) before its peak, so only a
    /// LOWER bound is measured; the row is declared above it.
    LowerBound,
    /// Declared from evidence at other shapes or arms, padded; not measured.
    Declared,
}

impl FloorProvenance {
    /// The contract spelling (`memoryFloors.provenance`).
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Measured => "measured",
            Self::MeasuredSpill => "measured_spill",
            Self::Extrapolated => "extrapolated",
            Self::LowerBound => "lower_bound",
            Self::Declared => "declared",
        }
    }

    /// How a refusal names a floor of this provenance ("needs at least N GiB … (<this>)").
    pub const fn describe(self) -> &'static str {
        match self {
            Self::Measured => "a measured floor",
            Self::MeasuredSpill => {
                "a measured need: on a 96 GB card the run completed only by spilling to system \
                 memory"
            }
            Self::Extrapolated => "a floor extrapolated from measured runs, not measured itself",
            Self::LowerBound => {
                "a declared floor above a measured lower bound; the true peak is not measured"
            }
            Self::Declared => "a declared floor, not yet measured",
        }
    }
}

/// One floor row: whole GiB and what it rests on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IrisMemoryFloor {
    pub gib: u32,
    pub provenance: FloorProvenance,
}

impl IrisMemoryFloor {
    /// A row measured at its own shape: the peak converted to GiB, plus the lane's allocator
    /// reserve, rounded UP.
    pub const fn measured(peak: Peak, reserve_gib: u32) -> Self {
        Self {
            gib: ceil_gib(peak.gib() + reserve_gib as f64),
            provenance: FloorProvenance::Measured,
        }
    }

    /// A row measured at its own shape on a device of `device_gib` that could NOT hold it: the run
    /// completed only by spilling to system memory. The floor is the need (+ reserve), and a need
    /// that would fit the device is not a spill — that does not compile.
    pub const fn measured_with_spill(peak: Peak, reserve_gib: u32, device_gib: f64) -> Self {
        assert!(
            peak.gib() > device_gib,
            "a spilled measurement is a need above the device it was measured on"
        );
        Self {
            gib: ceil_gib(peak.gib() + reserve_gib as f64),
            provenance: FloorProvenance::MeasuredSpill,
        }
    }

    /// The measured run completed on a host of `host_gib` capacity WITHOUT spilling, so that host
    /// is enough: the floor (peak + reserve) never exceeds the capacity that demonstrably held the
    /// run. A spilled measurement is never held to its device — that does not compile.
    pub const fn held_on(self, host_gib: f64) -> Self {
        assert!(
            matches!(self.provenance, FloorProvenance::Measured),
            "only a run that fit its host is held to that host"
        );
        let host = host_gib as u32;
        Self {
            gib: if self.gib > host { host } else { self.gib },
            provenance: self.provenance,
        }
    }

    /// A row derived from measured rows at other shapes: `estimate_gib` (already GiB — build it from
    /// [`Peak::gib`] values) plus the reserve, rounded UP.
    pub const fn extrapolated(estimate_gib: f64, reserve_gib: u32) -> Self {
        Self {
            gib: ceil_gib(estimate_gib + reserve_gib as f64),
            provenance: FloorProvenance::Extrapolated,
        }
    }

    /// A row declared at `gib`, above the measured lower `bound` of a run that was stopped before
    /// its peak. A declaration at or under the bound does not compile.
    pub const fn above_lower_bound(bound: Peak, gib: u32) -> Self {
        assert!(
            gib > ceil_gib(bound.gib()),
            "a lower-bound row must be declared above its measured lower bound"
        );
        Self {
            gib,
            provenance: FloorProvenance::LowerBound,
        }
    }

    /// A plain declaration.
    pub const fn declared(gib: u32) -> Self {
        Self {
            gib,
            provenance: FloorProvenance::Declared,
        }
    }

    /// A row that takes `other`'s value as its bound — `other` being a run that needs at least as
    /// much (a larger resolution, or the same run without activation checkpointing). A bound taken
    /// from a measured row is an extrapolation at this row's shape, not a measurement of it.
    pub const fn bound_from(other: Self) -> Self {
        Self {
            gib: other.gib,
            provenance: match other.provenance {
                FloorProvenance::Measured | FloorProvenance::MeasuredSpill => {
                    FloorProvenance::Extrapolated
                }
                other => other,
            },
        }
    }

    /// This row bounded by `larger` ([`Self::bound_from`]): the lower of the two.
    pub const fn bounded_by(self, larger: Self) -> Self {
        if larger.gib < self.gib {
            Self::bound_from(larger)
        } else {
            self
        }
    }

    /// The floor of a whole run on one device: `ranks` ranks of this row, each holding `extra_gib`
    /// more (the REPA teacher). Only the single-rank, no-extra case IS the row; anything else is
    /// arithmetic on it, so a measured row becomes an extrapolation. `None` on overflow (refuse).
    pub fn for_run(self, ranks: u32, extra_gib: u32) -> Option<Self> {
        let gib = self.gib.checked_add(extra_gib)?.checked_mul(ranks)?;
        let provenance = match self.provenance {
            FloorProvenance::Measured | FloorProvenance::MeasuredSpill
                if ranks != 1 || extra_gib != 0 =>
            {
                FloorProvenance::Extrapolated
            }
            other => other,
        };
        Some(Self { gib, provenance })
    }
}

/// The published `memoryFloors` object of a table: `lanes` (GiB) and `provenance` (per row, the
/// same nesting), the unit, and a `basis` that states the unit and method and then lists the rows
/// under each provenance — generated from the rows, so it cannot disagree with them. `path` is a
/// row's nesting under its lane (`[artifact, size]` or `[size]`).
pub fn memory_floors_json<'a>(
    key: Option<&str>,
    method: &str,
    rows: impl IntoIterator<Item = (&'a str, Vec<String>, IrisMemoryFloor)>,
) -> Value {
    let mut lanes = Map::new();
    let mut provenance = Map::new();
    let mut listed: [Vec<String>; 5] = Default::default();
    for (lane, path, floor) in rows {
        insert_row(&mut lanes, lane, &path, json!(floor.gib));
        insert_row(
            &mut provenance,
            lane,
            &path,
            json!(floor.provenance.as_str()),
        );
        let slot = match floor.provenance {
            FloorProvenance::Measured => 0,
            FloorProvenance::Extrapolated => 1,
            FloorProvenance::LowerBound => 2,
            FloorProvenance::Declared => 3,
            FloorProvenance::MeasuredSpill => 4,
        };
        listed[slot].push(format!("{lane} {}", path.join(" ")));
    }
    let mut basis = format!(
        "Floors are {FLOOR_UNIT} (2^30 bytes), judged against the device's capacity in {FLOOR_UNIT} \
         (a \"96 GB\" NVIDIA card is 95.59 {FLOOR_UNIT}; an MLX floor is judged against unified \
         memory minus the macOS/app reserve). {method} Each row carries its own provenance."
    );
    for (label, rows) in [
        ("Measured at the row's own shape", &listed[0]),
        (
            "Measured need that did not fit the measuring device (completed only by spilling to \
             system memory)",
            &listed[4],
        ),
        ("Extrapolated from measured rows", &listed[1]),
        ("Declared above a measured lower bound", &listed[2]),
        ("Declared, not measured", &listed[3]),
    ] {
        if !rows.is_empty() {
            basis.push_str(&format!(" {label}: {}.", rows.join("; ")));
        }
    }
    basis.push_str(
        " A machine under its floor is refused before the model loads, and a shape above the last \
         row is refused rather than admitted without a bound.",
    );
    let mut out = Map::new();
    out.insert("unit".to_owned(), json!(FLOOR_UNIT));
    if let Some(key) = key {
        out.insert("key".to_owned(), json!(key));
    }
    out.insert("basis".to_owned(), json!(basis));
    out.insert("lanes".to_owned(), Value::Object(lanes));
    out.insert("provenance".to_owned(), Value::Object(provenance));
    Value::Object(out)
}

fn insert_row(root: &mut Map<String, Value>, lane: &str, path: &[String], leaf: Value) {
    let mut node = root
        .entry(lane.to_owned())
        .or_insert_with(|| Value::Object(Map::new()));
    let (last, parents) = path.split_last().expect("a floor row has a size key");
    for part in parents {
        node = node
            .as_object_mut()
            .expect("a floor table level is an object")
            .entry(part.clone())
            .or_insert_with(|| Value::Object(Map::new()));
    }
    node.as_object_mut()
        .expect("a floor table level is an object")
        .insert(last.clone(), leaf);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Decimal GB → GiB is ÷ 1.073741824, the reserve is added in GiB, and the sum rounds UP. The
    /// 97.6 decimal-GB peak is 90.9 GiB: with a 2 GiB reserve it is a 93 GiB floor, not the 100 a
    /// decimal reading compared against GiB capacity produced.
    #[test]
    fn a_measured_peak_converts_to_gib_before_the_reserve_and_rounds_up() {
        assert!((Peak::DecimalGb(1.073_741_824).gib() - 1.0).abs() < 1e-12);
        assert!((Peak::DecimalGb(97.6).gib() - 90.897).abs() < 1e-3);
        assert_eq!(Peak::Mib(97_887.0).gib(), 97_887.0 / 1024.0);
        assert_eq!(Peak::Bytes(3 << 30).gib(), 3.0);
        assert_eq!(Peak::Gib(93.7).gib(), 93.7);

        assert_eq!(IrisMemoryFloor::measured(Peak::DecimalGb(97.6), 2).gib, 93);
        // Exactly whole stays whole; anything above goes to the next GiB.
        assert_eq!(IrisMemoryFloor::measured(Peak::Gib(64.0), 0).gib, 64);
        assert_eq!(IrisMemoryFloor::measured(Peak::Gib(64.001), 0).gib, 65);
        assert_eq!(
            IrisMemoryFloor::measured(Peak::Bytes((64 << 30) + 1), 2).gib,
            67
        );
        assert_eq!(ceil_gib(0.0), 0);
    }

    /// A run that completed on a host is admitted by that host: peak + reserve is capped at the
    /// capacity that held it.
    #[test]
    fn a_measured_floor_never_exceeds_the_host_that_held_the_run() {
        let card = CAMPAIGN_CUDA_CARD.gib();
        assert!((card - 95.59).abs() < 0.01, "{card}");
        let tight = IrisMemoryFloor::measured(Peak::Gib(94.8), 2);
        assert_eq!(tight.gib, 97);
        assert_eq!(tight.held_on(card).gib, 95);
        assert_eq!(tight.held_on(card).provenance, FloorProvenance::Measured);
        let roomy = IrisMemoryFloor::measured(Peak::Gib(60.0), 2);
        assert_eq!(roomy.held_on(card), roomy);
    }

    #[test]
    fn provenance_follows_how_a_row_was_built() {
        assert_eq!(
            IrisMemoryFloor::extrapolated(100.2, 2),
            IrisMemoryFloor {
                gib: 103,
                provenance: FloorProvenance::Extrapolated
            }
        );
        assert_eq!(
            IrisMemoryFloor::above_lower_bound(Peak::DecimalGb(116.75), 128).provenance,
            FloorProvenance::LowerBound
        );
        let declared = IrisMemoryFloor::declared(80);
        let measured = IrisMemoryFloor::measured(Peak::Gib(78.2), 0);
        // A smaller run bounded by a measured larger one: the lower value, as an extrapolation.
        assert_eq!(
            declared.bounded_by(measured),
            IrisMemoryFloor {
                gib: 79,
                provenance: FloorProvenance::Extrapolated
            }
        );
        // A bound that is not lower changes nothing.
        assert_eq!(declared.bounded_by(IrisMemoryFloor::declared(90)), declared);
        // A measured row reused at another shape is never called measured there.
        assert_eq!(
            IrisMemoryFloor::bound_from(measured).provenance,
            FloorProvenance::Extrapolated
        );
        assert_eq!(IrisMemoryFloor::bound_from(declared), declared);

        // The whole-run floor: only one rank with no extra IS the measured row.
        assert_eq!(measured.for_run(1, 0), Some(measured));
        assert_eq!(
            measured.for_run(2, 4),
            Some(IrisMemoryFloor {
                gib: 2 * (79 + 4),
                provenance: FloorProvenance::Extrapolated
            })
        );
        assert_eq!(
            declared.for_run(2, 0).unwrap().provenance,
            FloorProvenance::Declared
        );
        assert_eq!(measured.for_run(u32::MAX, 0), None);
    }

    /// A need measured by spilling is the need itself — above the device it ran on, so that device
    /// is refused — and is never reported as a run the device held.
    #[test]
    fn a_spilled_measurement_is_a_need_above_its_device() {
        let card = CAMPAIGN_CUDA_CARD.gib();
        let spilled = IrisMemoryFloor::measured_with_spill(Peak::DecimalGb(128.75), 2, card);
        assert_eq!(
            spilled,
            IrisMemoryFloor {
                gib: 122,
                provenance: FloorProvenance::MeasuredSpill
            }
        );
        assert!(f64::from(spilled.gib) > card);
        assert_eq!(FloorProvenance::MeasuredSpill.as_str(), "measured_spill");
        assert!(FloorProvenance::MeasuredSpill
            .describe()
            .contains("spilling to system memory"));
        // Reused at another shape, or scaled, it is an extrapolation like any measurement.
        assert_eq!(
            IrisMemoryFloor::bound_from(spilled).provenance,
            FloorProvenance::Extrapolated
        );
        assert_eq!(
            spilled.for_run(2, 0).unwrap().provenance,
            FloorProvenance::Extrapolated
        );
        assert_eq!(spilled.for_run(1, 0), Some(spilled));
        let published = memory_floors_json(
            None,
            "Method.",
            [(
                "candle",
                vec!["full".to_owned(), "1024".to_owned()],
                spilled,
            )],
        );
        assert_eq!(
            published["provenance"]["candle"]["full"]["1024"],
            json!("measured_spill")
        );
        assert!(published["basis"]
            .as_str()
            .unwrap()
            .contains("did not fit the measuring device (completed only by spilling to system memory): candle full 1024."));
    }

    /// The published object lists every row under its own provenance, in the same nesting as the
    /// values, and its basis is generated from those rows.
    #[test]
    fn the_published_table_carries_per_row_provenance_and_a_generated_basis() {
        let published = memory_floors_json(
            Some("microBatchPixels"),
            "Method.",
            [
                (
                    "mlx",
                    vec!["full".to_owned(), "512".to_owned()],
                    IrisMemoryFloor::measured(Peak::Gib(78.5), 0),
                ),
                (
                    "mlx",
                    vec!["full".to_owned(), "1024".to_owned()],
                    IrisMemoryFloor::above_lower_bound(Peak::Gib(108.0), 128),
                ),
                (
                    "candle",
                    vec!["16384".to_owned()],
                    IrisMemoryFloor::declared(68),
                ),
            ],
        );
        assert_eq!(published["unit"], json!("GiB"));
        assert_eq!(published["key"], json!("microBatchPixels"));
        assert!(published.get("measured").is_none(), "no table-wide flag");
        assert_eq!(published["lanes"]["mlx"]["full"]["512"], json!(79));
        assert_eq!(
            published["provenance"]["mlx"]["full"]["512"],
            json!("measured")
        );
        assert_eq!(
            published["provenance"]["mlx"]["full"]["1024"],
            json!("lower_bound")
        );
        assert_eq!(
            published["provenance"]["candle"]["16384"],
            json!("declared")
        );
        let basis = published["basis"].as_str().unwrap();
        assert!(basis.contains("Floors are GiB"), "{basis}");
        assert!(
            basis.contains("Measured at the row's own shape: mlx full 512."),
            "{basis}"
        );
        assert!(
            basis.contains("Declared above a measured lower bound: mlx full 1024."),
            "{basis}"
        );
        assert!(
            basis.contains("Declared, not measured: candle 16384."),
            "{basis}"
        );
        assert!(!basis.contains("Extrapolated"), "{basis}");
    }
}
