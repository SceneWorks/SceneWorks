//! Backend routing / gating / catalog logic split out of the `jobs_store` god module
//! (sc-8816). This is a pure code move: the SQLite jobs/workers store and the SQL-coupled
//! dispatch stay in `jobs_store.rs`, while the backend-eligibility predicates, the Mac
//! support/capability probes, the routed-model/kernel catalog, and the gap classifiers live
//! here. No routing decision, catalog membership, or public API changed.

pub(crate) mod candle;
pub(crate) mod catalog;
pub(crate) mod gaps;
pub(crate) mod mlx;
