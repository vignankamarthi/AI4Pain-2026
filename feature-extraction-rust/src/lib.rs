//! AI4Pain 2026 feature extraction library.
//!
//! Forked from `Blood-Pressure-Inference-with-BVP/Feature-Extraction-Rust-Complete`
//! during Phase 1 of the scaffolding plan (HIP-2 cleared 2026-04-10: fork
//! strategy chosen for self-containment). Adapted for the four AI4Pain 2026
//! signals (BVP, EDA, RESP, SpO2).
//!
//! The 22 + 10 + 8 feature implementations are signal-agnostic and were copied
//! verbatim from the BP inference reference. The only AI4Pain-specific code
//! lives in `types.rs` (SignalType enum) and `main.rs` (CLI subcommands and
//! per-signal output tagging).

pub mod catch22;
pub mod entropy;
pub mod stats;
pub mod signal_processing;
pub mod data_loader;
pub mod types;
pub mod utils;

pub use types::SignalType;
