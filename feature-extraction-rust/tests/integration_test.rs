//! Integration tests for the AI4Pain 2026 feature extractor.
//!
//! End-to-end smoke test that exercises Catch22, Entropy, and Stats on a
//! synthetic signal and confirms the full 40-feature pipeline runs without
//! panicking. Verifies the Phase 1 fork of BP inference compiles and that
//! the public API surface is reachable from outside the crate.

use ai4pain_feature_extraction::catch22::Catch22Features;
use ai4pain_feature_extraction::entropy::EntropyFeatures;
use ai4pain_feature_extraction::stats::StatFeatures;
use ai4pain_feature_extraction::types::{FeatureFramework, SignalType};

/// Build a deterministic synthetic signal that has nonzero variance in every
/// feature space (autocorrelation, distribution shape, fluctuation, etc.).
fn synth_signal(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let t = i as f64 / 64.0;
            (2.0 * std::f64::consts::PI * 1.2 * t).sin()
                + 0.5 * (2.0 * std::f64::consts::PI * 0.3 * t).cos()
                + 0.1 * (i as f64 * 0.07).sin()
        })
        .collect()
}

#[test]
fn signal_type_round_trip() {
    for s in &["bvp", "eda", "resp", "spo2"] {
        let st = SignalType::from_str(s).expect("known signal type");
        assert_eq!(st.as_str(), *s);
    }
    assert!(SignalType::from_str("bogus").is_none());
}

#[test]
fn feature_framework_round_trip() {
    for s in &["catch22", "entropy", "stats"] {
        let f = FeatureFramework::from_str(s).expect("known framework");
        assert_eq!(f.as_str(), *s);
    }
}

#[test]
fn end_to_end_pipeline_on_synthetic_signal() {
    let signal = synth_signal(1024);

    // Catch22
    let c22 = Catch22Features::compute(&signal).expect("catch22 should compute");
    assert!(c22.dn_histogram_mode_5.is_finite());
    assert!(c22.co_f1ecac.is_finite());

    // Entropy with default ordinal-pattern parameters
    let ent = EntropyFeatures::calculate(&signal, 7, 2).expect("entropy should compute");
    assert!(ent.permutation_entropy.is_finite());
    assert!(ent.sample_entropy.is_finite());

    // Stats
    let stats = StatFeatures::compute(&signal);
    assert!(stats.std > 0.0, "synthetic signal should have nonzero variance");
    assert!(stats.max > stats.min);
}
