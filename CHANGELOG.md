# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-04-09

### Added

- `carevalue_claims_ml.journey_signals`: merge professional/institutional and pharmacy claim lines, member-month utilization features, NDC polypharmacy proxy, CPT/HCPCS intensity, ICD-10 morbidity breadth.
- Episode archetypes (`general`, `orthopedic`, `cardiac`, `maternity`, `oncology`) and richer `score_episode_risk` (clinical/procedural breadth, financial intensity, severity percentile).
- CLI: `journey merge`, `journey monthly-features`; `episodes score` accepts optional diagnosis/procedure columns.
- `vbc_intel_core` re-exports journey helpers; `vbc_intel_episodes` exports `EPISODE_ARCHETYPES`.

### Changed

- Package description and keywords emphasize bundled episodes, patient journey, and pharmacy integration.

[0.2.0]: https://github.com/amolkodan/value-based-care-machine-learning-models/releases/tag/v0.2.0

## [0.1.0] - 2026-04-09

### Added

- Initial PyPI-oriented packaging for `carevalue-claims-ml`.
- **VBC Intelligence OS** sublibrary namespaces: `vbc_intel_core`, `vbc_intel_episodes`, `vbc_intel_policy`, `vbc_intel_benchmarks`, `vbc_intel_careops`.
- Bundled episode helpers in `carevalue_claims_ml.episodes`.
- CLI entrypoints `carevalue-ml` and `vbc-intel`; commands `libraries`, `episodes`, `benchmarks`, `careops`.
- Project metadata: license file reference, classifiers, keywords, repository URLs.

[0.1.0]: https://github.com/amolkodan/value-based-care-machine-learning-models/releases/tag/v0.1.0
