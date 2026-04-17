# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.2] - 2026-04-16

### Changed

- README **Real-World Insurance Use Cases**: rewritten as a flat 21-item list (dropped subgroup headings), professional and concise copy, expanded to cover bundled-episode intelligence (pharmacy-inclusive episode assignment, multi-attribution, episode cohort variance, trained definitions from claims history, provider attribution across mixed claim types) plus the existing payer-analytics and care-operations use cases.
- Refreshed **Use Case to Artifact Map** with one concrete artifact path or module per use case.

[0.3.2]: https://github.com/amolkodan/value-based-care-machine-learning-models/releases/tag/v0.3.2

## [0.3.1] - 2026-04-15

### Changed

- PyPI short **description** and README title/intro: *Open Source Healthcare ML Models for Cost Reduction and Outcome Improvement*; clarify **VBC Intelligence OS** positioning for cost and outcomes under value-based and bundled contracts.

[0.3.1]: https://github.com/amolkodan/value-based-care-machine-learning-models/releases/tag/v0.3.1

## [0.3.0] - 2026-04-15

### Added

- **ML bundled episode engine** (`carevalue_claims_ml.bundled_episode_engine`): learn episode code definitions from labeled bundles; train multi-label episode-family attribution from ICD/CPT/NDC-aware hashed features; score claims with **multi-attribution** (several episode families per line above a probability floor); materialize gap-based **episode instances** per member and family; attribute **rendering NPI** to episodes by allowed-amount weight.
- CLI: `episodes ml-learn-definitions`, `episodes ml-train`, `episodes ml-run`, `episodes ml-prep-training-from-gaps`.
- `vbc_intel_episodes` re-exports the new engine API.

[0.3.0]: https://github.com/amolkodan/value-based-care-machine-learning-models/releases/tag/v0.3.0

## [0.2.2] - 2026-04-09

### Changed

- Package metadata: set PyPI author to **Amol Kodan** (`[project].authors` in `pyproject.toml`).

[0.2.2]: https://github.com/amolkodan/value-based-care-machine-learning-models/releases/tag/v0.2.2

## [0.2.1] - 2026-04-09

### Changed

- Trim PyPI/GitHub README: remove the “Why this platform is differentiated” bullet list so the project description is shorter on [pypi.org](https://pypi.org/project/carevalue-claims-ml/).

[0.2.1]: https://github.com/amolkodan/value-based-care-machine-learning-models/releases/tag/v0.2.1

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
