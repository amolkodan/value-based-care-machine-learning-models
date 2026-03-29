## Model cards

Each trained model artifact should be accompanied by metadata describing:

- model name and version
- training cohort and feature window
- objective and target label
- metrics (global and subgroup)
- known limitations and intended use

### Intended use

- Population-level analytics for value-based care planning.
- Care-management prioritization experiments.

### Out-of-scope use

- Autonomous clinical diagnosis or treatment.
- Real-time bedside decision support.

### Risk management

- Inspect subgroup metrics before deployment-like use.
- Monitor calibration and drift over time.
- Keep a human-in-the-loop for intervention decisions.
