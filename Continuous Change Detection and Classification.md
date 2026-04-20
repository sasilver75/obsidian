---
aliases:
  - CCDC
---
A time-series algorithm that fits harmonic (sinusoidal) models to every pixel's spectral history.

Takes a full [[Landsat]] time-series for a pixel (decades of observation) and fits harmonic curves to each spectral band. Now you can monitor residuals; when observations deviate significantly from the model, you say a change has been detected. After change, fit a new model to the new stable period.

What you get per pixel:
- Harmonic coefficients (amplitude, phase, RMSE) for each band
- Change dates and magnitudes
- Land cover classification at any point in time

Strengths:
- Captures seasonal phenology explicitly
- Detects abrupt changes (deforestation, fire, urban conversion) precisely in time
- Dense temporal features even from sparse, cloudy observations

Weaknesses:
- Designed around Landsat cadence specifically
- Purely spectral, no spatial context
- Hand-engineered, not learned


