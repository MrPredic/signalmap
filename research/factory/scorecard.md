# Method Factory Scorecard — 2026-07-01

Honest multi-start (K=10) forecast + attractor-statistics rescue-metric. assay = cheap single-start pre-screen. nnz = sparsity (lower=more interpretable).

| system | method | nnz | assay | fc_mean | fc_min | attractor(wass,acf,bounded) | sec |
|---|---|---|---|---|---|---|---|
| lorenz | havok_linear | - | assay=-12.68 SKIP | - | - | - | 0.0 |
| lorenz | delay_sindy_th0.05 | 25 | assay=0.80 | -5.858 | -57.310 | 1.036,0.715,True | 12.1 |
| lorenz | delay_sindy_th0.5 | 8 | assay=0.80 | -6.253 | -61.242 | 1.271,0.727,True | 8.4 |
| rossler | havok_linear | - | assay=1.00 | 0.007 | -4.184 | 0.603,0.058,True | 0.1 |
| rossler | delay_sindy_th0.05 | 27 | assay=0.88 | 0.594 | -1.587 | 0.761,0.066,True | 52.7 |
| rossler | delay_sindy_th0.5 | 19 | assay=0.88 | 0.322 | -1.659 | 0.910,0.020,True | 3.9 |
| thomas | havok_linear | - | assay=-438.04 SKIP | - | - | - | 0.0 |
| thomas | delay_sindy_th0.05 | 26 | assay=-117325.69 SKIP | - | - | - | 0.1 |
| thomas | delay_sindy_th0.5 | 0 | assay=-3.35 SKIP | - | - | - | 0.0 |
