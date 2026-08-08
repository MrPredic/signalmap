# DCASE valve id_00 external-user readout (FALLBACK minimal spec)

NOT the distilled spec (distill LOGO infeasible in 30-min cap).
spec programs: ['acf1(id(id(x)))', 'crest(id(id(x)))', 'lcross(id(id(x)))', 'meanabs(id(id(x)))', 'peakcv(id(id(x)))', 'runcv(id(id(x)))', 'runmean(id(id(x)))', 'std(id(id(x)))', 'zcr(id(id(x)))']
spec premium: [] (none)

AUC = 0.2642, 95% CI [0.1929, 0.3360] (bootstrap n=2000 over 204 held-out clips, seed=0)
TPR@FPR=0.1 = 0.0481
threshold = 92.0089
n_clips=204 (anomaly=104, normal=100)
