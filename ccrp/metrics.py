def reduction_pct(baseline, treated):
    if baseline == 0.0:
        return 0.0
    return (baseline - treated) / baseline * 100.0
