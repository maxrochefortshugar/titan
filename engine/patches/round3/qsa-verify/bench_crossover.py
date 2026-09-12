import sys
sys.argv=[sys.argv[0]]
exec(open("bench_arms.py").read().split("print(f\"{'ctx'")[0])
print(f"{'ctx':>8} {'M':>3} | {'sparse':>8} {'dense':>8} | winner")
for N in (4096, 8192, 16384, 32768, 65536):
    for M in (1, 4):
        r = run(N, M)
        w = "sparse" if r['sparse'] < r['dense_causal'] else "DENSE"
        print(f"{N:>8} {M:>3} | {r['sparse']:>8.3f} {r['dense_causal']:>8.3f} | {w}")
