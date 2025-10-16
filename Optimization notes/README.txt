Optimization Recommendations
============================

1. Parallelize STDF ingestion when multiple files are present (ProcessPoolExecutor), assuming the reader is process-safe.
2. Vectorize outlier filtering using pandas groupby and vectorized math instead of per-group Python loops.
3. Keep the new itertuples-based measurement export; avoid reintroducing per-cell formatting loops.
4. Chart generation speed-ups:
   - Precompute heavy math (histogram bins, CDF arrays) in pandas before rendering.
   - Reuse Matplotlib figure/axes objects where possible and disable non-essential elements.
   - Optionally parallelize rendering across tests with ProcessPoolExecutor.
   - Allow users to skip or selectively generate charts to reduce load.
   - Consider caching generated PNGs across runs when inputs and limits are unchanged.
5. Monitor remaining iterrows() usages (summary/limits) in case future changes expand those tables dramatically.
6. Long-term options (execute only if profiling justifies):
   - Streaming Parquet writes or Arrow-native computation for memory-heavy runs.
   - Cythonizing compute-intensive routines (stats/outliers) only after Python-level optimizations.
   - Alternative renderers for charts, or storing raw data and letting Excel plot on demand.
