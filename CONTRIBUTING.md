# Contributing

Thanks for taking an interest in JumpStar and CCERL. This project is meant to
make Chinese Checkers engine work easier to inspect, reproduce, challenge, and
improve.

Good contributions include:

- bug reports with reproduction steps
- rules or referee fixes with small test cases
- benchmark-runner improvements
- external-engine adapters
- documentation clarifications
- careful benchmark results that include commands, versions, seeds, and logs

Before opening a pull request:

1. Build and test locally:

   ```sh
   cmake -S . -B build -DCCZERO_USE_ACCELERATE=OFF
   cmake --build build --parallel
   ctest --test-dir build --output-on-failure
   ```

2. Keep changes focused. A rules fix, benchmark-data update, and UI polish pass
   should usually be separate pull requests.

3. Be explicit about benchmark claims. Include the ruleset, position suite,
   simulation/search settings, commit hash, and whether results are official,
   provisional, or self-reported.

4. Credit upstream projects and prior work. If you adapt an external engine,
   document the source repository, license, build settings, and rule differences.

## Benchmark Etiquette

CCERL is intended to make disagreement productive. Stronger engines are welcome.
So are bug reports that lower JumpStar's rating, expose a rule issue, or show a
better methodology. Please keep the discussion centered on reproducible evidence.

## Model Files

The repository records model labels and checksums, but does not store `.ccpv`
model weights in Git history. If you add or update model artifacts, prefer
GitHub Release assets or another explicit download channel, and document hashes.

