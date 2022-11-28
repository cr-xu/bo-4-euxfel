# Beamtime 2022-11-24

## Measurements

- Ran BO with different hyperparameter settings: found proper lengthscale / proximal weighting
- Benchmarked BO performance with
  - Fix initial setting with slightly detuned air coils
  - For each undulator section, used a pair of horizontal and vertical air coil correctors
  - Increased from 2 to 12 dimensional problem, i.e. from 1 to 6 undulator sections distributed along the SASE1 beam line.
- Reduced the number of shots for averaging from 30 to 5, noisy but still working most of the time

## Beamtime Planning

Draft before 2022.11.24

- __Task 0__: get the environment working and fix any possible bug
- Define a (_not optimized_, _reproducible_) initial working condition
  - Save initial condition
  - Set back to this initial condition every time for fair comparison

### Task 1: SASE1 tuning with air coils

1. Apply BO for ~~first~~ _randomly selected_ 4 correctors, find proper settings
   1. Test BO with UCB, EI
   2. Test BO with different __proximal step sizes__
      1. 3 different proximal lengthscales from short to long
      2. __hard vs. proximal__ step size limits
   3. Test BO with __default settings__ vs. __explicit measured noise__
2. Increase the number of input parameters
   1. to 6 air coils
   2. to 8 air coils
   3. increase until failure?
3. Benchmark the BO performance against simplex (maybe for 6 parameters)

### Task 2: SASE1 tuning with Quadrupoles (optional to do task 3 first)

1. Use previously found BO settings, apply for `./conf/SASE1_matching_quads.json`

### Task 3: SASE2 tuning with air coils

If SASE2 is available

1. Use previously found BO settings, repeat step 2 in __Task 1__
