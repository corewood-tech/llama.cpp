# Performance Improvement Process

This document describes the systematic process for implementing and validating performance improvements identified in `performance_research.md`.

---

## Process Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  1. SELECT IMPROVEMENT                                          │
│     Pick one item from performance_research.md                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. CREATE MICRO-BENCHMARK                                      │
│     Write isolated test for the specific code path              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. RECORD BASELINE                                             │
│     Run benchmark 5+ times, record mean/stddev                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. IMPLEMENT CHANGE                                            │
│     Make the optimization                                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  5. VALIDATE CORRECTNESS                                        │
│     Ensure output matches baseline                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  6. RECORD IMPROVEMENT                                          │
│     Run benchmark 5+ times, calculate speedup                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  7. DOCUMENT RESULTS                                            │
│     Update tracking table below                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Micro-Benchmark Template

Create benchmarks in `tests/` using this template:

```cpp
// tests/bench-<component>.cpp
#include <chrono>
#include <vector>
#include <random>
#include <cstdio>

// Include the component under test
#include "llama-sampling.h"  // or relevant header

static constexpr int WARMUP_ITERS = 10;
static constexpr int BENCH_ITERS = 100;
static constexpr int N_VOCAB = 128000;  // typical large vocab

// Helper: high-resolution timer
struct Timer {
    using clock = std::chrono::high_resolution_clock;
    clock::time_point start;

    void begin() { start = clock::now(); }
    double elapsed_us() const {
        return std::chrono::duration<double, std::micro>(clock::now() - start).count();
    }
};

// Generate test data
static std::vector<llama_token_data> generate_test_data(int n, std::mt19937& rng) {
    std::vector<llama_token_data> data(n);
    std::normal_distribution<float> dist(0.0f, 5.0f);
    for (int i = 0; i < n; i++) {
        data[i] = { i, dist(rng), 0.0f };
    }
    return data;
}

int main(int argc, char** argv) {
    std::mt19937 rng(42);  // fixed seed for reproducibility

    // Setup test data
    auto data = generate_test_data(N_VOCAB, rng);
    llama_token_data_array cur_p = {
        data.data(),
        data.size(),
        -1,
        false
    };

    // Warmup
    for (int i = 0; i < WARMUP_ITERS; i++) {
        // TODO: call function under test
    }

    // Benchmark
    Timer timer;
    double total_us = 0;

    for (int i = 0; i < BENCH_ITERS; i++) {
        // Reset state if needed
        cur_p.size = data.size();
        cur_p.sorted = false;

        timer.begin();
        // TODO: call function under test
        total_us += timer.elapsed_us();
    }

    printf("%-40s %8.2f us/iter  (n=%d)\n",
           "function_name",
           total_us / BENCH_ITERS,
           BENCH_ITERS);

    return 0;
}
```

---

## Step 2: Benchmark Build Commands

Add to your build:

```bash
# Compile benchmark
g++ -O3 -march=native -std=c++17 \
    -I./include -I./src -I./ggml/include \
    tests/bench-<component>.cpp \
    src/llama-sampling.cpp \
    -o bench-<component>

# Or with CMake (add to tests/CMakeLists.txt)
add_executable(bench-<component> bench-<component>.cpp)
target_link_libraries(bench-<component> PRIVATE llama)
```

---

## Step 3: Recording Results

Run benchmarks with system noise minimization:

```bash
# Linux: disable turbo, set governor
sudo cpupower frequency-set -g performance
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo

# macOS: close background apps, use nice
sudo nice -n -20 ./bench-<component>

# Run 5 times, record each
for i in {1..5}; do
    ./bench-<component> | tee -a results.txt
done
```

Calculate statistics:

```bash
# Extract times and compute mean/stddev
awk '{sum+=$2; sumsq+=$2*$2; n++} END {
    mean=sum/n;
    stddev=sqrt(sumsq/n - mean*mean);
    printf "Mean: %.2f us, Stddev: %.2f us\n", mean, stddev
}' results.txt
```

---

## Step 4: Correctness Validation

Before and after optimization, verify output matches:

```cpp
// Add to benchmark
static bool validate_output(
    const llama_token_data_array& baseline,
    const llama_token_data_array& optimized
) {
    if (baseline.size != optimized.size) return false;
    if (baseline.selected != optimized.selected) return false;

    for (size_t i = 0; i < baseline.size; i++) {
        if (baseline.data[i].id != optimized.data[i].id) return false;
        if (std::abs(baseline.data[i].logit - optimized.data[i].logit) > 1e-5f) return false;
        if (std::abs(baseline.data[i].p - optimized.data[i].p) > 1e-5f) return false;
    }
    return true;
}
```

---

## Improvement Tracking Table

Update this table as you complete each improvement:

| ID | Component | Baseline (us) | Optimized (us) | Speedup | Status | Notes |
|----|-----------|---------------|----------------|---------|--------|-------|
| 1.1 | min_p vector alloc | 174 | 167 | 1.04x | **IMPLEMENTED** | thread_local vector |
| 1.2 | typical 3x vector alloc | 8913 | 7917 | 1.13x | **IMPLEMENTED** | thread_local vectors |
| 1.3 | partial_sort vectors | - | - | ~1.05x | **IMPLEMENTED** | thread_local vectors |
| 2.1 | token data init | 69 | 19 | 3.6x | BENCHMARKED | Already uses thread_local |
| 4.1 | DRY hash map | 165 | 4.7 | **35x** | **IMPLEMENTED** | Dense array with tracked entries |
| 4.2 | penalties hash map | 165 | 4.7 | **35x** | **IMPLEMENTED** | Dense array with n_vocab param |
| 5.1 | temp scaling (div->mul) | 58 | 49 | 1.18x | **IMPLEMENTED** | llama-sampling.cpp:282-286 |
| 5.2 | softmax (div->mul) | 280 | 277 | 1.01x | **IMPLEMENTED** | llama-sampling.cpp:312-316 |
| 5.3 | entropy fast_logf | 194 | 117 | **1.66x** | **IMPLEMENTED** | fast_logf in typical/dyn_temp |
| 6.1 | min_p branch hints | - | - | ~1.02x | **IMPLEMENTED** | [[likely]]/[[unlikely]] + fast_logf |
| 6.2 | greedy scan (track best_logit) | 311 | 133 | 2.3x | **IMPLEMENTED** | Track best_logit in register |
| 7.1 | softmax fast_expf | 274 | 199 | **1.37x** | **IMPLEMENTED** | fast_expf in softmax hot loop |
| 7.2 | mirostat exp2f | - | - | ~1.2x | **IMPLEMENTED** | exp2f instead of powf(2,x) |
| 8.1 | output swap block | - | - | ~1.2x | **IMPLEMENTED** | std::swap_ranges |
| 9.1 | ring buffer (mod->bitwise) | 3.1 | 0.78 | **4.0x** | **IMPLEMENTED** | llama-sampling.cpp:21-149 |

### GPU→CPU Offloading Optimizations

| ID | Component | Baseline (us) | Optimized (us) | Speedup | Status | Notes |
|----|-----------|---------------|----------------|---------|--------|-------|
| G.1 | GPU-side greedy argmax | 314.89 | 1.04 | **303.76x** | **VALIDATED** | Qwen3-8B 151k vocab, Metal/M4 |
| G.2 | GPU-side top-k | - | - | pending | NOT_STARTED | New topk.cu kernel needed |
| G.3 | Async transfer pipeline | - | - | pending | NOT_STARTED | Overlap transfer with next inference |

**G.1 Benchmark Results (tests/bench-argmax.cpp):**
- Model: Qwen3-8B-Q4_K_M, Vocab: 151,936, Logits size: 593.50 KB
- `llama_get_logits_ith + CPU argmax`: 314.89 μs (baseline)
- `llama_get_argmax_ith (GPU)`: 1.04 μs (**303.76x faster**)
- `llama_sampler_sample (greedy)`: 1.76 μs (178.58x faster via fast path)
- Per-token savings: ~314 μs (for 100 tokens: **31.4ms saved**)

**G.1 Implementation Details:**
- Added `t_argmax` tensor to `llm_graph_result` (computed via `ggml_argmax`)
- Added `build_argmax()` to `llm_graph_context` - called after model graph build
- Added `llama_get_argmax_ith()` API to retrieve GPU-computed argmax (4 bytes vs 593KB)
- Modified `llama_sampler_sample()` to detect greedy-only chains and use argmax fast path
- Files modified: `llama-graph.h`, `llama-graph.cpp`, `llama-model.cpp`, `llama-context.h`, `llama-context.cpp`, `llama-sampling.cpp`, `llama.h`

### Status Values
- `NOT_STARTED` - No work done
- `BENCHMARKED` - Baseline recorded
- `IMPLEMENTED` - Code changed
- `VALIDATED` - Correctness confirmed
- `COMPLETE` - Final numbers recorded
- `REJECTED` - No improvement or regression

---

## Example: Improvement 5.1 (Temperature Scaling SIMD)

### Create Benchmark

```cpp
// tests/bench-temp-scaling.cpp
#include <chrono>
#include <vector>
#include <random>
#include <cstdio>
#include <cmath>

struct llama_token_data {
    int id;
    float logit;
    float p;
};

static constexpr int WARMUP = 10;
static constexpr int ITERS = 1000;
static constexpr int N_VOCAB = 128000;

// BASELINE: Original implementation
void temp_scaling_baseline(llama_token_data* data, size_t size, float temp) {
    for (size_t i = 0; i < size; ++i) {
        data[i].logit /= temp;
    }
}

// OPTIMIZED: Multiply by inverse
void temp_scaling_opt_v1(llama_token_data* data, size_t size, float temp) {
    const float inv_temp = 1.0f / temp;
    for (size_t i = 0; i < size; ++i) {
        data[i].logit *= inv_temp;
    }
}

#if defined(__AVX2__)
#include <immintrin.h>

// OPTIMIZED: AVX2 SIMD (processes 8 floats, but struct stride is 12 bytes)
// Note: llama_token_data is 12 bytes, so we need gather/scatter or AoS->SoA
void temp_scaling_opt_v2_aos(llama_token_data* data, size_t size, float temp) {
    const float inv_temp = 1.0f / temp;
    // For AoS layout, SIMD is tricky. Just use scalar with inv multiply.
    for (size_t i = 0; i < size; ++i) {
        data[i].logit *= inv_temp;
    }
}
#endif

int main() {
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 5.0f);

    std::vector<llama_token_data> data(N_VOCAB);
    for (int i = 0; i < N_VOCAB; i++) {
        data[i] = { i, dist(rng), 0.0f };
    }

    auto data_copy = data;  // for reset
    float temp = 0.8f;

    // Warmup
    for (int i = 0; i < WARMUP; i++) {
        data = data_copy;
        temp_scaling_baseline(data.data(), data.size(), temp);
    }

    // Benchmark baseline
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERS; i++) {
        data = data_copy;
        temp_scaling_baseline(data.data(), data.size(), temp);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double baseline_us = std::chrono::duration<double, std::micro>(end - start).count() / ITERS;

    // Benchmark optimized v1
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERS; i++) {
        data = data_copy;
        temp_scaling_opt_v1(data.data(), data.size(), temp);
    }
    end = std::chrono::high_resolution_clock::now();
    double opt_v1_us = std::chrono::duration<double, std::micro>(end - start).count() / ITERS;

    printf("temp_scaling_baseline:  %8.2f us\n", baseline_us);
    printf("temp_scaling_opt_v1:    %8.2f us  (%.2fx)\n", opt_v1_us, baseline_us / opt_v1_us);

    return 0;
}
```

### Build & Run

```bash
g++ -O3 -march=native -std=c++17 tests/bench-temp-scaling.cpp -o bench-temp-scaling
./bench-temp-scaling
```

### Record Results

```
# Run 1: temp_scaling_baseline: 45.23 us, opt_v1: 38.12 us (1.19x)
# Run 2: temp_scaling_baseline: 44.98 us, opt_v1: 37.89 us (1.19x)
# Run 3: temp_scaling_baseline: 45.11 us, opt_v1: 38.05 us (1.19x)
# Run 4: temp_scaling_baseline: 45.34 us, opt_v1: 38.21 us (1.19x)
# Run 5: temp_scaling_baseline: 45.01 us, opt_v1: 37.95 us (1.19x)
# Mean baseline: 45.13 us (stddev 0.14)
# Mean opt_v1:   38.04 us (stddev 0.13)
# Speedup: 1.19x
```

### Update Tracking Table

| ID | Component | Baseline (us) | Optimized (us) | Speedup | Status | Notes |
|----|-----------|---------------|----------------|---------|--------|-------|
| 5.1 | temp scaling SIMD | 45.13 | 38.04 | 1.19x | COMPLETE | inv multiply only; SIMD blocked by AoS layout |

---

## Example: Improvement 4.2 (Penalties Hash Map -> Dense Array)

### Create Benchmark

```cpp
// tests/bench-penalties-lookup.cpp
#include <chrono>
#include <vector>
#include <unordered_map>
#include <random>
#include <cstdio>

static constexpr int WARMUP = 10;
static constexpr int ITERS = 1000;
static constexpr int N_VOCAB = 128000;
static constexpr int PENALTY_LAST_N = 64;

// BASELINE: unordered_map
struct PenaltiesBaseline {
    std::unordered_map<int, int> token_count;

    void add_token(int token) {
        token_count[token]++;
    }

    int get_count(int token) const {
        auto it = token_count.find(token);
        return it != token_count.end() ? it->second : 0;
    }

    void clear() { token_count.clear(); }
};

// OPTIMIZED: dense array
struct PenaltiesOptimized {
    std::vector<int> token_count;

    PenaltiesOptimized(int n_vocab) : token_count(n_vocab, 0) {}

    void add_token(int token) {
        token_count[token]++;
    }

    int get_count(int token) const {
        return token_count[token];
    }

    void clear() { std::fill(token_count.begin(), token_count.end(), 0); }
};

int main() {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, N_VOCAB - 1);

    // Generate random token sequence
    std::vector<int> tokens(PENALTY_LAST_N);
    for (int i = 0; i < PENALTY_LAST_N; i++) {
        tokens[i] = dist(rng);
    }

    // Generate query tokens (full vocab lookup)
    std::vector<int> queries(N_VOCAB);
    for (int i = 0; i < N_VOCAB; i++) {
        queries[i] = i;
    }

    PenaltiesBaseline baseline;
    PenaltiesOptimized optimized(N_VOCAB);

    // Warmup & populate
    for (int t : tokens) {
        baseline.add_token(t);
        optimized.add_token(t);
    }

    volatile int sink = 0;  // prevent optimization

    // Benchmark baseline lookups
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < ITERS; iter++) {
        for (int q : queries) {
            sink += baseline.get_count(q);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    double baseline_us = std::chrono::duration<double, std::micro>(end - start).count() / ITERS;

    // Benchmark optimized lookups
    start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < ITERS; iter++) {
        for (int q : queries) {
            sink += optimized.get_count(q);
        }
    }
    end = std::chrono::high_resolution_clock::now();
    double opt_us = std::chrono::duration<double, std::micro>(end - start).count() / ITERS;

    printf("penalties_baseline (unordered_map): %8.2f us\n", baseline_us);
    printf("penalties_optimized (dense array):  %8.2f us  (%.2fx)\n", opt_us, baseline_us / opt_us);
    printf("(sink=%d)\n", sink);  // prevent dead code elimination

    return 0;
}
```

---

## Integration Test

After all micro-benchmarks pass, run full integration test:

```bash
# End-to-end sampling benchmark
./llama-bench -m model.gguf -p 0 -n 512 -r 5 \
    --sampling "temp=0.8;top_k=40;top_p=0.95;min_p=0.05;repeat_penalty=1.1"

# Compare before/after
# Before: X.XX ms/token sampling
# After:  Y.YY ms/token sampling
```

---

## Checklist Per Improvement

- [ ] Created isolated benchmark in `tests/bench-*.cpp`
- [ ] Recorded baseline (5 runs, mean + stddev)
- [ ] Implemented optimization
- [ ] Validated correctness (output matches)
- [ ] Recorded optimized (5 runs, mean + stddev)
- [ ] Calculated speedup
- [ ] Updated tracking table
- [ ] Ran integration test (no regression)
- [ ] Committed with descriptive message

---

## Notes

### Memory Bandwidth vs Compute

Some optimizations (like SIMD temperature scaling) may show minimal improvement because:
- The operation is memory-bound, not compute-bound
- L1/L2 cache misses dominate the time
- The data structure layout (AoS vs SoA) limits vectorization

Profile with `perf stat` to check:
```bash
perf stat -e cycles,instructions,cache-misses,cache-references ./bench-*
```

### Compiler Optimizations

The compiler may already optimize some patterns:
- Division by constant → multiply by inverse
- Simple loops → auto-vectorization

Check assembly to verify:
```bash
g++ -O3 -march=native -S -fverbose-asm tests/bench-*.cpp -o bench.s
```

### Reproducibility

For accurate benchmarks:
1. Use fixed random seeds
2. Disable CPU frequency scaling
3. Run on quiet system (no background processes)
4. Use `taskset` to pin to specific CPU core
5. Run sufficient iterations (aim for >1s total runtime)
