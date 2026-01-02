# llama.cpp Hot Path Performance Research

This document identifies potential performance enhancements in the llama.cpp codebase, focusing on the inference hot path. Each finding includes the location, issue description, proposed fix, and expected impact.

---

## Table of Contents

1. [Sampling Hot Path Issues](#1-sampling-hot-path-issues)
2. [Memory Allocation Issues](#2-memory-allocation-issues)
3. [Virtual Function / Indirection Overhead](#3-virtual-function--indirection-overhead)
4. [Hash Map Operations in Hot Loops](#4-hash-map-operations-in-hot-loops)
5. [SIMD Vectorization Opportunities](#5-simd-vectorization-opportunities)
6. [Branch Prediction Issues](#6-branch-prediction-issues)
7. [Transcendental Function Overhead](#7-transcendental-function-overhead)
8. [Output Processing Inefficiencies](#8-output-processing-inefficiencies)
9. [Ring Buffer Modulo Operations](#9-ring-buffer-modulo-operations)
10. [std::discrete_distribution Overhead](#10-stddiscrete_distribution-overhead)

---

## 1. Sampling Hot Path Issues

### 1.1 Temporary Vector in min_p Sampler

**Location:** `src/llama-sampling.cpp:896-908`

**Issue:** The `llama_sampler_min_p_apply` function creates a temporary `std::vector<llama_token_data>` on every call without reserving capacity:

```cpp
std::vector<llama_token_data> filtered_tokens;
// ... loop pushes back elements
for (size_t i = 0; i < cur_p->size; ++i) {
    if (cur_p->data[i].logit >= min_logit) {
        filtered_tokens.push_back(cur_p->data[i]);
    }
}
```

**Fix:**
- Option A: Use `reserve(cur_p->size)` before the loop
- Option B: Use a thread_local vector (like `llama_sampler_sample` does at line 433)
- Option C: Filter in-place by swapping elements to the front

**Expected Impact:** Medium. Eliminates repeated heap allocations (potentially O(log n) reallocations per sample). For vocab size ~128k, this could save 1-3 allocations per min_p application.

---

### 1.2 Multiple Temporary Vectors in typical Sampler

**Location:** `src/llama-sampling.cpp:996-1034`

**Issue:** The `llama_sampler_typical_apply` function creates THREE temporary vectors per call:

```cpp
std::vector<float> shifted_scores;           // line 996
std::vector<size_t> indices(cur_p->size);    // line 1003
std::vector<llama_token_data> cur_p_new;     // line 1026
```

None use `reserve()`, and all are recreated on every sample.

**Fix:**
- Use thread_local vectors for `shifted_scores` and `cur_p_new`
- Add `shifted_scores.reserve(cur_p->size)` and `cur_p_new.reserve(last_idx)`
- Consider combining the entropy calculation and score computation loops

**Expected Impact:** High. This is a triple allocation penalty per typical sample. Could reduce sampling time by 10-20% when typical sampling is used.

---

### 1.3 Temporary Vector in partial_sort

**Location:** `src/llama-sampling.cpp:144-147, 205`

**Issue:** The `llama_token_data_array_partial_sort` function creates multiple vectors:

```cpp
std::vector<int> bucket_idx;
std::vector<int> histo(nbuckets, 0);
std::vector<llama_token_data*> bucket_ptrs;
```

And `llama_token_data_array_partial_sort_inplace` creates:
```cpp
std::vector<llama_token_data> tmp;
```

**Fix:**
- Use thread_local vectors with `clear()` + `resize()`/`reserve()`
- Pre-allocate the histogram as a static array (nbuckets=128 is fixed)

**Expected Impact:** Medium. These functions are called frequently during top-k/top-p filtering.

---

## 2. Memory Allocation Issues

### 2.1 Token Data Array Initialization

**Location:** `src/llama-sampling.cpp:433-437`

**Issue:** While the vector itself is thread_local (good!), the initialization loop is O(n_vocab) and sequential:

```cpp
static thread_local std::vector<llama_token_data> cur;
cur.resize(n_vocab);
for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
    cur[token_id] = llama_token_data{token_id, logits[token_id], 0.0f};
}
```

**Fix:**
- Vectorize this loop with SIMD (the struct is 12 bytes, but the id and logit can be parallelized)
- Consider using parallel copy for the logits followed by a single pass for token_id initialization
- Use `memcpy` for the logits portion if struct layout permits

**Expected Impact:** Medium-Low. This is O(n_vocab) per sample, but memory bandwidth may be the bottleneck rather than computation.

---

### 2.2 Sampler Chain Uses Dynamic Allocation for Each Sampler

**Location:** `src/llama-sampling.cpp:350, 527, 727, etc.`

**Issue:** Every sampler is heap-allocated with `new`:

```cpp
return new llama_sampler { ... };
// and
new llama_sampler_chain { ... }
new llama_sampler_dist { ... }
// etc.
```

**Fix:**
- Consider using a memory pool or arena allocator for samplers
- For the common case of a fixed sampler chain, pre-allocate the samplers contiguously
- Use placement new with a pre-allocated buffer

**Expected Impact:** Low during inference (samplers are created once), but could matter for applications that frequently reconfigure samplers.

---

## 3. Virtual Function / Indirection Overhead

### 3.1 Sampler Interface Uses Function Pointer Table

**Location:** `src/llama-sampling.cpp:374-381, 475-482`

**Issue:** Every sampler apply call goes through an indirect function pointer:

```cpp
void llama_sampler_apply(struct llama_sampler * smpl, struct llama_token_data_array * cur_p) {
    GGML_ASSERT(smpl->iface->apply);
    smpl->iface->apply(smpl, cur_p);  // indirect call
}

static void llama_sampler_chain_apply(...) {
    for (auto * smpl : chain->samplers) {
        llama_sampler_apply(smpl, cur_p);  // indirect call per sampler
    }
}
```

**Fix:**
- For hot paths, consider a "compiled" sampler chain that inlines common sampler combinations
- Use template-based static polymorphism for common sampler configurations
- Profile-guided optimization (PGO) can help the CPU predict these branches

**Expected Impact:** Medium. Each indirect call costs ~10-30 cycles for misprediction, and the chain typically has 3-5 samplers. Could save 50-150 cycles per sample.

---

### 3.2 Memory Interface Virtual Calls

**Location:** `src/llama-memory.h:51-61, 81-119`

**Issue:** The `llama_memory_i` interface uses virtual functions for memory operations:

```cpp
virtual bool next() = 0;
virtual bool apply() = 0;
virtual const llama_ubatch & get_ubatch() const = 0;
virtual llama_memory_status get_status() const = 0;
// ... many more
```

**Fix:**
- For the most common KV cache implementation, consider providing non-virtual fast-paths
- Use CRTP (Curiously Recurring Template Pattern) for compile-time polymorphism where possible

**Expected Impact:** Low-Medium. These are called per-ubatch, not per-token, so the overhead is amortized.

---

## 4. Hash Map Operations in Hot Loops

### 4.1 DRY Sampler Uses std::unordered_map in Hot Loop

**Location:** `src/llama-sampling.cpp:2170-2173, 2187-2210`

**Issue:** The DRY sampler does hash map lookups in O(n_vocab) loops:

```cpp
// Line 2170-2173: Per-repeat lookup and insert
const auto& it = ctx->dry_max_token_repeat.find(token);
if (it == ctx->dry_max_token_repeat.end() || it->second < repeat_len) {
    ctx->dry_max_token_repeat[token] = repeat_len;
}

// Line 2187-2210: Per-token lookup
for (size_t i = 0; i < cur_p->size; ++i) {
    const auto& af_kvp = ctx->dry_max_token_repeat.find(cur_p->data[i].id);
    if (af_kvp != ctx->dry_max_token_repeat.end()) {
        // ... also does equal_range on dry_processed_breakers
    }
}
```

**Fix:**
- Pre-size the hash map with `reserve()` based on expected token count
- Consider using a flat_hash_map (e.g., abseil's) which has better cache locality
- For small token sets, a sorted vector with binary search may be faster
- The inner `equal_range` on `dry_processed_breakers` (unordered_multimap) is particularly expensive

**Expected Impact:** High for DRY sampler users. Hash operations are ~50-100ns each; with n_vocab iterations, this can add milliseconds.

---

### 4.2 Penalties Sampler Token Counting

**Location:** `src/llama-sampling.cpp:1727, 1751-1758`

**Issue:** Uses `std::unordered_map<llama_token, int>` for frequency counting:

```cpp
std::unordered_map<llama_token, int> token_count;
// ...
ctx->prev.push_back(token);
```

**Fix:**
- Use a dense array (since token IDs are bounded by vocab size) instead of hash map
- `std::vector<int> token_count(n_vocab, 0)` with direct indexing: `token_count[token]++`

**Expected Impact:** High. Direct array access is O(1) with no hashing overhead, compared to O(1) amortized with hash computation and potential collisions.

---

## 5. SIMD Vectorization Opportunities

### 5.1 Temperature Scaling Loop

**Location:** `src/llama-sampling.cpp:282-284`

**Issue:** Simple division loop that could be vectorized:

```cpp
for (size_t i = 0; i < cur_p->size; ++i) {
    cur_p->data[i].logit /= temp;
}
```

**Fix:**
- Use SIMD intrinsics (AVX/AVX2/AVX-512 or NEON)
- Compute `1/temp` once and multiply (faster than repeated division)
- Process 4-8 logits per iteration with vector operations

```cpp
const float inv_temp = 1.0f / temp;
// SIMD: process 8 floats at once with AVX
```

**Expected Impact:** Medium. 4-8x speedup for this loop, but it's a small part of total sampling time.

---

### 5.2 Softmax Implementation

**Location:** `src/llama-sampling.cpp:295-312`

**Issue:** Sequential loops for max-finding, exp, and normalization:

```cpp
// Max finding (lines 296-300)
for (size_t i = 1; i < cur_p->size; ++i) {
    max_l = std::max(max_l, cur_p->data[i].logit);
}

// Exp and sum (lines 304-308)
for (size_t i = 0; i < cur_p->size; ++i) {
    float p = expf(cur_p->data[i].logit - max_l);
    cur_p->data[i].p = p;
    cum_sum += p;
}

// Normalize (lines 310-312)
for (size_t i = 0; i < cur_p->size; ++i) {
    cur_p->data[i].p /= cum_sum;
}
```

**Fix:**
- Use SIMD for horizontal max reduction
- Use vectorized `expf` (available in Intel MKL, AMD AOCL, or hand-rolled with polynomial approximation)
- Combine the normalization pass with the next operation when possible
- Consider fused exp-sum operation

**Expected Impact:** High. Softmax is called frequently and operates on O(n_vocab) elements. SIMD expf can be 4-8x faster.

---

### 5.3 Entropy Calculation

**Location:** `src/llama-sampling.cpp:990-993`

**Issue:** Sequential loop with transcendental functions:

```cpp
float entropy = 0.0f;
for (size_t i = 0; i < cur_p->size; ++i) {
    entropy += -cur_p->data[i].p * logf(cur_p->data[i].p);
}
```

**Fix:**
- Use SIMD vectorized `logf`
- Consider lookup tables for log approximation
- Use Kahan summation if precision is critical (entropy accumulation)

**Expected Impact:** Medium. Entropy is calculated in typical sampling and dynamic temperature.

---

## 6. Branch Prediction Issues

### 6.1 min_p Sampler Fallback Path

**Location:** `src/llama-sampling.cpp:895-936`

**Issue:** The min_p sampler has a two-path structure that can cause branch misprediction:

```cpp
if (!cur_p->sorted) {
    // Unsorted path - creates vector, filters
    // ...
    if (!filtered_tokens.empty() && filtered_tokens.size() >= ctx->min_keep) {
        // Success path
        min_p_applied = true;
    }
}

if (!min_p_applied) {
    // Fallback to sorted path
}
```

**Fix:**
- Profile to determine which path is more common
- Use `[[likely]]` / `[[unlikely]]` attributes (C++20)
- Consider always using the sorted path if it's competitive

**Expected Impact:** Low-Medium. Branch misprediction costs 10-30 cycles.

---

### 6.2 Greedy Sampler Linear Search

**Location:** `src/llama-sampling.cpp:580-587`

**Issue:** Linear scan for max logit:

```cpp
static void llama_sampler_greedy_apply(...) {
    cur_p->selected = 0;
    for (size_t i = 1; i < cur_p->size; ++i) {
        if (cur_p->data[i].logit > cur_p->data[cur_p->selected].logit) {
            cur_p->selected = i;  // Unpredictable branch
        }
    }
}
```

**Fix:**
- Use SIMD horizontal max with index tracking
- Process data in chunks and find local maxima first
- Use conditional move instructions instead of branching

**Expected Impact:** Medium. This loop runs O(n_vocab) times and the branch is data-dependent, causing frequent mispredictions.

---

## 7. Transcendental Function Overhead

### 7.1 Multiple logf/expf Calls

**Location:** `src/llama-sampling.cpp` (multiple locations)

**Issue:** Many transcendental function calls:
- Line 305: `expf()` in softmax (O(n_vocab) calls)
- Line 645: `expf()` in dist sampler
- Lines 992, 998: `logf()` in typical sampler
- Lines 1351-1352: `logf()` in mirostat
- Line 2207: `std::pow()` in DRY penalty

**Fix:**
- Use fast math approximations where precision allows (e.g., fast_exp, fast_log)
- Enable `-ffast-math` for sampling code (carefully, as it affects precision)
- Use lookup tables with interpolation for log/exp
- Use vectorized versions (Intel SVML, AMD LibM, or custom)

**Expected Impact:** High. Transcendental functions are 10-100x slower than basic arithmetic. For n_vocab=128k, this adds up significantly.

---

### 7.2 Repeated pow() Calls in Mirostat

**Location:** `src/llama-sampling.cpp:1360`

**Issue:** Multiple `powf()` calls in a single expression:

```cpp
float k = powf((epsilon_hat * powf(2, ctx->mu)) / (1 - powf(ctx->n_vocab, -epsilon_hat)), 1 / s_hat);
```

**Fix:**
- Cache repeated subexpressions
- Use `exp2f(ctx->mu)` instead of `powf(2, ctx->mu)` (faster)
- Precompute constants where possible

**Expected Impact:** Low (mirostat is one sample), but easy optimization.

---

## 8. Output Processing Inefficiencies

### 8.1 Element-by-Element Output Swap

**Location:** `src/llama-context.cpp:1417-1429`

**Issue:** Output reordering swaps elements one at a time:

```cpp
for (size_t s = 0; s < output_swaps.size(); ++s) {
    if (logits_size > 0) {
        for (uint64_t k = 0; k < n_vocab; k++) {
            std::swap(logits[i0*n_vocab + k], logits[i1*n_vocab + k]);
        }
    }
    if (embd_size > 0) {
        for (uint64_t k = 0; k < n_embd; k++) {
            std::swap(embd[i0*n_embd + k], embd[i1*n_embd + k]);
        }
    }
}
```

**Fix:**
- Use `std::swap_ranges` or `memcpy` with a temporary buffer
- For logits (n_vocab * sizeof(float) = 512KB for 128k vocab), use block operations
- Consider SIMD-based swap

**Expected Impact:** Medium. When output reordering is needed, this can be a significant overhead.

---

### 8.2 Selection Sort for Output Ordering

**Location:** `src/llama-context.cpp:1307-1322`

**Issue:** Uses O(n²) selection sort:

```cpp
// selection sort, to minimize swaps
for (uint32_t i = 0; i < n_outputs - 1; ++i) {
    uint32_t j_min = i;
    for (uint32_t j = i + 1; j < n_outputs; ++j) {
        if (out_ids[j] < out_ids[j_min]) {
            j_min = j;
        }
    }
    // ...
}
```

**Fix:**
- Use a permutation-based approach: compute the permutation, then apply it in one pass
- For small n_outputs, selection sort is fine; add a threshold to switch to O(n log n) for larger sizes
- Consider whether the "minimize swaps" goal is actually important

**Expected Impact:** Low (n_outputs is typically small), but could matter for large batch inference.

---

## 9. Ring Buffer Modulo Operations

### 9.1 Modulo in Ring Buffer Operations

**Location:** `src/llama-sampling.cpp:61, 66, 74, 97`

**Issue:** Ring buffer uses modulo operations which are expensive:

```cpp
first = (first + 1) % capacity;
// ...
data[pos] = value;
pos = (pos + 1) % capacity;
// ...
return data[(first + sz - i - 1) % capacity];
```

**Fix:**
- Use power-of-2 capacity with bitwise AND: `(pos + 1) & (capacity - 1)`
- Use branch-based wrap: `if (++pos >= capacity) pos = 0;`
- For sequential access patterns, unroll and avoid modulo

**Expected Impact:** Low-Medium. Modulo is ~20-80 cycles on modern CPUs vs 1 cycle for bitwise AND.

---

## 10. std::discrete_distribution Overhead

### 10.1 Distribution Object Creation

**Location:** `src/llama-sampling.cpp:242`

**Issue:** Creates a new `std::discrete_distribution` object on every call to `llama_sample_dist`:

```cpp
std::discrete_distribution<int> dist(probs_iterator{cur_p->data}, probs_iterator{cur_p->data + cur_p->size});
return dist(rng);
```

**Fix:**
- Note: The code at lines 650-674 already has an optimized path that samples without creating a distribution
- Ensure the optimized path (the `#if 1` block) is always used
- Consider removing the `llama_sample_dist` function if it's not used

**Expected Impact:** The optimization is already in place (line 650-674 in dist sampler), but `llama_sample_dist` at line 242 is still used by mirostat (line 1366). Consider updating mirostat to use the inline sampling approach.

---

## Summary: Priority Ranking

### High Priority (Significant Impact)
1. **SIMD softmax implementation** - Called on every sample, O(n_vocab)
2. **DRY sampler hash map operations** - O(n_vocab) hash lookups
3. **Penalties sampler: replace unordered_map with dense array**
4. **Typical sampler temporary vectors** - 3 allocations per sample
5. **Fast transcendental functions** - expf/logf are hot

### Medium Priority (Moderate Impact)
6. **min_p sampler temporary vector**
7. **Sampler chain indirect calls** - Consider fused/compiled samplers
8. **Token data array initialization SIMD**
9. **Output reordering with block operations**
10. **Greedy sampler SIMD max**

### Low Priority (Minor Impact)
11. **Ring buffer power-of-2 optimization**
12. **Partial sort thread_local vectors**
13. **Mirostat pow() optimization**
14. **Selection sort threshold**

---

## Benchmarking Recommendations

Before implementing any changes:

1. **Profile first**: Use perf/VTune/Instruments to confirm actual bottlenecks
2. **Measure sampling time separately**: The sampling hot path may be masked by decode time
3. **Test with various sampling configurations**: Temperature-only vs full chain (temp + top_k + top_p + min_p + penalties)
4. **Test vocab sizes**: Performance differs significantly between 32k and 128k+ vocab models

Consider creating a standalone sampling benchmark:
```bash
# Suggested test: time spent in sampling vs total inference
./llama-bench -m model.gguf --sampling temp=0.8,top_k=40,top_p=0.95 -n 1000
```
