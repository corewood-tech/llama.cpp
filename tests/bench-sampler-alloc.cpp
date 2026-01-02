// Benchmark: sampler hot path optimizations
// Build: g++ -O3 -march=native -std=c++17 -o bench-sampler-alloc tests/bench-sampler-alloc.cpp
// Run:   ./bench-sampler-alloc [n_vocab] [iterations]

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>
#include <numeric>
#include <unordered_map>

// Fast log approximation using IEEE 754 bit manipulation
// ~5-10x faster than logf, accurate to ~1-2% relative error
inline float fast_log2f(float x) {
    union { float f; uint32_t i; } vx = { x };
    float y = (float)(vx.i - 1064866805) * (1.0f / 8388608.0f);
    return y;
}

inline float fast_logf(float x) {
    return fast_log2f(x) * 0.6931471805599453f; // ln(2)
}

// Fast exp approximation using Schraudolph's method
// ~5x faster than expf, accurate to ~1-2% relative error
inline float fast_expf(float x) {
    union { float f; uint32_t i; } v;
    v.i = (uint32_t)((x * 12102203.0f) + 1065353216.0f);
    return v.f;
}

struct llama_token_data {
    int32_t id;
    float logit;
    float p;
};

using Clock = std::chrono::high_resolution_clock;

// Prevent compiler from optimizing away results
volatile int32_t g_sink = 0;
volatile float g_sink_f = 0;

// ---------------------------------------------------------------------------
// Isolate each component
// ---------------------------------------------------------------------------

double bench_alloc_only(int32_t n_vocab, int iterations) {
    auto start = Clock::now();
    for (int i = 0; i < iterations; i++) {
        std::vector<llama_token_data> cur;
        cur.reserve(n_vocab);
        g_sink += cur.capacity();
    }
    auto end = Clock::now();
    return std::chrono::duration<double, std::micro>(end - start).count() / iterations;
}

double bench_alloc_and_fill(const float * logits, int32_t n_vocab, int iterations) {
    auto start = Clock::now();
    for (int i = 0; i < iterations; i++) {
        std::vector<llama_token_data> cur;
        cur.reserve(n_vocab);
        for (int32_t j = 0; j < n_vocab; j++) {
            cur.emplace_back(llama_token_data{j, logits[j], 0.0f});
        }
        g_sink += cur.size();
    }
    auto end = Clock::now();
    return std::chrono::duration<double, std::micro>(end - start).count() / iterations;
}

double bench_fill_only_reuse(const float * logits, int32_t n_vocab, int iterations) {
    std::vector<llama_token_data> cur(n_vocab);  // pre-allocate once

    auto start = Clock::now();
    for (int i = 0; i < iterations; i++) {
        for (int32_t j = 0; j < n_vocab; j++) {
            cur[j] = {j, logits[j], 0.0f};
        }
        g_sink += cur[0].id;
    }
    auto end = Clock::now();
    return std::chrono::duration<double, std::micro>(end - start).count() / iterations;
}

double bench_scan_only(const float * logits, int32_t n_vocab, int iterations) {
    auto start = Clock::now();
    for (int i = 0; i < iterations; i++) {
        int32_t best = 0;
        float best_val = logits[0];
        for (int32_t j = 1; j < n_vocab; j++) {
            if (logits[j] > best_val) {
                best_val = logits[j];
                best = j;
            }
        }
        g_sink += best;
    }
    auto end = Clock::now();
    return std::chrono::duration<double, std::micro>(end - start).count() / iterations;
}

// Branchless argmax to test if branch misprediction is the issue
double bench_scan_branchless(const float * logits, int32_t n_vocab, int iterations) {
    auto start = Clock::now();
    for (int i = 0; i < iterations; i++) {
        int32_t best = 0;
        float best_val = logits[0];
        for (int32_t j = 1; j < n_vocab; j++) {
            bool is_better = logits[j] > best_val;
            best = is_better ? j : best;
            best_val = is_better ? logits[j] : best_val;
        }
        g_sink += best;
    }
    auto end = Clock::now();
    return std::chrono::duration<double, std::micro>(end - start).count() / iterations;
}

// What llama.cpp actually does: scan over token_data array
double bench_scan_token_data(const float * logits, int32_t n_vocab, int iterations) {
    std::vector<llama_token_data> cur(n_vocab);
    for (int32_t j = 0; j < n_vocab; j++) {
        cur[j] = {j, logits[j], 0.0f};
    }

    auto start = Clock::now();
    for (int i = 0; i < iterations; i++) {
        int64_t selected = 0;
        for (size_t j = 1; j < cur.size(); ++j) {
            if (cur[j].logit > cur[selected].logit) {
                selected = j;
            }
        }
        g_sink += cur[selected].id;
    }
    auto end = Clock::now();
    return std::chrono::duration<double, std::micro>(end - start).count() / iterations;
}

double bench_partial_sort(const float * logits, int32_t n_vocab, int32_t top_k, int iterations) {
    std::vector<llama_token_data> cur(n_vocab);

    auto start = Clock::now();
    for (int i = 0; i < iterations; i++) {
        // Fill
        for (int32_t j = 0; j < n_vocab; j++) {
            cur[j] = {j, logits[j], 0.0f};
        }
        // Partial sort
        std::partial_sort(cur.begin(), cur.begin() + top_k, cur.end(),
            [](const llama_token_data & a, const llama_token_data & b) {
                return a.logit > b.logit;
            });
        g_sink += cur[0].id;
    }
    auto end = Clock::now();
    return std::chrono::duration<double, std::micro>(end - start).count() / iterations;
}

double bench_nth_element(const float * logits, int32_t n_vocab, int32_t top_k, int iterations) {
    std::vector<llama_token_data> cur(n_vocab);

    auto start = Clock::now();
    for (int i = 0; i < iterations; i++) {
        // Fill
        for (int32_t j = 0; j < n_vocab; j++) {
            cur[j] = {j, logits[j], 0.0f};
        }
        // nth_element (faster than partial_sort if we don't need sorted order)
        std::nth_element(cur.begin(), cur.begin() + top_k, cur.end(),
            [](const llama_token_data & a, const llama_token_data & b) {
                return a.logit > b.logit;
            });
        g_sink += cur[0].id;
    }
    auto end = Clock::now();
    return std::chrono::duration<double, std::micro>(end - start).count() / iterations;
}

// ---------------------------------------------------------------------------
// 5.1: Temperature Scaling
// ---------------------------------------------------------------------------

double bench_temp_div(const float * logits, int32_t n_vocab, int iterations) {
    std::vector<llama_token_data> cur(n_vocab);
    for (int32_t j = 0; j < n_vocab; j++) {
        cur[j] = {j, logits[j], 0.0f};
    }
    float temp = 0.8f;

    auto start = Clock::now();
    for (int i = 0; i < iterations; i++) {
        for (size_t j = 0; j < cur.size(); ++j) {
            cur[j].logit /= temp;
        }
        g_sink_f += cur[0].logit;
        // Reset for next iteration
        for (int32_t j = 0; j < n_vocab; j++) {
            cur[j].logit = logits[j];
        }
    }
    auto end = Clock::now();
    return std::chrono::duration<double, std::micro>(end - start).count() / iterations;
}

double bench_temp_mul_inv(const float * logits, int32_t n_vocab, int iterations) {
    std::vector<llama_token_data> cur(n_vocab);
    for (int32_t j = 0; j < n_vocab; j++) {
        cur[j] = {j, logits[j], 0.0f};
    }
    float temp = 0.8f;
    float inv_temp = 1.0f / temp;

    auto start = Clock::now();
    for (int i = 0; i < iterations; i++) {
        for (size_t j = 0; j < cur.size(); ++j) {
            cur[j].logit *= inv_temp;
        }
        g_sink_f += cur[0].logit;
        // Reset for next iteration
        for (int32_t j = 0; j < n_vocab; j++) {
            cur[j].logit = logits[j];
        }
    }
    auto end = Clock::now();
    return std::chrono::duration<double, std::micro>(end - start).count() / iterations;
}

// ---------------------------------------------------------------------------
// 7.1: Fast Exp/Log Approximations
// ---------------------------------------------------------------------------

double bench_softmax_fast_exp(const float * logits, int32_t n_vocab, int iterations) {
    std::vector<llama_token_data> cur(n_vocab);
    for (int32_t j = 0; j < n_vocab; j++) {
        cur[j] = {j, logits[j], 0.0f};
    }
    std::vector<llama_token_data> backup = cur;

    auto start = Clock::now();
    for (int i = 0; i < iterations; i++) {
        // Find max
        float max_l = cur[0].logit;
        for (size_t j = 1; j < cur.size(); ++j) {
            max_l = std::max(max_l, cur[j].logit);
        }
        // Exp and sum using fast_expf
        float sum = 0.0f;
        for (size_t j = 0; j < cur.size(); ++j) {
            float p = fast_expf(cur[j].logit - max_l);
            cur[j].p = p;
            sum += p;
        }
        // Normalize
        float inv_sum = 1.0f / sum;
        for (size_t j = 0; j < cur.size(); ++j) {
            cur[j].p *= inv_sum;
        }
        g_sink_f += cur[0].p;
        cur = backup;
    }
    auto end = Clock::now();
    return std::chrono::duration<double, std::micro>(end - start).count() / iterations;
}

double bench_entropy_baseline(const float * logits, int32_t n_vocab, int iterations) {
    // Pre-compute probabilities
    std::vector<llama_token_data> cur(n_vocab);
    float sum = 0;
    for (int32_t j = 0; j < n_vocab; j++) {
        cur[j].id = j;
        cur[j].logit = logits[j];
        cur[j].p = expf(logits[j]);
        sum += cur[j].p;
    }
    for (auto& c : cur) c.p /= sum;

    auto start = Clock::now();
    for (int iter = 0; iter < iterations; iter++) {
        float entropy = 0.0f;
        for (size_t i = 0; i < cur.size(); ++i) {
            if (cur[i].p > 0) entropy += -cur[i].p * logf(cur[i].p);
        }
        g_sink_f += entropy;
    }
    auto end = Clock::now();
    return std::chrono::duration<double, std::micro>(end - start).count() / iterations;
}

double bench_entropy_fast_log(const float * logits, int32_t n_vocab, int iterations) {
    // Pre-compute probabilities
    std::vector<llama_token_data> cur(n_vocab);
    float sum = 0;
    for (int32_t j = 0; j < n_vocab; j++) {
        cur[j].id = j;
        cur[j].logit = logits[j];
        cur[j].p = expf(logits[j]);
        sum += cur[j].p;
    }
    for (auto& c : cur) c.p /= sum;

    auto start = Clock::now();
    for (int iter = 0; iter < iterations; iter++) {
        float entropy = 0.0f;
        for (size_t i = 0; i < cur.size(); ++i) {
            if (cur[i].p > 0) entropy += -cur[i].p * fast_logf(cur[i].p);
        }
        g_sink_f += entropy;
    }
    auto end = Clock::now();
    return std::chrono::duration<double, std::micro>(end - start).count() / iterations;
}

// ---------------------------------------------------------------------------
// 5.2: Softmax
// ---------------------------------------------------------------------------

double bench_softmax_baseline(const float * logits, int32_t n_vocab, int iterations) {
    std::vector<llama_token_data> cur(n_vocab);
    for (int32_t j = 0; j < n_vocab; j++) {
        cur[j] = {j, logits[j], 0.0f};
    }
    std::vector<llama_token_data> backup = cur;

    auto start = Clock::now();
    for (int i = 0; i < iterations; i++) {
        // Find max
        float max_l = cur[0].logit;
        for (size_t j = 1; j < cur.size(); ++j) {
            max_l = std::max(max_l, cur[j].logit);
        }
        // Exp and sum
        float sum = 0.0f;
        for (size_t j = 0; j < cur.size(); ++j) {
            float p = expf(cur[j].logit - max_l);
            cur[j].p = p;
            sum += p;
        }
        // Normalize (division)
        for (size_t j = 0; j < cur.size(); ++j) {
            cur[j].p /= sum;
        }
        g_sink_f += cur[0].p;
        cur = backup;
    }
    auto end = Clock::now();
    return std::chrono::duration<double, std::micro>(end - start).count() / iterations;
}

double bench_softmax_inv_mul(const float * logits, int32_t n_vocab, int iterations) {
    std::vector<llama_token_data> cur(n_vocab);
    for (int32_t j = 0; j < n_vocab; j++) {
        cur[j] = {j, logits[j], 0.0f};
    }
    std::vector<llama_token_data> backup = cur;

    auto start = Clock::now();
    for (int i = 0; i < iterations; i++) {
        // Find max
        float max_l = cur[0].logit;
        for (size_t j = 1; j < cur.size(); ++j) {
            max_l = std::max(max_l, cur[j].logit);
        }
        // Exp and sum
        float sum = 0.0f;
        for (size_t j = 0; j < cur.size(); ++j) {
            float p = expf(cur[j].logit - max_l);
            cur[j].p = p;
            sum += p;
        }
        // Normalize (multiply by inverse)
        float inv_sum = 1.0f / sum;
        for (size_t j = 0; j < cur.size(); ++j) {
            cur[j].p *= inv_sum;
        }
        g_sink_f += cur[0].p;
        cur = backup;
    }
    auto end = Clock::now();
    return std::chrono::duration<double, std::micro>(end - start).count() / iterations;
}

// ---------------------------------------------------------------------------
// 4.2: Penalties Lookup (Hash Map vs Dense Array)
// ---------------------------------------------------------------------------

double bench_penalties_hashmap(const float * logits, int32_t n_vocab, int penalty_last_n, int iterations) {
    std::unordered_map<int32_t, int> token_count;

    // Fill with some tokens
    std::mt19937 rng(123);
    std::uniform_int_distribution<int32_t> dist(0, n_vocab - 1);
    for (int i = 0; i < penalty_last_n; i++) {
        token_count[dist(rng)]++;
    }

    auto start = Clock::now();
    for (int i = 0; i < iterations; i++) {
        int sum = 0;
        for (int32_t j = 0; j < n_vocab; j++) {
            auto it = token_count.find(j);
            if (it != token_count.end()) {
                sum += it->second;
            }
        }
        g_sink += sum;
    }
    auto end = Clock::now();
    return std::chrono::duration<double, std::micro>(end - start).count() / iterations;
}

double bench_penalties_dense(const float * logits, int32_t n_vocab, int penalty_last_n, int iterations) {
    std::vector<int> token_count(n_vocab, 0);

    // Fill with same pattern
    std::mt19937 rng(123);
    std::uniform_int_distribution<int32_t> dist(0, n_vocab - 1);
    for (int i = 0; i < penalty_last_n; i++) {
        token_count[dist(rng)]++;
    }

    auto start = Clock::now();
    for (int i = 0; i < iterations; i++) {
        int sum = 0;
        for (int32_t j = 0; j < n_vocab; j++) {
            sum += token_count[j];
        }
        g_sink += sum;
    }
    auto end = Clock::now();
    return std::chrono::duration<double, std::micro>(end - start).count() / iterations;
}

// ---------------------------------------------------------------------------
// 1.1: min_p Vector Allocation
// ---------------------------------------------------------------------------

double bench_min_p_baseline(const float * logits, int32_t n_vocab, int iterations) {
    std::vector<llama_token_data> cur(n_vocab);
    for (int32_t j = 0; j < n_vocab; j++) {
        cur[j] = {j, logits[j], 0.0f};
    }
    std::vector<llama_token_data> backup = cur;
    float p = 0.05f;
    size_t min_keep = 1;

    auto start = Clock::now();
    for (int i = 0; i < iterations; i++) {
        // Baseline: allocate new vector every time
        std::vector<llama_token_data> filtered;

        float max_l = -INFINITY;
        for (size_t j = 0; j < cur.size(); ++j) {
            max_l = std::max(max_l, cur[j].logit);
        }
        float min_logit = max_l + logf(p);

        for (size_t j = 0; j < cur.size(); ++j) {
            if (cur[j].logit >= min_logit) {
                filtered.push_back(cur[j]);
            }
        }

        if (!filtered.empty() && filtered.size() >= min_keep) {
            g_sink += filtered.size();
        }
        cur = backup;
    }
    auto end = Clock::now();
    return std::chrono::duration<double, std::micro>(end - start).count() / iterations;
}

double bench_min_p_thread_local(const float * logits, int32_t n_vocab, int iterations) {
    std::vector<llama_token_data> cur(n_vocab);
    for (int32_t j = 0; j < n_vocab; j++) {
        cur[j] = {j, logits[j], 0.0f};
    }
    std::vector<llama_token_data> backup = cur;
    float p = 0.05f;
    size_t min_keep = 1;

    // Thread-local buffer
    static thread_local std::vector<llama_token_data> filtered;

    auto start = Clock::now();
    for (int i = 0; i < iterations; i++) {
        filtered.clear();
        filtered.reserve(cur.size());

        float max_l = -INFINITY;
        for (size_t j = 0; j < cur.size(); ++j) {
            max_l = std::max(max_l, cur[j].logit);
        }
        float min_logit = max_l + logf(p);

        for (size_t j = 0; j < cur.size(); ++j) {
            if (cur[j].logit >= min_logit) {
                filtered.push_back(cur[j]);
            }
        }

        if (!filtered.empty() && filtered.size() >= min_keep) {
            g_sink += filtered.size();
        }
        cur = backup;
    }
    auto end = Clock::now();
    return std::chrono::duration<double, std::micro>(end - start).count() / iterations;
}

double bench_min_p_inplace(const float * logits, int32_t n_vocab, int iterations) {
    std::vector<llama_token_data> cur(n_vocab);
    for (int32_t j = 0; j < n_vocab; j++) {
        cur[j] = {j, logits[j], 0.0f};
    }
    std::vector<llama_token_data> backup = cur;
    float p = 0.05f;
    size_t min_keep = 1;

    auto start = Clock::now();
    for (int i = 0; i < iterations; i++) {
        float max_l = -INFINITY;
        for (size_t j = 0; j < cur.size(); ++j) {
            max_l = std::max(max_l, cur[j].logit);
        }
        float min_logit = max_l + logf(p);

        size_t write_idx = 0;
        for (size_t j = 0; j < cur.size(); ++j) {
            if (cur[j].logit >= min_logit) {
                if (write_idx != j) {
                    cur[write_idx] = cur[j];
                }
                write_idx++;
            }
        }

        if (write_idx >= min_keep) {
            g_sink += write_idx;
        }
        cur = backup;
    }
    auto end = Clock::now();
    return std::chrono::duration<double, std::micro>(end - start).count() / iterations;
}

// ---------------------------------------------------------------------------
// 9.1: Ring Buffer Modulo vs Bitwise
// ---------------------------------------------------------------------------

template<typename T>
struct ring_buffer_mod {
    size_t cap, sz = 0, first = 0, pos = 0;
    std::vector<T> data;
    ring_buffer_mod(size_t c) : cap(c), data(c) {}

    void push(T v) {
        if (sz == cap) first = (first + 1) % cap;
        else sz++;
        data[pos] = v;
        pos = (pos + 1) % cap;
    }

    T rat(size_t i) const { return data[(first + sz - i - 1) % cap]; }
};

template<typename T>
struct ring_buffer_bit {
    size_t cap, mask, sz = 0, first = 0, pos = 0;
    std::vector<T> data;
    ring_buffer_bit(size_t c) {
        size_t p2 = 1;
        while (p2 < c) p2 <<= 1;
        cap = p2; mask = p2 - 1;
        data.resize(p2);
    }

    void push(T v) {
        if (sz == cap) first = (first + 1) & mask;
        else sz++;
        data[pos] = v;
        pos = (pos + 1) & mask;
    }

    T rat(size_t i) const { return data[(first + sz - i - 1) & mask]; }
};

double bench_ring_mod(int size, int ops, int iterations) {
    ring_buffer_mod<int32_t> rb(size);

    auto start = Clock::now();
    for (int iter = 0; iter < iterations; iter++) {
        rb.sz = rb.first = rb.pos = 0;
        for (int i = 0; i < ops; i++) {
            rb.push(i);
        }
        int sum = 0;
        for (size_t i = 0; i < std::min((size_t)ops, rb.sz); i++) {
            sum += rb.rat(i);
        }
        g_sink += sum;
    }
    auto end = Clock::now();
    return std::chrono::duration<double, std::micro>(end - start).count() / iterations;
}

double bench_ring_bit(int size, int ops, int iterations) {
    ring_buffer_bit<int32_t> rb(size);

    auto start = Clock::now();
    for (int iter = 0; iter < iterations; iter++) {
        rb.sz = rb.first = rb.pos = 0;
        for (int i = 0; i < ops; i++) {
            rb.push(i);
        }
        int sum = 0;
        for (size_t i = 0; i < std::min((size_t)ops, rb.sz); i++) {
            sum += rb.rat(i);
        }
        g_sink += sum;
    }
    auto end = Clock::now();
    return std::chrono::duration<double, std::micro>(end - start).count() / iterations;
}

// ---------------------------------------------------------------------------
// 1.2: Typical Sampler (3 vector allocations)
// ---------------------------------------------------------------------------

double bench_typical_baseline(const float * logits, int32_t n_vocab, int iterations) {
    // Pre-compute probabilities
    std::vector<llama_token_data> cur(n_vocab);
    float sum = 0;
    for (int32_t j = 0; j < n_vocab; j++) {
        cur[j].id = j;
        cur[j].logit = logits[j];
        cur[j].p = expf(logits[j]);
        sum += cur[j].p;
    }
    for (auto& c : cur) c.p /= sum;
    std::vector<llama_token_data> backup = cur;

    float p = 0.9f;
    size_t min_keep = 1;

    auto start = Clock::now();
    for (int iter = 0; iter < iterations; iter++) {
        // Compute entropy
        float entropy = 0.0f;
        for (size_t i = 0; i < cur.size(); ++i) {
            if (cur[i].p > 0) entropy += -cur[i].p * logf(cur[i].p);
        }

        // Allocation 1
        std::vector<float> shifted_scores;
        for (size_t i = 0; i < cur.size(); ++i) {
            shifted_scores.push_back(fabsf(-logf(cur[i].p) - entropy));
        }

        // Allocation 2
        std::vector<size_t> indices(cur.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
            return shifted_scores[a] < shifted_scores[b];
        });

        float cum = 0.0f;
        size_t last_idx = indices.size();
        for (size_t i = 0; i < indices.size(); ++i) {
            cum += cur[indices[i]].p;
            if (cum > p && i >= min_keep - 1) { last_idx = i + 1; break; }
        }

        // Allocation 3
        std::vector<llama_token_data> result;
        for (size_t i = 0; i < last_idx; ++i) {
            result.push_back(cur[indices[i]]);
        }

        g_sink += result.size();
        cur = backup;
    }
    auto end = Clock::now();
    return std::chrono::duration<double, std::micro>(end - start).count() / iterations;
}

double bench_typical_optimized(const float * logits, int32_t n_vocab, int iterations) {
    // Pre-compute probabilities
    std::vector<llama_token_data> cur(n_vocab);
    float sum = 0;
    for (int32_t j = 0; j < n_vocab; j++) {
        cur[j].id = j;
        cur[j].logit = logits[j];
        cur[j].p = expf(logits[j]);
        sum += cur[j].p;
    }
    for (auto& c : cur) c.p /= sum;
    std::vector<llama_token_data> backup = cur;

    float p = 0.9f;
    size_t min_keep = 1;

    // Thread-local buffers
    static thread_local std::vector<float> shifted_scores;
    static thread_local std::vector<size_t> indices;
    static thread_local std::vector<llama_token_data> result;

    auto start = Clock::now();
    for (int iter = 0; iter < iterations; iter++) {
        shifted_scores.clear();
        shifted_scores.reserve(cur.size());
        indices.resize(cur.size());
        result.clear();

        // Compute entropy
        float entropy = 0.0f;
        for (size_t i = 0; i < cur.size(); ++i) {
            if (cur[i].p > 0) entropy += -cur[i].p * logf(cur[i].p);
        }

        for (size_t i = 0; i < cur.size(); ++i) {
            shifted_scores.push_back(fabsf(-logf(cur[i].p) - entropy));
        }

        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
            return shifted_scores[a] < shifted_scores[b];
        });

        float cum = 0.0f;
        size_t last_idx = indices.size();
        for (size_t i = 0; i < indices.size(); ++i) {
            cum += cur[indices[i]].p;
            if (cum > p && i >= min_keep - 1) { last_idx = i + 1; break; }
        }

        result.reserve(last_idx);
        for (size_t i = 0; i < last_idx; ++i) {
            result.push_back(cur[indices[i]]);
        }

        g_sink += result.size();
        cur = backup;
    }
    auto end = Clock::now();
    return std::chrono::duration<double, std::micro>(end - start).count() / iterations;
}

void print_speedup(const char* name, double baseline, double optimized) {
    double speedup = baseline / optimized;
    const char* status = speedup >= 1.0 ? "faster" : "SLOWER";
    if (speedup < 1.0) speedup = 1.0 / speedup;
    printf("  %-40s %8.2f -> %8.2f us  (%.2fx %s)\n",
           name, baseline, optimized, speedup, status);
}

int main(int argc, char ** argv) {
    int32_t n_vocab = 128256;
    int iterations = 1000;
    int32_t top_k = 40;
    int penalty_last_n = 64;

    if (argc > 1) n_vocab = atoi(argv[1]);
    if (argc > 2) iterations = atoi(argv[2]);

    printf("============================================================\n");
    printf("  SAMPLER HOT PATH BENCHMARK\n");
    printf("============================================================\n");
    printf("Vocab: %d, Iterations: %d, Top-K: %d\n\n", n_vocab, iterations, top_k);

    // Generate random logits
    std::vector<float> logits(n_vocab);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto & l : logits) l = dist(rng);

    // Warmup
    printf("Warming up...\n");
    for (int i = 0; i < 10; i++) {
        bench_alloc_and_fill(logits.data(), n_vocab, 1);
        bench_scan_only(logits.data(), n_vocab, 1);
        bench_softmax_baseline(logits.data(), n_vocab, 1);
    }

    // =========================================================================
    printf("\n=== ORIGINAL BENCHMARKS ===\n");
    // =========================================================================

    double t_alloc = bench_alloc_only(n_vocab, iterations);
    double t_alloc_fill = bench_alloc_and_fill(logits.data(), n_vocab, iterations);
    double t_fill_reuse = bench_fill_only_reuse(logits.data(), n_vocab, iterations);
    double t_scan = bench_scan_only(logits.data(), n_vocab, iterations);
    double t_scan_branchless = bench_scan_branchless(logits.data(), n_vocab, iterations);
    double t_scan_token_data = bench_scan_token_data(logits.data(), n_vocab, iterations);
    double t_partial_sort = bench_partial_sort(logits.data(), n_vocab, top_k, iterations);
    double t_nth = bench_nth_element(logits.data(), n_vocab, top_k, iterations);

    printf("%-40s: %8.2f us\n", "alloc only (reserve)", t_alloc);
    printf("%-40s: %8.2f us\n", "alloc + fill (emplace_back)", t_alloc_fill);
    printf("%-40s: %8.2f us\n", "fill only (reuse, direct assign)", t_fill_reuse);
    printf("%-40s: %8.2f us\n", "scan raw floats (argmax)", t_scan);
    printf("%-40s: %8.2f us\n", "scan raw floats (branchless)", t_scan_branchless);
    printf("%-40s: %8.2f us\n", "scan token_data array", t_scan_token_data);
    printf("%-40s: %8.2f us\n", "fill + partial_sort (k=40)", t_partial_sort);
    printf("%-40s: %8.2f us\n", "fill + nth_element (k=40)", t_nth);

    // =========================================================================
    printf("\n=== 5.1: TEMPERATURE SCALING ===\n");
    // =========================================================================

    double t_temp_div = bench_temp_div(logits.data(), n_vocab, iterations);
    double t_temp_mul = bench_temp_mul_inv(logits.data(), n_vocab, iterations);
    print_speedup("temp: div -> mul inverse", t_temp_div, t_temp_mul);

    // =========================================================================
    printf("\n=== 5.2: SOFTMAX ===\n");
    // =========================================================================

    double t_softmax_base = bench_softmax_baseline(logits.data(), n_vocab, iterations);
    double t_softmax_inv = bench_softmax_inv_mul(logits.data(), n_vocab, iterations);
    double t_softmax_fast = bench_softmax_fast_exp(logits.data(), n_vocab, iterations);
    print_speedup("softmax: div -> mul inverse", t_softmax_base, t_softmax_inv);
    print_speedup("softmax: expf -> fast_expf", t_softmax_inv, t_softmax_fast);

    // =========================================================================
    printf("\n=== 7.1: FAST EXP/LOG (5.3 entropy) ===\n");
    // =========================================================================

    double t_entropy_base = bench_entropy_baseline(logits.data(), n_vocab, iterations);
    double t_entropy_fast = bench_entropy_fast_log(logits.data(), n_vocab, iterations);
    print_speedup("entropy: logf -> fast_logf", t_entropy_base, t_entropy_fast);

    // =========================================================================
    printf("\n=== 4.2: PENALTIES LOOKUP ===\n");
    // =========================================================================

    double t_pen_hash = bench_penalties_hashmap(logits.data(), n_vocab, penalty_last_n, iterations);
    double t_pen_dense = bench_penalties_dense(logits.data(), n_vocab, penalty_last_n, iterations);
    print_speedup("penalties: hashmap -> dense array", t_pen_hash, t_pen_dense);

    // =========================================================================
    printf("\n=== 1.1: MIN_P VECTOR ALLOCATION ===\n");
    // =========================================================================

    double t_minp_base = bench_min_p_baseline(logits.data(), n_vocab, iterations);
    double t_minp_tl = bench_min_p_thread_local(logits.data(), n_vocab, iterations);
    double t_minp_ip = bench_min_p_inplace(logits.data(), n_vocab, iterations);
    print_speedup("min_p: new vec -> thread_local", t_minp_base, t_minp_tl);
    print_speedup("min_p: new vec -> in-place", t_minp_base, t_minp_ip);

    // =========================================================================
    printf("\n=== 1.2: TYPICAL SAMPLER (3 ALLOCS) ===\n");
    // =========================================================================

    // Use fewer iterations for typical (it's slow)
    int typical_iters = std::min(iterations, 100);
    double t_typ_base = bench_typical_baseline(logits.data(), n_vocab, typical_iters);
    double t_typ_opt = bench_typical_optimized(logits.data(), n_vocab, typical_iters);
    print_speedup("typical: 3 allocs -> thread_local", t_typ_base, t_typ_opt);

    // =========================================================================
    printf("\n=== 9.1: RING BUFFER MODULO ===\n");
    // =========================================================================

    int ring_size = 64;
    int ring_ops = 1000;
    double t_ring_mod = bench_ring_mod(ring_size, ring_ops, iterations);
    double t_ring_bit = bench_ring_bit(ring_size, ring_ops, iterations);
    print_speedup("ring: modulo -> bitwise AND", t_ring_mod, t_ring_bit);

    // =========================================================================
    printf("\n=== 6.2: GREEDY MAX (from original) ===\n");
    // =========================================================================

    print_speedup("greedy: token_data vs raw float", t_scan_token_data, t_scan);

    // =========================================================================
    printf("\n============================================================\n");
    printf("  SUMMARY\n");
    printf("============================================================\n");
    // =========================================================================

    printf("\n%-45s %10s %10s %10s\n", "Optimization", "Baseline", "Optimized", "Speedup");
    printf("%-45s %10s %10s %10s\n", "---------------------------------------------", "----------", "----------", "----------");

    auto summary = [](const char* name, double base, double opt) {
        double spd = base / opt;
        printf("%-45s %9.2f %9.2f %9.2fx\n", name, base, opt, spd);
    };

    summary("5.1 Temperature (div -> mul)", t_temp_div, t_temp_mul);
    summary("5.2 Softmax (div -> mul)", t_softmax_base, t_softmax_inv);
    summary("7.1 Softmax (expf -> fast_expf)", t_softmax_inv, t_softmax_fast);
    summary("5.3 Entropy (logf -> fast_logf)", t_entropy_base, t_entropy_fast);
    summary("4.2 Penalties (hashmap -> dense)", t_pen_hash, t_pen_dense);
    summary("1.1 min_p (alloc -> thread_local)", t_minp_base, t_minp_tl);
    summary("1.1 min_p (alloc -> in-place)", t_minp_base, t_minp_ip);
    summary("1.2 typical (3 allocs -> thread_local)", t_typ_base, t_typ_opt);
    summary("9.1 ring buffer (mod -> bitwise)", t_ring_mod, t_ring_bit);
    summary("6.2 greedy scan (struct vs raw)", t_scan_token_data, t_scan);

    printf("\n");

    return 0;
}
