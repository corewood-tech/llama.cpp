// Benchmark: GPU argmax vs CPU argmax (full logits transfer)
// Tests the performance difference between:
// 1. llama_get_logits_ith() + CPU argmax (old path)
// 2. llama_get_argmax_ith() (new GPU path)

#include "llama.h"
#include "common.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

using Clock = std::chrono::high_resolution_clock;

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [n_tokens=100]\n", argv[0]);
        return 1;
    }

    const char * model_path = argv[1];
    const int n_tokens = argc > 2 ? atoi(argv[2]) : 100;

    // Initialize
    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(model_path, model_params);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;
    ctx_params.n_batch = 512;
    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "Failed to create context\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    printf("Model: %s\n", model_path);
    printf("Vocab size: %d\n", n_vocab);
    printf("Tokens to generate: %d\n", n_tokens);
    printf("Logits transfer size: %.2f KB\n", n_vocab * sizeof(float) / 1024.0);
    printf("\n");

    // Create greedy sampler
    llama_sampler * smpl_greedy = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl_greedy, llama_sampler_init_greedy());

    // Tokenize a simple prompt
    std::vector<llama_token> tokens = {1}; // BOS or simple start token

    // Decode prompt
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "Failed to decode prompt\n");
        return 1;
    }

    // ============================================
    // CORRECTNESS VERIFICATION
    // ============================================
    printf("=== Verifying GPU argmax correctness ===\n");
    int mismatches = 0;
    const int n_verify = 20;

    for (int i = 0; i < n_verify; i++) {
        batch = llama_batch_get_one(&tokens.back(), 1);
        llama_decode(ctx, batch);
        llama_synchronize(ctx);

        // Get GPU argmax
        llama_token gpu_argmax = llama_get_argmax_ith(ctx, -1);

        // Get CPU argmax from logits
        const float * logits = llama_get_logits_ith(ctx, -1);
        llama_token cpu_argmax = 0;
        float best_logit = logits[0];
        for (int j = 1; j < n_vocab; j++) {
            if (logits[j] > best_logit) {
                best_logit = logits[j];
                cpu_argmax = j;
            }
        }

        if (gpu_argmax != cpu_argmax) {
            printf("  MISMATCH at token %d: GPU=%d, CPU=%d (logit=%.4f)\n",
                   i, gpu_argmax, cpu_argmax, best_logit);
            mismatches++;
        }

        tokens.push_back(gpu_argmax);
    }

    if (mismatches == 0) {
        printf("  PASSED: All %d tokens match between GPU and CPU argmax\n", n_verify);
    } else {
        printf("  FAILED: %d/%d mismatches\n", mismatches, n_verify);
        return 1;
    }

    // Warm up
    printf("\nWarming up...\n");
    for (int i = 0; i < 5; i++) {
        llama_token token = llama_sampler_sample(smpl_greedy, ctx, -1);
        tokens.push_back(token);
        batch = llama_batch_get_one(&tokens.back(), 1);
        llama_decode(ctx, batch);
    }

    // ============================================
    // Benchmark 1: Using llama_sampler_sample (should use GPU argmax for greedy)
    // ============================================
    printf("\n=== Benchmark: llama_sampler_sample() with greedy sampler ===\n");

    std::vector<double> times_sampler;
    for (int i = 0; i < n_tokens; i++) {
        // Decode first (outside timing)
        batch = llama_batch_get_one(&tokens.back(), 1);
        llama_decode(ctx, batch);
        llama_synchronize(ctx);  // Ensure GPU work is done before timing

        // Time only the sampling
        auto t0 = Clock::now();
        llama_token token = llama_sampler_sample(smpl_greedy, ctx, -1);
        auto t1 = Clock::now();

        double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        times_sampler.push_back(us);

        tokens.push_back(token);
    }

    // ============================================
    // Benchmark 2: Direct llama_get_argmax_ith (explicit GPU argmax)
    // ============================================
    printf("\n=== Benchmark: llama_get_argmax_ith() (explicit GPU argmax) ===\n");

    std::vector<double> times_argmax;
    for (int i = 0; i < n_tokens; i++) {
        // Decode first (outside timing)
        batch = llama_batch_get_one(&tokens.back(), 1);
        llama_decode(ctx, batch);
        llama_synchronize(ctx);  // Ensure GPU work is done before timing

        // Time only the argmax retrieval
        auto t0 = Clock::now();
        llama_token argmax_token = llama_get_argmax_ith(ctx, -1);
        auto t1 = Clock::now();

        double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        times_argmax.push_back(us);

        tokens.push_back(argmax_token);
    }

    // ============================================
    // Benchmark 3: llama_get_logits_ith + CPU argmax (old path)
    // ============================================
    printf("\n=== Benchmark: llama_get_logits_ith() + CPU argmax (old path) ===\n");

    std::vector<double> times_logits;
    for (int i = 0; i < n_tokens; i++) {
        // Decode first (outside timing)
        batch = llama_batch_get_one(&tokens.back(), 1);
        llama_decode(ctx, batch);
        llama_synchronize(ctx);  // Ensure GPU work is done before timing

        // Time the logits retrieval + CPU argmax
        auto t0 = Clock::now();
        const float * logits = llama_get_logits_ith(ctx, -1);

        // CPU argmax
        llama_token best_token = 0;
        float best_logit = logits[0];
        for (int j = 1; j < n_vocab; j++) {
            if (logits[j] > best_logit) {
                best_logit = logits[j];
                best_token = j;
            }
        }
        auto t1 = Clock::now();

        double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        times_logits.push_back(us);

        tokens.push_back(best_token);
    }

    // ============================================
    // Results
    // ============================================
    auto calc_stats = [](const std::vector<double>& times) {
        double sum = 0, min = times[0], max = times[0];
        for (double t : times) {
            sum += t;
            if (t < min) min = t;
            if (t > max) max = t;
        }
        double mean = sum / times.size();
        return std::make_tuple(mean, min, max);
    };

    auto [mean1, min1, max1] = calc_stats(times_sampler);
    auto [mean2, min2, max2] = calc_stats(times_argmax);
    auto [mean3, min3, max3] = calc_stats(times_logits);

    printf("\n");
    printf("=== RESULTS ===\n");
    printf("%-40s %8.2f us (min: %.2f, max: %.2f)\n", "llama_sampler_sample (greedy):", mean1, min1, max1);
    printf("%-40s %8.2f us (min: %.2f, max: %.2f)\n", "llama_get_argmax_ith (GPU):", mean2, min2, max2);
    printf("%-40s %8.2f us (min: %.2f, max: %.2f)\n", "llama_get_logits_ith + CPU argmax:", mean3, min3, max3);
    printf("\n");
    printf("Speedup (GPU argmax vs logits+CPU): %.2fx\n", mean3 / mean2);
    printf("Speedup (sampler vs logits+CPU):    %.2fx\n", mean3 / mean1);

    // Cleanup
    llama_sampler_free(smpl_greedy);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
