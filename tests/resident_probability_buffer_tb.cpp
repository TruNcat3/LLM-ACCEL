#include "../kernel/control_cache_8x64.cpp"

#include <cstdio>

static fm_t score_value(unsigned int row, unsigned int elem) {
    const int raw = int((row * 23 + elem * 5 + 11) % 67) - 33;
    return fm_t(raw) / fm_t(16);
}

int main() {
    constexpr unsigned int kTileLen = CC8_ATTN_TILE - 3;
    cc8_attention_buffer_t scores;
    cc8_attention_buffer_t legacy_probabilities;
    cc8_resident_probability_buffer_t resident_probabilities;
    clear_cc8_attention_buffer(scores);

    for (unsigned int row = 0; row < GQA_GROUP_SIZE; row++) {
        for (unsigned int elem = 0; elem < CC8_ATTN_TILE; elem++) {
            write_cc8_attention_buffer_value(
                scores,
                row,
                elem,
                score_value(row, elem)
            );
        }
    }

    fm_t legacy_max[MM_STREAM_8X64_TOKENS];
    fm_t resident_max[MM_STREAM_8X64_TOKENS];
    fm_t legacy_scale[MM_STREAM_8X64_TOKENS];
    fm_t resident_scale[MM_STREAM_8X64_TOKENS];
    fm_accum_t legacy_sum[MM_STREAM_8X64_TOKENS];
    fm_accum_t resident_sum[MM_STREAM_8X64_TOKENS];
    for (unsigned int row = 0; row < MM_STREAM_8X64_TOKENS; row++) {
        fm_t initial_max = fm_t(-128);
        fm_accum_t initial_sum = fm_accum_t(0);
        if (row < GQA_GROUP_SIZE) {
            initial_max = fm_t(
                fm_t(-1.5) + fm_t(row) / fm_t(8)
            );
            initial_sum = fm_accum_t(
                fm_accum_t(0.75) +
                fm_accum_t(row) / fm_accum_t(16)
            );
        }
        legacy_max[row] = initial_max;
        resident_max[row] = initial_max;
        legacy_sum[row] = initial_sum;
        resident_sum[row] = initial_sum;
        legacy_scale[row] = fm_t(7);
        resident_scale[row] = fm_t(7);
    }

    compute_cc8_flash_probabilities_one_core(
        legacy_probabilities,
        scores,
        legacy_max,
        legacy_sum,
        legacy_scale,
        kTileLen
    );
    compute_cc8_resident_flash_probabilities_one_core(
        resident_probabilities,
        scores,
        resident_max,
        resident_sum,
        resident_scale,
        kTileLen
    );

    int errors = 0;
    for (unsigned int row = 0; row < MM_STREAM_8X64_TOKENS; row++) {
        if (legacy_max[row] != resident_max[row] ||
            legacy_sum[row] != resident_sum[row] ||
            legacy_scale[row] != resident_scale[row]) {
            if (errors < 8) {
                std::printf("state mismatch row=%u\n", row);
            }
            errors++;
        }
        if (row < GQA_GROUP_SIZE) {
            for (unsigned int elem = 0;
                 elem < CC8_ATTN_TILE;
                 elem++) {
                const fm_t legacy =
                    read_cc8_gbuf_value(legacy_probabilities, row, elem);
                const fm_t resident =
                    resident_probabilities.value[row][elem];
                if (legacy != resident) {
                    if (errors < 8) {
                        std::printf(
                            "probability mismatch row=%u elem=%u\n",
                            row,
                            elem
                        );
                    }
                    errors++;
                }
            }
        }
    }

    std::printf(
        "RESIDENT PROBABILITY BUFFER TEST %s errors=%d\n",
        errors == 0 ? "PASS" : "FAIL",
        errors
    );
    return errors == 0 ? 0 : 1;
}
