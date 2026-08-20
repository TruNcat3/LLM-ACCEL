#include "../kernel/control_cache_8x64.cpp"

#include <cstdio>

static fm_t score_value(unsigned int row, unsigned int elem) {
    const int raw = int((row * 23 + elem * 5 + 11) % 67) - 33;
    return fm_t(raw) / fm_t(16);
}

int main() {
    constexpr unsigned int kTileLen = CC8_ATTN_TILE - 3;
    cc8_attention_buffer_t scores;
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

    fm_t resident_max[MM_STREAM_8X64_TOKENS];
    attention_prob_t resident_scale[MM_STREAM_8X64_TOKENS];
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
        resident_max[row] = initial_max;
        resident_sum[row] = initial_sum;
        resident_scale[row] = attention_prob_t(0);
    }

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
        if (row >= GQA_GROUP_SIZE) {
            if (resident_scale[row] != attention_prob_t(0)) {
                if (errors < 8) {
                    std::printf("inactive scale mismatch row=%u\n", row);
                }
                errors++;
            }
            continue;
        }

        fm_t expected_max = fm_t(-1.5) + fm_t(row) / fm_t(8);
        fm_accum_t expected_sum =
            fm_accum_t(0.75) + fm_accum_t(row) / fm_accum_t(16);
        for (unsigned int elem = 0; elem < kTileLen; elem++) {
            const fm_t score = score_value(row, elem);
            if (score > expected_max) {
                expected_max = score;
            }
        }
        const attention_prob_t expected_scale =
            cc8_exp_attention_probability(
                (fm_t(-1.5) + fm_t(row) / fm_t(8)) - expected_max
            );
        fm_accum_t expected_tile_sum = fm_accum_t(0);
        for (unsigned int elem = 0; elem < CC8_ATTN_TILE; elem++) {
            const attention_prob_t expected_probability =
                elem < kTileLen ?
                cc8_exp_attention_probability(
                    score_value(row, elem) - expected_max
                ) :
                attention_prob_t(0);
            expected_tile_sum += fm_accum_t(expected_probability);
            if (resident_probabilities.value[row][elem] !=
                expected_probability) {
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
        expected_sum =
            expected_sum * fm_accum_t(expected_scale) + expected_tile_sum;

        if (resident_max[row] != expected_max ||
            resident_sum[row] != expected_sum ||
            resident_scale[row] != expected_scale) {
            if (errors < 8) {
                std::printf("state mismatch row=%u\n", row);
            }
            errors++;
        }
    }

    // The PV stream keeps the external payload at 16 bits.  Verify that the
    // Q2.14 probability is reinterpreted bit-for-bit as the compute CU's Q8.8
    // activation type; numerical conversion here would silently lose the
    // six extra fractional bits before the task-level 1/64 compensation.
    cc8_attention_pv_packet_panel_t v0;
    cc8_attention_pv_packet_panel_t v1;
    for (unsigned int group = 0;
         group < CC8_ATTN_PACKET_GROUPS;
         group++) {
        for (unsigned int panel_index = 0;
             panel_index < CC8_ATTN_PV_PACKET_DEPTH;
             panel_index++) {
            v0.packet[group][panel_index] = 0;
            v1.packet[group][panel_index] = 0;
        }
    }

    hls::stream<mm_stream_8x64_activation_packet_t> activation0;
    hls::stream<mm_stream_8x64_activation_packet_t> activation1;
    hls::stream<mm_stream_8x64_weight_packet_t> weights00;
    hls::stream<mm_stream_8x64_weight_packet_t> weights01;
    hls::stream<mm_stream_8x64_weight_packet_t> weights02;
    hls::stream<mm_stream_8x64_weight_packet_t> weights03;
    hls::stream<mm_stream_8x64_weight_packet_t> weights10;
    hls::stream<mm_stream_8x64_weight_packet_t> weights11;
    hls::stream<mm_stream_8x64_weight_packet_t> weights12;
    hls::stream<mm_stream_8x64_weight_packet_t> weights13;
    emit_cc8_attention_pv_packet_inputs(
        activation0,
        weights00,
        weights01,
        weights02,
        weights03,
        activation1,
        weights10,
        weights11,
        weights12,
        weights13,
        resident_probabilities,
        resident_probabilities,
        v0,
        v1,
        0,
        1
    );

    const mm_stream_8x64_activation_packet_t payload0 = activation0.read();
    const mm_stream_8x64_activation_packet_t payload1 = activation1.read();
    (void)weights00.read();
    (void)weights01.read();
    (void)weights02.read();
    (void)weights03.read();
    (void)weights10.read();
    (void)weights11.read();
    (void)weights12.read();
    (void)weights13.read();
    for (unsigned int row = 0; row < MM_STREAM_8X64_TOKENS; row++) {
        ap_uint<fm_t::width> expected_raw = 0;
        if (row < GQA_GROUP_SIZE) {
            expected_raw = resident_probabilities.value[row][0].range(
                attention_prob_t::width - 1,
                0
            );
        }
        const ap_uint<fm_t::width> actual0 = payload0.data[row].range(
            fm_t::width - 1,
            0
        );
        const ap_uint<fm_t::width> actual1 = payload1.data[row].range(
            fm_t::width - 1,
            0
        );
        if (actual0 != expected_raw || actual1 != expected_raw) {
            if (errors < 8) {
                std::printf("PV payload mismatch row=%u\n", row);
            }
            errors++;
        }
    }

    std::printf(
        "RESIDENT PROBABILITY BUFFER TEST %s errors=%d\n",
        errors == 0 ? "PASS" : "FAIL",
        errors
    );
    return errors == 0 ? 0 : 1;
}
