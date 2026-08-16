#include "../kernel/control_cache_8x64.cpp"

#include <cstdio>

static fm_word_t kv_cache_k[CC8_KV_CACHE_WORDS];
static fm_word_t norm_words[CC8_DATA_PORT_WORDS];
static fm_word_t rope_words[CC8_DATA_PORT_WORDS];

static fm_t test_value(
    unsigned int major,
    unsigned int elem,
    unsigned int salt
) {
    const int raw = int((major * 29 + elem * 7 + salt) % 101) - 50;
    return fm_t(raw) / fm_t(16);
}

static void set_gbuf_value(
    cc8_global_buffer_t& gbuf,
    unsigned int elem,
    fm_t value
) {
    const unsigned int block = elem / MM_PE_IN;
    const unsigned int lane = elem % MM_PE_IN;
    mm_input_block_t packed = gbuf.block[0][block];
    set_mm_input_block_lane(packed, lane, value);
    gbuf.block[0][block] = packed;
}

static void set_aux_value(
    fm_word_t words[CC8_DATA_PORT_WORDS],
    unsigned int word_offset,
    unsigned int elem,
    fm_t value
) {
    const unsigned int word_idx = word_offset + elem / FM_BLOCK_SIZE;
    const unsigned int lane = elem % FM_BLOCK_SIZE;
    fm_word_t word = words[word_idx];
    set_fm_word_lane(word, lane, value);
    words[word_idx] = word;
}

static fm_t get_gbuf_value(
    const cc8_global_buffer_t& gbuf,
    unsigned int elem
) {
    const unsigned int block = elem / MM_PE_IN;
    const unsigned int lane = elem % MM_PE_IN;
    return unpack_mm_input_block_lane(gbuf.block[0][block], lane);
}

int main() {
    cc8_global_buffer_t source;
    for (unsigned int block = 0; block < CC8_GBUF_BLOCKS; block++) {
        source.block[0][block] = 0;
    }
    for (unsigned int word = 0; word < CC8_DATA_PORT_WORDS; word++) {
        norm_words[word] = 0;
        rope_words[word] = 0;
    }
    for (unsigned int word = 0; word < CC8_KV_CACHE_WORDS; word++) {
        kv_cache_k[word] = 0;
    }

    int errors = 0;

    // Give the two layer-1 norm rows and the final norm row distinct values.
    // Loading them through the same offset helpers used by Tasks 18, 19, and
    // 20 catches row-aliasing in the persistent auxiliary-HBM layout.
    constexpr unsigned int kNormLayer = NUM_LAYERS > 1 ? 1 : 0;
    const unsigned int norm_offsets[3] = {
        cc8_attention_norm_word_offset(kNormLayer),
        cc8_ffn_norm_word_offset(kNormLayer),
        cc8_final_norm_word_offset()
    };
    const fm_t norm_expected[3] = {
        fm_t(0.5),
        fm_t(1.25),
        fm_t(-0.75)
    };
    for (unsigned int row = 0; row < 3; row++) {
        for (unsigned int elem = 0; elem < HIDDEN_SIZE; elem++) {
            set_aux_value(
                norm_words,
                norm_offsets[row],
                elem,
                norm_expected[row]
            );
        }
    }
    for (unsigned int row = 0; row < 3; row++) {
        cc8_global_buffer_t loaded_norm;
        for (unsigned int block = 0; block < CC8_GBUF_BLOCKS; block++) {
            loaded_norm.block[0][block] = 0;
        }
        load_cc8_feature_gbuf(
            loaded_norm,
            norm_words,
            norm_words,
            1,
            1,
            HIDDEN_SIZE,
            norm_offsets[row]
        );
        for (unsigned int elem = 0; elem < HIDDEN_SIZE; elem++) {
            if (get_gbuf_value(loaded_norm, elem) != norm_expected[row]) {
                if (errors < 8) {
                    std::printf(
                        "norm row mismatch row=%u elem=%u\n",
                        row,
                        elem
                    );
                }
                errors++;
            }
        }
    }

    fm_t query_source[NUM_ATTENTION_HEADS][HEAD_DIM];
    for (unsigned int head = 0; head < NUM_ATTENTION_HEADS; head++) {
        for (unsigned int elem = 0; elem < HEAD_DIM; elem++) {
            query_source[head][elem] = test_value(head, elem, 3);
            set_gbuf_value(
                source,
                head * HEAD_DIM + elem,
                query_source[head][elem]
            );
        }
    }

    cc8_resident_query_buffer_t query0;
    cc8_resident_query_buffer_t query1;
    extract_cc8_resident_query_group_from_gbuf(query0, source, 0, 0);
    extract_cc8_resident_query_group_from_gbuf(
        query1,
        source,
        0,
        GQA_GROUP_SIZE
    );

    for (unsigned int block = 0; block < CC8_GBUF_BLOCKS; block++) {
        source.block[0][block] = 0;
    }
    fm_t key_source[NUM_KEY_VALUE_HEADS][HEAD_DIM];
    for (unsigned int head = 0; head < NUM_KEY_VALUE_HEADS; head++) {
        for (unsigned int elem = 0; elem < HEAD_DIM; elem++) {
            key_source[head][elem] = test_value(head, elem, 41);
            set_gbuf_value(
                source,
                head * HEAD_DIM + elem,
                key_source[head][elem]
            );
        }
    }
    cc8_resident_k_buffer_t current_k;
    extract_cc8_resident_k_from_gbuf(current_k, source, 0);

    fm_t cosine[CC8_ROPE_HALF_ELEMS];
    fm_t sine[CC8_ROPE_HALF_ELEMS];
    constexpr unsigned int position = 3;
    const unsigned int position_word_offset =
        position * CC8_ROPE_POSITION_WORDS;
    for (unsigned int i = 0; i < CC8_ROPE_HALF_ELEMS; i++) {
        cosine[i] = fm_t(0.75) + fm_t(i % 3) / fm_t(32);
        sine[i] = fm_t(-0.25) + fm_t(i % 5) / fm_t(64);
        set_aux_value(
            rope_words,
            position_word_offset + CC8_ROPE_COS_WORD_OFFSET,
            i,
            cosine[i]
        );
        set_aux_value(
            rope_words,
            position_word_offset + CC8_ROPE_SIN_WORD_OFFSET,
            i,
            sine[i]
        );
    }

    apply_cc8_rope_from_aux(
        query0,
        query1,
        current_k,
        rope_words,
        position
    );

    for (unsigned int head = 0; head < NUM_ATTENTION_HEADS; head++) {
        const bool second_group = head >= GQA_GROUP_SIZE;
        const unsigned int row = second_group ?
            head - GQA_GROUP_SIZE : head;
        const cc8_resident_query_buffer_t& query =
            second_group ? query1 : query0;
        for (unsigned int i = 0; i < CC8_ROPE_HALF_ELEMS; i++) {
            const fm_t low = query_source[head][i];
            const fm_t high =
                query_source[head][CC8_ROPE_HALF_ELEMS + i];
            const fm_t expected_low =
                fm_t(low * cosine[i] - high * sine[i]);
            const fm_t expected_high =
                fm_t(high * cosine[i] + low * sine[i]);
            if (query.value[row][0][i] != expected_low ||
                query.value[row][1][i] != expected_high) {
                if (errors < 8) {
                    std::printf(
                        "query mismatch head=%u i=%u\n",
                        head,
                        i
                    );
                }
                errors++;
            }
        }
    }

    constexpr unsigned int kLayer = 0;
    constexpr unsigned int kPosition = 3;
    store_cc8_resident_current_k_row_major(
        kv_cache_k,
        current_k,
        kLayer,
        kPosition
    );
    store_cc8_resident_current_k_transposed(
        kv_cache_k,
        current_k,
        kLayer,
        kPosition
    );

    for (unsigned int head = 0; head < NUM_KEY_VALUE_HEADS; head++) {
        for (unsigned int i = 0; i < CC8_ROPE_HALF_ELEMS; i++) {
            const fm_t low = key_source[head][i];
            const fm_t high =
                key_source[head][CC8_ROPE_HALF_ELEMS + i];
            const fm_t expected[2] = {
                fm_t(low * cosine[i] - high * sine[i]),
                fm_t(high * cosine[i] + low * sine[i])
            };
            for (unsigned int bank = 0; bank < 2; bank++) {
                const unsigned int elem =
                    bank * CC8_ROPE_HALF_ELEMS + i;
                const unsigned int row_word = elem / FM_BLOCK_SIZE;
                const unsigned int row_lane = elem % FM_BLOCK_SIZE;
                const fm_t row_value = unpack_fm_word_lane(
                    kv_cache_k[cc8_kv_cache_word_index(
                        kLayer,
                        kPosition,
                        head,
                        row_word
                    )],
                    row_lane
                );
                const fm_t transposed_value = unpack_fm_word_lane(
                    kv_cache_k[cc8_k_cache_transposed_word_index(
                        kLayer,
                        head,
                        elem,
                        kPosition / FM_BLOCK_SIZE
                    )],
                    kPosition % FM_BLOCK_SIZE
                );
                if (row_value != expected[bank] ||
                    transposed_value != expected[bank]) {
                    if (errors < 8) {
                        std::printf(
                            "K cache mismatch head=%u elem=%u\n",
                            head,
                            elem
                        );
                    }
                    errors++;
                }
            }
        }
    }

    std::printf(
        "RESIDENT ROPE BANKED CACHE TEST %s errors=%d norm_rows=3\n",
        errors == 0 ? "PASS" : "FAIL",
        errors
    );
    return errors == 0 ? 0 : 1;
}
