#include "../kernel/control_cache_8x64.cpp"

#include <cstdio>

static fm_word_t kv_cache_k[CC8_KV_CACHE_WORDS];
static fm_word_t kv_cache_v[CC8_KV_CACHE_WORDS];

static fm_t panel_value(
    unsigned int position,
    unsigned int kv_head,
    unsigned int elem,
    unsigned int salt
) {
    const int raw =
        int((position * 11 + kv_head * 19 + elem * 3 + salt) % 97) - 48;
    return fm_t(raw) / fm_t(16);
}

static void set_current_value(
    cc8_current_kv_words_t& words,
    unsigned int kv_head,
    unsigned int elem,
    fm_t value
) {
    const unsigned int word_idx = elem / FM_BLOCK_SIZE;
    const unsigned int lane = elem % FM_BLOCK_SIZE;
    fm_word_t word = words.word[kv_head][word_idx];
    set_fm_word_lane(word, lane, value);
    words.word[kv_head][word_idx] = word;
}

int main() {
    constexpr unsigned int kLayer = 0;
    constexpr unsigned int kTileBegin = 0;
    constexpr unsigned int kTileLen = 17;

    for (unsigned int word = 0; word < CC8_KV_CACHE_WORDS; word++) {
        kv_cache_k[word] = 0;
        kv_cache_v[word] = 0;
    }

    for (unsigned int position = 0; position < kTileLen; position++) {
        cc8_current_kv_words_t current_k;
        cc8_current_kv_words_t current_v;
        clear_cc8_current_kv_words(current_k);
        clear_cc8_current_kv_words(current_v);
        for (unsigned int kv_head = 0;
             kv_head < NUM_KEY_VALUE_HEADS;
             kv_head++) {
            for (unsigned int elem = 0; elem < HEAD_DIM; elem++) {
                set_current_value(
                    current_k,
                    kv_head,
                    elem,
                    panel_value(position, kv_head, elem, 5)
                );
                set_current_value(
                    current_v,
                    kv_head,
                    elem,
                    panel_value(position, kv_head, elem, 23)
                );
            }
        }
        store_cc8_current_kv_to_cache(
            kv_cache_k,
            kv_cache_v,
            current_k,
            current_v,
            kLayer,
            position
        );
        store_cc8_current_k_to_transposed_cache(
            kv_cache_k,
            current_k,
            kLayer,
            position
        );
    }

    cc8_attention_qk_packet_panel_t k_panel0;
    cc8_attention_qk_packet_panel_t k_panel1;
    cc8_attention_pv_packet_panel_t v_panel0;
    cc8_attention_pv_packet_panel_t v_panel1;
    load_cc8_k_attention_packet_panel(
        k_panel0,
        k_panel1,
        kv_cache_k,
        kLayer,
        kTileBegin,
        kTileLen
    );
    load_cc8_v_attention_packet_panel(
        v_panel0,
        v_panel1,
        kv_cache_v,
        kLayer,
        kTileBegin,
        kTileLen
    );

    int errors = 0;
    for (unsigned int kv_head = 0;
         kv_head < NUM_KEY_VALUE_HEADS;
         kv_head++) {
        const cc8_attention_qk_packet_panel_t& k_panel =
            kv_head == 0 ? k_panel0 : k_panel1;
        const cc8_attention_pv_packet_panel_t& v_panel =
            kv_head == 0 ? v_panel0 : v_panel1;

        for (unsigned int elem = 0; elem < HEAD_DIM; elem++) {
            for (unsigned int position = 0;
                 position < CC8_ATTN_TILE;
                 position++) {
                const unsigned int group = position / CU_VEC_LANES;
                const unsigned int lane = position % CU_VEC_LANES;
                const mm_stream_8x64_weight_packet_t packet =
                    unpack_cc8_attention_weight_packet(
                        k_panel.packet[group][elem]
                    );
                const wt_linear_t expected =
                    position < kTileLen ?
                    wt_linear_t(panel_value(position, kv_head, elem, 5)) :
                    wt_linear_t(0);
                if (packet.data[lane] != expected) {
                    if (errors < 8) {
                        std::printf(
                            "K packet mismatch head=%u pos=%u elem=%u\n",
                            kv_head,
                            position,
                            elem
                        );
                    }
                    errors++;
                }
            }
        }

        for (unsigned int output_wave = 0;
             output_wave < CC8_ATTN_PV_WAVES;
             output_wave++) {
            for (unsigned int position = 0;
                 position < CC8_ATTN_TILE;
                 position++) {
                const unsigned int panel_index =
                    output_wave * CC8_ATTN_TILE + position;
                for (unsigned int group = 0;
                     group < CC8_ATTN_PACKET_GROUPS;
                     group++) {
                    const mm_stream_8x64_weight_packet_t packet =
                        unpack_cc8_attention_weight_packet(
                            v_panel.packet[group][panel_index]
                        );
                    for (unsigned int lane = 0;
                         lane < CU_VEC_LANES;
                         lane++) {
                        const unsigned int elem =
                            output_wave * MM_STREAM_8X64_OUTPUTS +
                            group * CU_VEC_LANES +
                            lane;
                        const wt_linear_t expected =
                            position < kTileLen && elem < HEAD_DIM ?
                            wt_linear_t(
                                panel_value(position, kv_head, elem, 23)
                            ) :
                            wt_linear_t(0);
                        if (packet.data[lane] != expected) {
                            if (errors < 8) {
                                std::printf(
                                    "V packet mismatch head=%u wave=%u pos=%u group=%u lane=%u\n",
                                    kv_head,
                                    output_wave,
                                    position,
                                    group,
                                    lane
                                );
                            }
                            errors++;
                        }
                    }
                }
            }
        }
    }

    std::printf(
        "ATTENTION PACKET PANEL TEST %s errors=%d\n",
        errors == 0 ? "PASS" : "FAIL",
        errors
    );
    return errors == 0 ? 0 : 1;
}
