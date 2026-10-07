#ifndef LLM_FPGA_QUANTIZED_BATCH_DECODE_TASK_HPP
#define LLM_FPGA_QUANTIZED_BATCH_DECODE_TASK_HPP

#include "hardware.hpp"

#include <cstdint>

// One descriptor describes one independent decode sequence.  The descriptor
// is intentionally separate from quantized_layer_task_word_t: the latter is a
// per-wave compute ABI and remains unchanged for existing xclbins.
constexpr unsigned int QUANTIZED_BATCH_DECODE_MAX = 8;
constexpr unsigned int QUANTIZED_BATCH_DECODE_DESCRIPTOR_BITS = 512;
using quantized_batch_decode_word_t = ap_uint<512>;

// Offsets are measured in mm_input_block_t words.  A KV offset is the base of
// one complete sequence cache, with a fixed layer/position layout:
//
//   base + layer * (MAX_SEQ_LEN * KV_WORDS) + position * KV_WORDS
//
// Keeping these values in the ABI header makes the reservation size and the
// address calculation agree for both the W4 and W8 controller paths.
constexpr unsigned int QUANTIZED_BATCH_DECODE_HIDDEN_WORDS =
    (HIDDEN_SIZE + MM_PE_IN - 1) / MM_PE_IN;
constexpr unsigned int QUANTIZED_BATCH_DECODE_KV_WORDS =
    (KV_CHANNELS + MM_PE_OUT - 1) / MM_PE_OUT;
constexpr unsigned int QUANTIZED_BATCH_DECODE_KV_LAYER_WORDS =
    MAX_SEQ_LEN * QUANTIZED_BATCH_DECODE_KV_WORDS;
constexpr unsigned int QUANTIZED_BATCH_DECODE_FULL_KV_WORDS =
    NUM_LAYERS * QUANTIZED_BATCH_DECODE_KV_LAYER_WORDS;

static_assert(QUANTIZED_BATCH_DECODE_MAX == 8,
              "batch decode ABI has eight descriptor slots");
static_assert(QUANTIZED_BATCH_DECODE_HIDDEN_WORDS > 0,
              "batch decode hidden rows must contain a word");
static_assert(QUANTIZED_BATCH_DECODE_KV_WORDS > 0,
              "batch decode KV rows must contain a word");
static_assert(QUANTIZED_BATCH_DECODE_DESCRIPTOR_BITS == 512,
              "batch decode descriptors are fixed at 512 bits");

struct quantized_batch_decode_entry_t {
    std::uint32_t sequence_id;
    std::uint32_t position;
    std::uint32_t kv_context_length;
    std::uint32_t hidden_input_offset;
    std::uint32_t hidden_output_offset;
    std::uint32_t key_cache_offset;
    std::uint32_t value_cache_offset;
};

static_assert(sizeof(quantized_batch_decode_entry_t) == 7 * sizeof(std::uint32_t),
              "batch decode entry fields must remain seven uint32 values");

// Validation status values are stable ABI-facing integers.  Keep PASS/OK
// aliases because host code historically uses both spellings for status zero.
enum quantized_batch_decode_status_t {
    QUANTIZED_BATCH_DECODE_STATUS_PASS = 0,
    QUANTIZED_BATCH_DECODE_STATUS_OK = QUANTIZED_BATCH_DECODE_STATUS_PASS,
    QUANTIZED_BATCH_DECODE_PASS = QUANTIZED_BATCH_DECODE_STATUS_PASS,
    QUANTIZED_BATCH_DECODE_OK = QUANTIZED_BATCH_DECODE_STATUS_PASS,

    QUANTIZED_BATCH_DECODE_STATUS_BAD_BATCH_COUNT = 1,
    QUANTIZED_BATCH_DECODE_STATUS_INVALID_BATCH_COUNT =
        QUANTIZED_BATCH_DECODE_STATUS_BAD_BATCH_COUNT,
    QUANTIZED_BATCH_DECODE_STATUS_BAD_LAYER = 2,
    QUANTIZED_BATCH_DECODE_STATUS_INVALID_LAYER =
        QUANTIZED_BATCH_DECODE_STATUS_BAD_LAYER,
    QUANTIZED_BATCH_DECODE_STATUS_RESERVED_BITS = 3,
    QUANTIZED_BATCH_DECODE_STATUS_BAD_RESERVED_BITS =
        QUANTIZED_BATCH_DECODE_STATUS_RESERVED_BITS,
    QUANTIZED_BATCH_DECODE_STATUS_BAD_POSITION = 4,
    QUANTIZED_BATCH_DECODE_STATUS_INVALID_POSITION =
        QUANTIZED_BATCH_DECODE_STATUS_BAD_POSITION,
    QUANTIZED_BATCH_DECODE_STATUS_HIDDEN_INPUT_BOUNDS = 5,
    QUANTIZED_BATCH_DECODE_STATUS_INPUT_BOUNDS =
        QUANTIZED_BATCH_DECODE_STATUS_HIDDEN_INPUT_BOUNDS,
    QUANTIZED_BATCH_DECODE_STATUS_HIDDEN_OUTPUT_BOUNDS = 6,
    QUANTIZED_BATCH_DECODE_STATUS_OUTPUT_BOUNDS =
        QUANTIZED_BATCH_DECODE_STATUS_HIDDEN_OUTPUT_BOUNDS,
    QUANTIZED_BATCH_DECODE_STATUS_KEY_CACHE_BOUNDS = 7,
    QUANTIZED_BATCH_DECODE_STATUS_VALUE_CACHE_BOUNDS = 8,
    QUANTIZED_BATCH_DECODE_STATUS_DUPLICATE_SEQUENCE_ID = 9,
    QUANTIZED_BATCH_DECODE_STATUS_INPUT_OVERLAP = 10,
    QUANTIZED_BATCH_DECODE_STATUS_OUTPUT_OVERLAP = 11,
    QUANTIZED_BATCH_DECODE_STATUS_KEY_CACHE_OVERLAP = 12,
    QUANTIZED_BATCH_DECODE_STATUS_VALUE_CACHE_OVERLAP = 13
};

inline quantized_batch_decode_word_t pack_quantized_batch_decode_entry(
    const quantized_batch_decode_entry_t& entry) {
    #pragma HLS inline
    quantized_batch_decode_word_t word = 0;
    word.range(31, 0) = entry.sequence_id;
    word.range(63, 32) = entry.position;
    word.range(95, 64) = entry.kv_context_length;
    word.range(127, 96) = entry.hidden_input_offset;
    word.range(159, 128) = entry.hidden_output_offset;
    word.range(191, 160) = entry.key_cache_offset;
    word.range(223, 192) = entry.value_cache_offset;
    return word;
}

// Bits [511:224] are reserved and must be checked by the caller before the
// decoded fields are consumed.  The validator below performs that check for
// an entire batch before unpacking any entry for use by a controller.
inline bool quantized_batch_decode_reserved_bits_zero(
    const quantized_batch_decode_word_t& word) {
    #pragma HLS inline
    return word.range(511, 224) == 0;
}

inline quantized_batch_decode_entry_t unpack_quantized_batch_decode_entry(
    const quantized_batch_decode_word_t& word) {
    #pragma HLS inline
    quantized_batch_decode_entry_t entry{};
    entry.sequence_id = word.range(31, 0).to_uint();
    entry.position = word.range(63, 32).to_uint();
    entry.kv_context_length = word.range(95, 64).to_uint();
    entry.hidden_input_offset = word.range(127, 96).to_uint();
    entry.hidden_output_offset = word.range(159, 128).to_uint();
    entry.key_cache_offset = word.range(191, 160).to_uint();
    entry.value_cache_offset = word.range(223, 192).to_uint();
    return entry;
}

// A status-returning form is useful at ABI boundaries that want to reject a
// malformed word without exposing reserved fields to the controller.
inline bool try_unpack_quantized_batch_decode_entry(
    const quantized_batch_decode_word_t& word,
    quantized_batch_decode_entry_t& entry) {
    #pragma HLS inline
    if (!quantized_batch_decode_reserved_bits_zero(word)) return false;
    entry = unpack_quantized_batch_decode_entry(word);
    return true;
}

// Return the word offset of one KV row within an entry's full cache base.
inline unsigned int quantized_batch_decode_kv_word_offset(
    unsigned int cache_base,
    unsigned int layer,
    unsigned int position,
    unsigned int kv_block = 0) {
    #pragma HLS inline
    return cache_base + layer * QUANTIZED_BATCH_DECODE_KV_LAYER_WORDS +
        position * QUANTIZED_BATCH_DECODE_KV_WORDS + kv_block;
}

inline bool quantized_batch_decode_ranges_overlap(
    std::uint64_t first_begin,
    std::uint64_t first_size,
    std::uint64_t second_begin,
    std::uint64_t second_size) {
    #pragma HLS inline
    return first_begin < second_begin + second_size &&
        second_begin < first_begin + first_size;
}

// Validate all descriptors before the first controller task is issued.  The
// input/output arrays contain one hidden row per descriptor.  Each KV offset
// reserves a complete sequence cache, not merely the row appended by this
// decode, so that all layer and position rows remain within that entry's
// allocation.
inline unsigned int validate_quantized_batch_decode(
    const quantized_batch_decode_word_t descriptors[8],
    unsigned int batch_count,
    unsigned int layer,
    unsigned int input_words,
    unsigned int output_words,
    unsigned int key_words,
    unsigned int value_words) {
    #pragma HLS inline off
    if (batch_count == 0 || batch_count > QUANTIZED_BATCH_DECODE_MAX) {
        return QUANTIZED_BATCH_DECODE_STATUS_BAD_BATCH_COUNT;
    }
    if (layer >= NUM_LAYERS) {
        return QUANTIZED_BATCH_DECODE_STATUS_BAD_LAYER;
    }

    // Check reserved bits before decoding fields for controller use.
    for (unsigned int i = 0; i < batch_count; ++i) {
        #pragma HLS pipeline II=1
        if (!quantized_batch_decode_reserved_bits_zero(descriptors[i])) {
            return QUANTIZED_BATCH_DECODE_STATUS_RESERVED_BITS;
        }
    }

    quantized_batch_decode_entry_t entries[QUANTIZED_BATCH_DECODE_MAX];
    for (unsigned int i = 0; i < batch_count; ++i) {
        #pragma HLS pipeline II=1
        entries[i] = unpack_quantized_batch_decode_entry(descriptors[i]);

        const quantized_batch_decode_entry_t& entry = entries[i];
        if (entry.position >= MAX_SEQ_LEN ||
            entry.position != entry.kv_context_length) {
            return QUANTIZED_BATCH_DECODE_STATUS_BAD_POSITION;
        }

        // Widen before addition so malformed uint32 offsets cannot wrap.
        const std::uint64_t input_end =
            std::uint64_t(entry.hidden_input_offset) +
            std::uint64_t(QUANTIZED_BATCH_DECODE_HIDDEN_WORDS);
        if (input_end > std::uint64_t(input_words)) {
            return QUANTIZED_BATCH_DECODE_STATUS_HIDDEN_INPUT_BOUNDS;
        }

        const std::uint64_t output_end =
            std::uint64_t(entry.hidden_output_offset) +
            std::uint64_t(QUANTIZED_BATCH_DECODE_HIDDEN_WORDS);
        if (output_end > std::uint64_t(output_words)) {
            return QUANTIZED_BATCH_DECODE_STATUS_HIDDEN_OUTPUT_BOUNDS;
        }

        const std::uint64_t key_end =
            std::uint64_t(entry.key_cache_offset) +
            std::uint64_t(QUANTIZED_BATCH_DECODE_FULL_KV_WORDS);
        if (key_end > std::uint64_t(key_words)) {
            return QUANTIZED_BATCH_DECODE_STATUS_KEY_CACHE_BOUNDS;
        }

        const std::uint64_t value_end =
            std::uint64_t(entry.value_cache_offset) +
            std::uint64_t(QUANTIZED_BATCH_DECODE_FULL_KV_WORDS);
        if (value_end > std::uint64_t(value_words)) {
            return QUANTIZED_BATCH_DECODE_STATUS_VALUE_CACHE_BOUNDS;
        }
    }

    for (unsigned int i = 0; i < batch_count; ++i) {
        #pragma HLS pipeline II=1
        for (unsigned int j = i + 1; j < batch_count; ++j) {
            #pragma HLS loop_flatten
            if (entries[i].sequence_id == entries[j].sequence_id) {
                return QUANTIZED_BATCH_DECODE_STATUS_DUPLICATE_SEQUENCE_ID;
            }

            if (quantized_batch_decode_ranges_overlap(
                    entries[i].hidden_input_offset,
                    QUANTIZED_BATCH_DECODE_HIDDEN_WORDS,
                    entries[j].hidden_input_offset,
                    QUANTIZED_BATCH_DECODE_HIDDEN_WORDS)) {
                return QUANTIZED_BATCH_DECODE_STATUS_INPUT_OVERLAP;
            }
            if (quantized_batch_decode_ranges_overlap(
                    entries[i].hidden_output_offset,
                    QUANTIZED_BATCH_DECODE_HIDDEN_WORDS,
                    entries[j].hidden_output_offset,
                    QUANTIZED_BATCH_DECODE_HIDDEN_WORDS)) {
                return QUANTIZED_BATCH_DECODE_STATUS_OUTPUT_OVERLAP;
            }
            if (quantized_batch_decode_ranges_overlap(
                    entries[i].key_cache_offset,
                    QUANTIZED_BATCH_DECODE_FULL_KV_WORDS,
                    entries[j].key_cache_offset,
                    QUANTIZED_BATCH_DECODE_FULL_KV_WORDS)) {
                return QUANTIZED_BATCH_DECODE_STATUS_KEY_CACHE_OVERLAP;
            }
            if (quantized_batch_decode_ranges_overlap(
                    entries[i].value_cache_offset,
                    QUANTIZED_BATCH_DECODE_FULL_KV_WORDS,
                    entries[j].value_cache_offset,
                    QUANTIZED_BATCH_DECODE_FULL_KV_WORDS)) {
                return QUANTIZED_BATCH_DECODE_STATUS_VALUE_CACHE_OVERLAP;
            }
        }
    }

    return QUANTIZED_BATCH_DECODE_STATUS_PASS;
}

#endif
