#ifndef LLM_FPGA_QUANTIZED_ATTENTION_CONFIG_HPP
#define LLM_FPGA_QUANTIZED_ATTENTION_CONFIG_HPP

// Keep the established schedule as the default until matched RTL gates pass.
#ifndef QUANTIZED_ATTENTION_VALID_TAIL
#define QUANTIZED_ATTENTION_VALID_TAIL 0
#endif
static_assert(QUANTIZED_ATTENTION_VALID_TAIL == 0 ||
              QUANTIZED_ATTENTION_VALID_TAIL == 1,
              "QUANTIZED_ATTENTION_VALID_TAIL must be 0 or 1");

// Tile-level packed-K/V producer/consumer overlap is an opt-in candidate.
// Keeping the established serial schedule as the default makes every
// existing kernel identity reproducible while the bounded packed-word FIFO
// is checked with deadlock-enabled CoSim.
#ifndef QUANTIZED_ATTENTION_TILE_PREFETCH
#define QUANTIZED_ATTENTION_TILE_PREFETCH 0
#endif
static_assert(QUANTIZED_ATTENTION_TILE_PREFETCH == 0 ||
              QUANTIZED_ATTENTION_TILE_PREFETCH == 1,
              "QUANTIZED_ATTENTION_TILE_PREFETCH must be 0 or 1");

// Independent experimental controls. Defaults preserve reference behavior.
#ifndef QUANTIZED_ATTENTION_PROBABILITY_FUSE
#define QUANTIZED_ATTENTION_PROBABILITY_FUSE 0
#endif
#ifndef QUANTIZED_ATTENTION_PACK_LANES
#define QUANTIZED_ATTENTION_PACK_LANES 1
#endif
#ifndef QUANTIZED_ATTENTION_ACTIVE_STATE
#define QUANTIZED_ATTENTION_ACTIVE_STATE 0
#endif
#ifndef QUANTIZED_ATTENTION_HEAD_LANES
#define QUANTIZED_ATTENTION_HEAD_LANES 1
#endif
#ifndef QUANTIZED_ATTENTION_WAVE_PIPELINE
#define QUANTIZED_ATTENTION_WAVE_PIPELINE 0
#endif
static_assert(QUANTIZED_ATTENTION_PROBABILITY_FUSE == 0 || QUANTIZED_ATTENTION_PROBABILITY_FUSE == 1,
              "PROBABILITY_FUSE must be boolean");
static_assert(QUANTIZED_ATTENTION_ACTIVE_STATE == 0 || QUANTIZED_ATTENTION_ACTIVE_STATE == 1,
              "ACTIVE_STATE must be boolean");
static_assert(QUANTIZED_ATTENTION_WAVE_PIPELINE == 0 || QUANTIZED_ATTENTION_WAVE_PIPELINE == 1,
              "WAVE_PIPELINE must be boolean");
static_assert(QUANTIZED_ATTENTION_PACK_LANES == 1 || QUANTIZED_ATTENTION_PACK_LANES == 2 || QUANTIZED_ATTENTION_PACK_LANES == 4,
              "PACK_LANES must be 1, 2 or 4");
static_assert(QUANTIZED_ATTENTION_HEAD_LANES == 1 || QUANTIZED_ATTENTION_HEAD_LANES == 2 || QUANTIZED_ATTENTION_HEAD_LANES == 4,
              "HEAD_LANES must be 1, 2 or 4");
static_assert(QUANTIZED_ATTENTION_TILE_PREFETCH ||
              !(QUANTIZED_ATTENTION_PROBABILITY_FUSE || QUANTIZED_ATTENTION_ACTIVE_STATE ||
                QUANTIZED_ATTENTION_WAVE_PIPELINE || QUANTIZED_ATTENTION_PACK_LANES != 1 ||
                QUANTIZED_ATTENTION_HEAD_LANES != 1),
              "Attention optimization experiments require TILE_PREFETCH=1");

template <unsigned int Capacity>
inline unsigned int quantized_attention_prepared_positions(unsigned int length) {
    #pragma HLS inline
#if QUANTIZED_ATTENTION_VALID_TAIL
    return length < Capacity ? length : Capacity;
#else
    return Capacity;
#endif
}

#endif
