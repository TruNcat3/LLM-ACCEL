#ifndef LLM_FPGA_WEIGHT_PIPELINE_CONFIG_HPP
#define LLM_FPGA_WEIGHT_PIPELINE_CONFIG_HPP

#ifndef CC8_ENABLE_WEIGHT_TILE_PIPELINE
#define CC8_ENABLE_WEIGHT_TILE_PIPELINE 1
#endif

#ifndef CC8_WEIGHT_TILE_FIFO_DEPTH
#define CC8_WEIGHT_TILE_FIFO_DEPTH 2
#endif

#ifndef CC8_WEIGHT_TILE_LOAD_II
#define CC8_WEIGHT_TILE_LOAD_II 2
#endif

#ifndef CC8_ENABLE_MM_WAVE_REPEAT
#define CC8_ENABLE_MM_WAVE_REPEAT 0
#endif

// Keep the packed K-tile ping-pong FIFOs alive for the complete projection
// instead of rebuilding a DATAFLOW region for every output wave.  This lets
// the HBM loaders move into wave n+1 while the stream emit/compute/result
// stages are still retiring wave n.
#ifndef CC8_ENABLE_MM_CROSS_WAVE_DATAFLOW
#define CC8_ENABLE_MM_CROSS_WAVE_DATAFLOW 1
#endif

// The banked projection path completes its dataflow producer before starting
// the sequential GBUF commit.  Vitis HLS BRAM FIFOs reserve one requested
// slot for their output register, but that register is not prefetched while
// the downstream commit function is disabled.  Keep one physical RAM slot
// beyond the 32 packets produced by each compute core.
#ifndef CC8_MM_WAVE_RESULT_FIFO_DEPTH
#define CC8_MM_WAVE_RESULT_FIFO_DEPTH \
    (MM_STREAM_8X64_PACKETS_PER_BLOCK + 1)
#endif

constexpr bool CC8_WEIGHT_TILE_PIPELINE_ENABLED =
    CC8_ENABLE_WEIGHT_TILE_PIPELINE != 0;
constexpr unsigned int CC8_WEIGHT_TILE_FIFO_DEPTH_VALUE =
    CC8_WEIGHT_TILE_FIFO_DEPTH;
constexpr unsigned int CC8_WEIGHT_TILE_LOAD_II_VALUE =
    CC8_WEIGHT_TILE_LOAD_II;
constexpr bool CC8_MM_WAVE_REPEAT_ENABLED =
    CC8_ENABLE_MM_WAVE_REPEAT != 0;
constexpr bool CC8_MM_CROSS_WAVE_DATAFLOW_ENABLED =
    CC8_ENABLE_MM_CROSS_WAVE_DATAFLOW != 0;
constexpr unsigned int CC8_MM_WAVE_RESULT_FIFO_DEPTH_VALUE =
    CC8_MM_WAVE_RESULT_FIFO_DEPTH;
static_assert(
    CC8_WEIGHT_TILE_FIFO_DEPTH_VALUE == 1 ||
    CC8_WEIGHT_TILE_FIFO_DEPTH_VALUE == 2 ||
    CC8_WEIGHT_TILE_FIFO_DEPTH_VALUE == 4 ||
    CC8_WEIGHT_TILE_FIFO_DEPTH_VALUE == 8,
    "CC8 packed weight-tile FIFO depth must be 1, 2, 4, or 8"
);
static_assert(
    CC8_WEIGHT_TILE_LOAD_II_VALUE == 1 ||
    CC8_WEIGHT_TILE_LOAD_II_VALUE == 2 ||
    CC8_WEIGHT_TILE_LOAD_II_VALUE == 4 ||
    CC8_WEIGHT_TILE_LOAD_II_VALUE == 8,
    "CC8 packed weight-block loader II must be 1, 2, 4, or 8"
);
static_assert(
    CC8_MM_WAVE_RESULT_FIFO_DEPTH_VALUE >=
        MM_STREAM_8X64_PACKETS_PER_BLOCK + 1,
    "CC8 banked projection result FIFO needs one slot beyond a full wave"
);
#endif
