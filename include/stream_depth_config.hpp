#ifndef LLM_FPGA_STREAM_DEPTH_CONFIG_HPP
#define LLM_FPGA_STREAM_DEPTH_CONFIG_HPP

#include "mm_stream_8x64.hpp"

#ifndef CC8_NK_TASK_STREAM_DEPTH
#define CC8_NK_TASK_STREAM_DEPTH 2
#endif

#ifndef CC8_NK_DATA_STREAM_DEPTH
#define CC8_NK_DATA_STREAM_DEPTH 16
#endif

#ifndef CC8_NK_STATUS_STREAM_DEPTH
#define CC8_NK_STATUS_STREAM_DEPTH 2
#endif

#ifndef CU8_NK_TASK_STREAM_DEPTH
#define CU8_NK_TASK_STREAM_DEPTH 3
#endif

#ifndef CU8_NK_DATA_STREAM_DEPTH
#define CU8_NK_DATA_STREAM_DEPTH 16
#endif

#ifndef CU8_UNIFIED_ACCUM_STREAM_DEPTH
#define CU8_UNIFIED_ACCUM_STREAM_DEPTH MM_STREAM_8X64_PACKETS_PER_BLOCK
#endif

#ifndef CU8_UNIFIED_CONVERTED_STREAM_DEPTH
#define CU8_UNIFIED_CONVERTED_STREAM_DEPTH 2
#endif

constexpr unsigned int CC8_NK_TASK_STREAM_DEPTH_VALUE =
    CC8_NK_TASK_STREAM_DEPTH;
constexpr unsigned int CC8_NK_DATA_STREAM_DEPTH_VALUE =
    CC8_NK_DATA_STREAM_DEPTH;
constexpr unsigned int CC8_NK_STATUS_STREAM_DEPTH_VALUE =
    CC8_NK_STATUS_STREAM_DEPTH;
constexpr unsigned int CU8_NK_TASK_STREAM_DEPTH_VALUE =
    CU8_NK_TASK_STREAM_DEPTH;
constexpr unsigned int CU8_NK_DATA_STREAM_DEPTH_VALUE =
    CU8_NK_DATA_STREAM_DEPTH;
constexpr unsigned int CU8_UNIFIED_ACCUM_STREAM_DEPTH_VALUE =
    CU8_UNIFIED_ACCUM_STREAM_DEPTH;
constexpr unsigned int CU8_UNIFIED_CONVERTED_STREAM_DEPTH_VALUE =
    CU8_UNIFIED_CONVERTED_STREAM_DEPTH;

static_assert(CC8_NK_TASK_STREAM_DEPTH_VALUE >= 2,
    "controller task/result streams need at least two entries");
static_assert(CC8_NK_DATA_STREAM_DEPTH_VALUE >= 2,
    "controller data streams need at least two entries");
static_assert(CC8_NK_STATUS_STREAM_DEPTH_VALUE >= 2,
    "controller status stream needs at least two entries");
static_assert(CU8_NK_TASK_STREAM_DEPTH_VALUE >= 2,
    "compute task/result streams need at least two entries");
static_assert(CU8_NK_DATA_STREAM_DEPTH_VALUE >= 2,
    "compute data streams need at least two entries");
static_assert(CU8_UNIFIED_ACCUM_STREAM_DEPTH_VALUE >= 2,
    "unified compute accum stream needs at least two entries");
static_assert(CU8_UNIFIED_CONVERTED_STREAM_DEPTH_VALUE >= 2,
    "unified compute converted stream needs at least two entries");

#endif
