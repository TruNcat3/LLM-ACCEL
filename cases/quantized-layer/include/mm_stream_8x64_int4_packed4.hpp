#ifndef LLM_FPGA_MM_STREAM_8X64_INT4_PACKED4_HPP
#define LLM_FPGA_MM_STREAM_8X64_INT4_PACKED4_HPP

#include <ap_int.h>
#include <hls_stream.h>

// Feasibility probe for a true INT4 x INT4 2x2 outer product.  One DSP
// multiply produces four logical products: a0*w0, a0*w1, a1*w0, a1*w1.
// This is deliberately separate from the production weight-only INT8 x INT4
// path, whose 8-bit activation operand limits the safe digit packing factor.
#ifndef MM_STREAM_8X64_INT4_PACKED4_PHYSICAL_LANES_CONFIG
#define MM_STREAM_8X64_INT4_PACKED4_PHYSICAL_LANES_CONFIG 256
#endif

constexpr unsigned int MM_STREAM_8X64_INT4_PACKED4_PHYSICAL_LANES =
    MM_STREAM_8X64_INT4_PACKED4_PHYSICAL_LANES_CONFIG;
constexpr unsigned int MM_STREAM_8X64_INT4_PACKED4_PACKING_FACTOR = 4;
constexpr unsigned int MM_STREAM_8X64_INT4_PACKED4_PACKETS = 4;
constexpr unsigned int MM_STREAM_8X64_INT4_PACKED4_PRODUCT_BITS = 8;
constexpr unsigned int MM_STREAM_8X64_INT4_PACKED4_LOGICAL_PRODUCTS =
    MM_STREAM_8X64_INT4_PACKED4_PHYSICAL_LANES *
    MM_STREAM_8X64_INT4_PACKED4_PACKING_FACTOR;

static_assert(MM_STREAM_8X64_INT4_PACKED4_PHYSICAL_LANES > 0,
              "packed4 probe needs at least one physical lane");
static_assert(MM_STREAM_8X64_INT4_PACKED4_PACKING_FACTOR == 4,
              "packed4 probe must expose four products per DSP lane");

struct mm_stream_8x64_int4_packed4_input_t {
    ap_int<4> activation0[MM_STREAM_8X64_INT4_PACKED4_PHYSICAL_LANES];
    ap_int<4> activation1[MM_STREAM_8X64_INT4_PACKED4_PHYSICAL_LANES];
    ap_int<4> weight0[MM_STREAM_8X64_INT4_PACKED4_PHYSICAL_LANES];
    ap_int<4> weight1[MM_STREAM_8X64_INT4_PACKED4_PHYSICAL_LANES];
};

struct mm_stream_8x64_int4_packed4_output_t {
    ap_int<MM_STREAM_8X64_INT4_PACKED4_PRODUCT_BITS>
        product00[MM_STREAM_8X64_INT4_PACKED4_PHYSICAL_LANES];
    ap_int<MM_STREAM_8X64_INT4_PACKED4_PRODUCT_BITS>
        product01[MM_STREAM_8X64_INT4_PACKED4_PHYSICAL_LANES];
    ap_int<MM_STREAM_8X64_INT4_PACKED4_PRODUCT_BITS>
        product10[MM_STREAM_8X64_INT4_PACKED4_PHYSICAL_LANES];
    ap_int<MM_STREAM_8X64_INT4_PACKED4_PRODUCT_BITS>
        product11[MM_STREAM_8X64_INT4_PACKED4_PHYSICAL_LANES];
};

void compute_mm_stream_8x64_int4_packed4_core(
    hls::stream<mm_stream_8x64_int4_packed4_output_t>& out_stream,
    hls::stream<mm_stream_8x64_int4_packed4_input_t>& in_stream);

#endif
