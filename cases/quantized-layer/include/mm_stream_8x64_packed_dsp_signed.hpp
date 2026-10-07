#ifndef LLM_FPGA_MM_STREAM_8X64_PACKED_DSP_SIGNED_HPP
#define LLM_FPGA_MM_STREAM_8X64_PACKED_DSP_SIGNED_HPP

#include <ap_int.h>
#include <hls_stream.h>

// Signed symmetric-quantization variant of the packed-DSP feasibility probe.
// Magnitudes are packed; the two sign bits are restored after the multiply.
// The logical product count is intentionally independent from the number of
// physical DSP lanes: one packed lane evaluates two INT8xINT4 products.
#ifndef MM_STREAM_8X64_PACKED_DSP_SIGNED_PHYSICAL_LANES_CONFIG
#define MM_STREAM_8X64_PACKED_DSP_SIGNED_PHYSICAL_LANES_CONFIG 256
#endif
constexpr unsigned int MM_STREAM_8X64_PACKED_DSP_SIGNED_PACKING_FACTOR = 2;
constexpr unsigned int MM_STREAM_8X64_PACKED_DSP_SIGNED_PHYSICAL_LANES =
    MM_STREAM_8X64_PACKED_DSP_SIGNED_PHYSICAL_LANES_CONFIG;
constexpr unsigned int MM_STREAM_8X64_PACKED_DSP_SIGNED_LOGICAL_MACS_PER_PACKET =
    MM_STREAM_8X64_PACKED_DSP_SIGNED_PHYSICAL_LANES *
    MM_STREAM_8X64_PACKED_DSP_SIGNED_PACKING_FACTOR;
// Compatibility alias used by the existing probe and testbench.
constexpr unsigned int MM_STREAM_8X64_PACKED_DSP_SIGNED_LANES =
    MM_STREAM_8X64_PACKED_DSP_SIGNED_PHYSICAL_LANES;
constexpr unsigned int MM_STREAM_8X64_PACKED_DSP_SIGNED_PACKETS = 4;
constexpr unsigned int MM_STREAM_8X64_PACKED_DSP_SIGNED_PRODUCT_BITS = 12;

static_assert(MM_STREAM_8X64_PACKED_DSP_SIGNED_PACKING_FACTOR == 2,
              "the packed signed probe emits two products per physical lane");
static_assert(MM_STREAM_8X64_PACKED_DSP_SIGNED_PHYSICAL_LANES > 0,
              "the packed signed probe needs at least one physical lane");

struct mm_stream_8x64_packed_dsp_signed_input_t {
    ap_int<8> activation[MM_STREAM_8X64_PACKED_DSP_SIGNED_LANES];
    ap_int<4> weight0[MM_STREAM_8X64_PACKED_DSP_SIGNED_LANES];
    ap_int<4> weight1[MM_STREAM_8X64_PACKED_DSP_SIGNED_LANES];
};

struct mm_stream_8x64_packed_dsp_signed_output_t {
    ap_int<MM_STREAM_8X64_PACKED_DSP_SIGNED_PRODUCT_BITS>
        product0[MM_STREAM_8X64_PACKED_DSP_SIGNED_LANES];
    ap_int<MM_STREAM_8X64_PACKED_DSP_SIGNED_PRODUCT_BITS>
        product1[MM_STREAM_8X64_PACKED_DSP_SIGNED_LANES];
};

void compute_mm_stream_8x64_packed_dsp_signed_core(
    hls::stream<mm_stream_8x64_packed_dsp_signed_output_t>& out_stream,
    hls::stream<mm_stream_8x64_packed_dsp_signed_input_t>& in_stream);

#endif
