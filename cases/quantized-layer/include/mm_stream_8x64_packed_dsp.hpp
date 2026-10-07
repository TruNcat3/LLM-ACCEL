#ifndef LLM_FPGA_MM_STREAM_8X64_PACKED_DSP_HPP
#define LLM_FPGA_MM_STREAM_8X64_PACKED_DSP_HPP

#include <ap_int.h>
#include <hls_stream.h>

// A feasibility probe for two unsigned INT8xINT4 products per DSP48E2.
// The two 12-bit products are separated by 12 zero bits in the packed
// operand, so the packed multiply is exact for the stated unsigned ranges.
#ifndef MM_STREAM_8X64_PACKED_DSP_PHYSICAL_LANES_CONFIG
#define MM_STREAM_8X64_PACKED_DSP_PHYSICAL_LANES_CONFIG 256
#endif
constexpr unsigned int MM_STREAM_8X64_PACKED_DSP_PACKING_FACTOR = 2;
constexpr unsigned int MM_STREAM_8X64_PACKED_DSP_PHYSICAL_LANES =
    MM_STREAM_8X64_PACKED_DSP_PHYSICAL_LANES_CONFIG;
constexpr unsigned int MM_STREAM_8X64_PACKED_DSP_LOGICAL_MACS_PER_PACKET =
    MM_STREAM_8X64_PACKED_DSP_PHYSICAL_LANES *
    MM_STREAM_8X64_PACKED_DSP_PACKING_FACTOR;
// Compatibility alias used by the existing probe and testbench.
constexpr unsigned int MM_STREAM_8X64_PACKED_DSP_LANES =
    MM_STREAM_8X64_PACKED_DSP_PHYSICAL_LANES;
constexpr unsigned int MM_STREAM_8X64_PACKED_DSP_PACKETS = 4;
constexpr unsigned int MM_STREAM_8X64_PACKED_DSP_PRODUCT_BITS = 12;

static_assert(MM_STREAM_8X64_PACKED_DSP_PACKING_FACTOR == 2,
              "the packed unsigned probe emits two products per physical lane");
static_assert(MM_STREAM_8X64_PACKED_DSP_PHYSICAL_LANES > 0,
              "the packed unsigned probe needs at least one physical lane");

struct mm_stream_8x64_packed_dsp_input_t {
    ap_uint<8> activation[MM_STREAM_8X64_PACKED_DSP_LANES];
    ap_uint<4> weight0[MM_STREAM_8X64_PACKED_DSP_LANES];
    ap_uint<4> weight1[MM_STREAM_8X64_PACKED_DSP_LANES];
};

struct mm_stream_8x64_packed_dsp_output_t {
    ap_uint<MM_STREAM_8X64_PACKED_DSP_PRODUCT_BITS>
        product0[MM_STREAM_8X64_PACKED_DSP_LANES];
    ap_uint<MM_STREAM_8X64_PACKED_DSP_PRODUCT_BITS>
        product1[MM_STREAM_8X64_PACKED_DSP_LANES];
};

void compute_mm_stream_8x64_packed_dsp_core(
    hls::stream<mm_stream_8x64_packed_dsp_output_t>& out_stream,
    hls::stream<mm_stream_8x64_packed_dsp_input_t>& in_stream);

#endif
