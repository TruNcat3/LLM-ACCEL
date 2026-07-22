#include "compute_stream.hpp"
#include <hls_math.h>

static fm_t cu_clamp_fm(fm_t x, fm_t lo, fm_t hi) {
    #pragma HLS inline
    if (x < lo) {
        return lo;
    }
    if (x > hi) {
        return hi;
    }
    return x;
}

static fm_t cu_recip_approx(fm_t x) {
    #pragma HLS inline
    fm_t safe_x = (x < fm_t(0.000244140625)) ? fm_t(0.000244140625) : x;
    fm_t y;
    if (safe_x >= fm_t(64)) {
        y = fm_t(0.015625);
    } else if (safe_x >= fm_t(32)) {
        y = fm_t(0.03125);
    } else if (safe_x >= fm_t(16)) {
        y = fm_t(0.0625);
    } else if (safe_x >= fm_t(8)) {
        y = fm_t(0.125);
    } else if (safe_x >= fm_t(4)) {
        y = fm_t(0.25);
    } else if (safe_x >= fm_t(2)) {
        y = fm_t(0.5);
    } else if (safe_x >= fm_t(1)) {
        y = fm_t(1);
    } else if (safe_x >= fm_t(0.5)) {
        y = fm_t(2);
    } else if (safe_x >= fm_t(0.25)) {
        y = fm_t(4);
    } else if (safe_x >= fm_t(0.125)) {
        y = fm_t(8);
    } else {
        y = fm_t(16);
    }

    for (unsigned int i = 0; i < 3; i++) {
        #pragma HLS unroll
        y = y * (fm_t(2) - safe_x * y);
    }
    return cu_clamp_fm(y, fm_t(0), fm_t(64));
}

static fm_t cu_exp_approx(fm_t x) {
    #pragma HLS inline
    fm_t xc = cu_clamp_fm(x, fm_t(-8), fm_t(8));
    ap_fixed<16, 8> exp_in = ap_fixed<16, 8>(xc);
    return fm_t(hls::exp(exp_in));
}

static fm_t cu_rsqrt_approx(fm_accum_t x) {
    #pragma HLS inline
    const fm_accum_t min_x = fm_accum_t(0.000244140625);
    fm_accum_t safe_x = (x < min_x) ? min_x : x;

    fm_t y;
    if (safe_x >= fm_accum_t(64)) {
        y = fm_t(0.125);
    } else if (safe_x >= fm_accum_t(16)) {
        y = fm_t(0.25);
    } else if (safe_x >= fm_accum_t(4)) {
        y = fm_t(0.5);
    } else if (safe_x >= fm_accum_t(1)) {
        y = fm_t(1);
    } else if (safe_x >= fm_accum_t(0.25)) {
        y = fm_t(2);
    } else if (safe_x >= fm_accum_t(0.0625)) {
        y = fm_t(4);
    } else {
        y = fm_t(8);
    }

    for (unsigned int i = 0; i < 3; i++) {
        #pragma HLS unroll
        fm_accum_t yy = fm_accum_t(y) * fm_accum_t(y);
        fm_accum_t refine = fm_accum_t(1.5) - fm_accum_t(0.5) * safe_x * yy;
        y = fm_t(fm_accum_t(y) * refine);
    }
    return cu_clamp_fm(y, fm_t(0), fm_t(32));
}

static fm_t cu_silu_approx(fm_t x) {
    #pragma HLS inline
    if (x >= fm_t(0)) {
        fm_t e = cu_exp_approx(fm_t(-x));
        return x * cu_recip_approx(fm_t(1) + e);
    }
    fm_t e = cu_exp_approx(x);
    return x * e * cu_recip_approx(fm_t(1) + e);
}

void cast_accum_to_fm_stream(
    hls::stream<cu_vec16_packet_t>& out_stream,
    hls::stream<cu_accum16_packet_t>& in_stream,
    unsigned int packet_count
) {
    #pragma HLS inline off

    for (unsigned int pkt = 0; pkt < packet_count; pkt++) {
        #pragma HLS pipeline II=1
        cu_accum16_packet_t in_pkt = in_stream.read();
        cu_vec16_packet_t out_pkt;
        out_pkt.valid_mask = in_pkt.valid_mask;
        out_pkt.token_lane = in_pkt.token_lane;
        out_pkt.elem_base = in_pkt.elem_base;
        out_pkt.block_id = in_pkt.block_id;
        out_pkt.last_block = in_pkt.last_block;
        out_pkt.last_stream = in_pkt.last_stream;
        for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
            #pragma HLS unroll
            out_pkt.data[lane] = fm_t(in_pkt.data[lane]);
        }
        out_stream.write(out_pkt);
    }
}

void stream_silu_mul(
    hls::stream<cu_vec16_packet_t>& out_stream,
    hls::stream<cu_vec16_packet_t>& gate_stream,
    hls::stream<cu_vec16_packet_t>& up_stream,
    unsigned int packet_count
) {
    #pragma HLS inline off

    constexpr unsigned int groups_per_packet = CU_VEC_LANES / CU_NL_LANES;
    constexpr unsigned int max_group_steps =
        CU_STREAM_MAX_PACKETS * groups_per_packet;
    unsigned int group = 0;
    cu_vec16_packet_t gate_pkt;
    cu_vec16_packet_t up_pkt;
    cu_vec16_packet_t out_pkt;
    #pragma HLS array_partition variable=gate_pkt.data complete dim=1
    #pragma HLS array_partition variable=up_pkt.data complete dim=1
    #pragma HLS array_partition variable=out_pkt.data complete dim=1

    for (unsigned int step = 0; step < packet_count * groups_per_packet; step++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_tripcount min=groups_per_packet max=max_group_steps
        if (group == 0) {
            gate_pkt = gate_stream.read();
            up_pkt = up_stream.read();
            out_pkt = gate_pkt;
            out_pkt.valid_mask = gate_pkt.valid_mask & up_pkt.valid_mask;
        }
        for (unsigned int lane = 0; lane < CU_NL_LANES; lane++) {
            #pragma HLS unroll
            unsigned int idx = group * CU_NL_LANES + lane;
            fm_t activated = cu_silu_approx(gate_pkt.data[idx]);
            out_pkt.data[idx] = activated * up_pkt.data[idx];
        }
        if (group + 1 == groups_per_packet) {
            out_stream.write(out_pkt);
            group = 0;
        } else {
            group++;
        }
    }
}

void stream_silu(
    hls::stream<cu_vec16_packet_t>& out_stream,
    hls::stream<cu_vec16_packet_t>& in_stream,
    unsigned int packet_count
) {
    #pragma HLS inline off

    constexpr unsigned int groups_per_packet = CU_VEC_LANES / CU_NL_LANES;
    constexpr unsigned int max_group_steps =
        CU_STREAM_MAX_PACKETS * groups_per_packet;
    unsigned int group = 0;
    cu_vec16_packet_t in_pkt;
    cu_vec16_packet_t out_pkt;
    #pragma HLS array_partition variable=in_pkt.data complete dim=1
    #pragma HLS array_partition variable=out_pkt.data complete dim=1

    for (unsigned int step = 0; step < packet_count * groups_per_packet; step++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_tripcount min=groups_per_packet max=max_group_steps
        if (group == 0) {
            in_pkt = in_stream.read();
            out_pkt = in_pkt;
        }
        for (unsigned int lane = 0; lane < CU_NL_LANES; lane++) {
            #pragma HLS unroll
            unsigned int idx = group * CU_NL_LANES + lane;
            out_pkt.data[idx] = in_pkt.valid_mask[idx]
                ? cu_silu_approx(in_pkt.data[idx])
                : fm_t(0);
        }
        if (group + 1 == groups_per_packet) {
            out_stream.write(out_pkt);
            group = 0;
        } else {
            group++;
        }
    }
}

void stream_add_residual(
    hls::stream<cu_vec16_packet_t>& out_stream,
    hls::stream<cu_vec16_packet_t>& lhs_stream,
    hls::stream<cu_vec16_packet_t>& rhs_stream,
    unsigned int packet_count
) {
    #pragma HLS inline off

    for (unsigned int pkt = 0; pkt < packet_count; pkt++) {
        #pragma HLS pipeline II=1
        cu_vec16_packet_t lhs = lhs_stream.read();
        cu_vec16_packet_t rhs = rhs_stream.read();
        cu_vec16_packet_t out = lhs;
        out.valid_mask = lhs.valid_mask & rhs.valid_mask;
        for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
            #pragma HLS unroll
            out.data[lane] = lhs.data[lane] + rhs.data[lane];
        }
        out_stream.write(out);
    }
}

void stream_rmsnorm(
    hls::stream<cu_vec16_packet_t>& out_stream,
    hls::stream<cu_vec16_packet_t>& in_stream,
    const wt_norm_t weights[MAX_LINEAR_OUT_DIM],
    unsigned int elem_count
) {
    #pragma HLS inline off

    fm_t cache[MAX_LINEAR_OUT_DIM];
    ap_uint<CU_VEC_LANES> masks[MAX_LINEAR_OUT_BLOCKS];
    unsigned int bases[MAX_LINEAR_OUT_BLOCKS];
    unsigned int token_lane = 0;
    #pragma HLS bind_storage variable=cache type=ram_2p impl=bram

    unsigned int packet_count = ceildiv(elem_count, CU_VEC_LANES);
    fm_accum_t sum_sq = fm_accum_t(0);

    for (unsigned int pkt = 0; pkt < MAX_LINEAR_OUT_BLOCKS; pkt++) {
        if (pkt < packet_count) {
            cu_vec16_packet_t in_pkt = in_stream.read();
            masks[pkt] = in_pkt.valid_mask;
            bases[pkt] = in_pkt.elem_base;
            token_lane = in_pkt.token_lane;
            for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
                #pragma HLS pipeline II=1
                unsigned int elem = in_pkt.elem_base + lane;
                fm_t value = in_pkt.data[lane];
                if (elem < elem_count && in_pkt.valid_mask[lane]) {
                    cache[elem] = value;
                    sum_sq += fm_accum_t(value) * fm_accum_t(value);
                }
            }
        }
    }

    fm_t inv_elems = cu_recip_approx(fm_t(elem_count));
    fm_accum_t mean_sq = sum_sq * fm_accum_t(inv_elems);
    fm_t inv_rms = cu_rsqrt_approx(mean_sq + fm_accum_t(RMS_NORM_EPS));

    for (unsigned int pkt = 0; pkt < MAX_LINEAR_OUT_BLOCKS; pkt++) {
        if (pkt < packet_count) {
            cu_vec16_packet_t out_pkt;
            out_pkt.valid_mask = masks[pkt];
            out_pkt.token_lane = token_lane;
            out_pkt.elem_base = bases[pkt];
            out_pkt.block_id = pkt;
            out_pkt.last_block = pkt + 1 == packet_count;
            out_pkt.last_stream = out_pkt.last_block;

            for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
                #pragma HLS pipeline II=1
                unsigned int elem = bases[pkt] + lane;
                fm_t value = fm_t(0);
                if (elem < elem_count && masks[pkt][lane]) {
                    value = cache[elem] * inv_rms * weights[elem];
                }
                out_pkt.data[lane] = value;
            }
            out_stream.write(out_pkt);
        }
    }
}

void stream_attention_mv_softmax(
    hls::stream<cu_vec16_packet_t>& out_stream,
    hls::stream<cu_vec16_packet_t>& q_stream,
    hls::stream<cu_vec16_packet_t>& k_stream,
    hls::stream<cu_vec16_packet_t>& v_stream,
    unsigned int tile_len,
    unsigned int head_dim
) {
    #pragma HLS inline off

    fm_t q_cache[MAX_LINEAR_IN_DIM];
    fm_t v_cache[MAX_LINEAR_IN_DIM];
    fm_accum_t acc[MAX_LINEAR_IN_DIM];
    #pragma HLS bind_storage variable=q_cache type=ram_2p impl=bram
    #pragma HLS bind_storage variable=v_cache type=ram_2p impl=bram
    #pragma HLS bind_storage variable=acc type=ram_2p impl=bram

    unsigned int block_count = ceildiv(head_dim, CU_VEC_LANES);

    for (unsigned int block = 0; block < MAX_LINEAR_IN_BLOCKS; block++) {
        if (block < block_count) {
            cu_vec16_packet_t q_pkt = q_stream.read();
            for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
                #pragma HLS pipeline II=1
                unsigned int elem = q_pkt.elem_base + lane;
                if (elem < head_dim && q_pkt.valid_mask[lane]) {
                    q_cache[elem] = q_pkt.data[lane];
                    acc[elem] = fm_accum_t(0);
                }
            }
        }
    }

    fm_t online_max = fm_t(-128);
    fm_accum_t online_sum = fm_accum_t(0);

    for (unsigned int pos = 0; pos < MAX_SEQ_LEN; pos++) {
        if (pos < tile_len) {
            fm_accum_t dot = fm_accum_t(0);
            for (unsigned int block = 0; block < MAX_LINEAR_IN_BLOCKS; block++) {
                if (block < block_count) {
                    cu_vec16_packet_t k_pkt = k_stream.read();
                    cu_vec16_packet_t v_pkt = v_stream.read();
                    for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
                        #pragma HLS pipeline II=1
                        unsigned int elem = k_pkt.elem_base + lane;
                        if (elem < head_dim && k_pkt.valid_mask[lane]) {
                            fm_t k_val = k_pkt.data[lane];
                            fm_t v_val = v_pkt.data[lane];
                            v_cache[elem] = v_val;
                            dot += fm_accum_t(q_cache[elem]) * fm_accum_t(k_val);
                        }
                    }
                }
            }

            fm_t score = fm_t(dot * fm_accum_t(ATTENTION_SCALE));
            fm_t new_max = (score > online_max) ? score : online_max;
            fm_t old_scale = cu_exp_approx(online_max - new_max);
            fm_t new_scale = cu_exp_approx(score - new_max);
            online_sum = online_sum * fm_accum_t(old_scale) + fm_accum_t(new_scale);

            for (unsigned int elem = 0; elem < MAX_LINEAR_IN_DIM; elem++) {
                if (elem < head_dim) {
                    #pragma HLS pipeline II=1
                    acc[elem] = acc[elem] * fm_accum_t(old_scale) +
                        fm_accum_t(new_scale) * fm_accum_t(v_cache[elem]);
                }
            }
            online_max = new_max;
        }
    }

    fm_t inv_sum = cu_recip_approx(fm_t(online_sum));
    for (unsigned int block = 0; block < MAX_LINEAR_IN_BLOCKS; block++) {
        if (block < block_count) {
            cu_vec16_packet_t out_pkt;
            out_pkt.valid_mask = 0;
            out_pkt.token_lane = 0;
            out_pkt.elem_base = block * CU_VEC_LANES;
            out_pkt.block_id = block;
            out_pkt.last_block = block + 1 == block_count;
            out_pkt.last_stream = out_pkt.last_block;
            for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
                #pragma HLS pipeline II=1
                unsigned int elem = out_pkt.elem_base + lane;
                if (elem < head_dim) {
                    out_pkt.valid_mask[lane] = 1;
                    out_pkt.data[lane] = fm_t(acc[elem] * fm_accum_t(inv_sum));
                } else {
                    out_pkt.data[lane] = fm_t(0);
                }
            }
            out_stream.write(out_pkt);
        }
    }
}

void stream_softmax_rows(
    hls::stream<cu_vec16_packet_t>& out_stream,
    hls::stream<cu_vec16_packet_t>& in_stream,
    unsigned int row_count,
    unsigned int elem_count
) {
    #pragma HLS inline off

    fm_t row_cache[CU_SOFTMAX_ELEMS];
    fm_t exp_cache[CU_SOFTMAX_ELEMS];
    ap_uint<CU_VEC_LANES> masks[CU_SOFTMAX_PACKETS];
    #pragma HLS array_partition variable=row_cache cyclic factor=CU_VEC_LANES dim=1
    #pragma HLS array_partition variable=exp_cache cyclic factor=CU_VEC_LANES dim=1
    #pragma HLS array_partition variable=masks complete dim=1

    unsigned int packet_count = ceildiv(elem_count, CU_VEC_LANES);
    for (unsigned int row = 0; row < CU_SOFTMAX_ROWS; row++) {
        if (row < row_count) {
            fm_t row_max = fm_t(-128);
            unsigned int token_lane = row;
            unsigned int block_id = 0;

            for (unsigned int packet = 0;
                 packet < CU_SOFTMAX_PACKETS;
                 packet++) {
                if (packet < packet_count) {
                    cu_vec16_packet_t in_packet = in_stream.read();
                    masks[packet] = in_packet.valid_mask;
                    token_lane = in_packet.token_lane;
                    block_id = in_packet.block_id;
                    for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
                        #pragma HLS pipeline II=1
                        unsigned int elem = packet * CU_VEC_LANES + lane;
                        fm_t value = in_packet.data[lane];
                        row_cache[elem] = value;
                        if (elem < elem_count &&
                            in_packet.valid_mask[lane] &&
                            value > row_max) {
                            row_max = value;
                        }
                    }
                }
            }

            fm_accum_t sum = fm_accum_t(0);
            for (unsigned int elem = 0;
                 elem < CU_SOFTMAX_ELEMS;
                 elem++) {
                #pragma HLS pipeline II=1
                if (elem < elem_count) {
                    unsigned int packet = elem / CU_VEC_LANES;
                    unsigned int lane = elem % CU_VEC_LANES;
                    if (masks[packet][lane]) {
                        fm_t exp_value = cu_exp_approx(
                            row_cache[elem] - row_max
                        );
                        exp_cache[elem] = exp_value;
                        sum += fm_accum_t(exp_value);
                    }
                }
            }

            fm_t inv_sum = cu_recip_approx(fm_t(sum));
            for (unsigned int packet = 0;
                 packet < CU_SOFTMAX_PACKETS;
                 packet++) {
                if (packet < packet_count) {
                    cu_vec16_packet_t out_packet;
                    out_packet.valid_mask = masks[packet];
                    out_packet.token_lane = token_lane;
                    out_packet.elem_base = packet * CU_VEC_LANES;
                    out_packet.block_id = block_id;
                    out_packet.last_block =
                        row + 1 == row_count &&
                        packet + 1 == packet_count;
                    out_packet.last_stream = out_packet.last_block;
                    for (unsigned int lane = 0;
                         lane < CU_VEC_LANES;
                         lane++) {
                        #pragma HLS pipeline II=1
                        unsigned int elem = packet * CU_VEC_LANES + lane;
                        out_packet.data[lane] =
                            elem < elem_count && masks[packet][lane] ?
                            fm_t(exp_cache[elem] * inv_sum) :
                            fm_t(0);
                    }
                    out_stream.write(out_packet);
                }
            }
        }
    }
}

void compute_stream_nonlinear_unit_synth_stub(
    fm_t sink[CU_VEC_LANES],
    fm_t gate_in[CU_VEC_LANES],
    fm_t up_in[CU_VEC_LANES],
    wt_norm_t norm_weights[CU_VEC_LANES]
) {
    #pragma HLS inline off
    #pragma HLS interface m_axi port=sink offset=slave bundle=sink depth=16 max_widen_bitwidth=512
    #pragma HLS interface m_axi port=gate_in offset=slave bundle=gate depth=16 max_widen_bitwidth=512
    #pragma HLS interface m_axi port=up_in offset=slave bundle=up depth=16 max_widen_bitwidth=512
    #pragma HLS interface m_axi port=norm_weights offset=slave bundle=norm depth=16 max_widen_bitwidth=512
    #pragma HLS interface s_axilite port=sink bundle=control
    #pragma HLS interface s_axilite port=gate_in bundle=control
    #pragma HLS interface s_axilite port=up_in bundle=control
    #pragma HLS interface s_axilite port=norm_weights bundle=control
    #pragma HLS interface s_axilite port=return bundle=control

    hls::stream<cu_vec16_packet_t> gate_stream;
    hls::stream<cu_vec16_packet_t> up_stream;
    hls::stream<cu_vec16_packet_t> silu_stream;
    hls::stream<cu_vec16_packet_t> rms_stream;
    hls::stream<cu_vec16_packet_t> rms_out_stream;
    #pragma HLS stream variable=gate_stream depth=2
    #pragma HLS stream variable=up_stream depth=2
    #pragma HLS stream variable=silu_stream depth=2
    #pragma HLS stream variable=rms_stream depth=2
    #pragma HLS stream variable=rms_out_stream depth=2

    cu_vec16_packet_t gate_pkt;
    cu_vec16_packet_t up_pkt;
    gate_pkt.valid_mask = 0xffff;
    up_pkt.valid_mask = 0xffff;
    gate_pkt.token_lane = 0;
    up_pkt.token_lane = 0;
    gate_pkt.elem_base = 0;
    up_pkt.elem_base = 0;
    gate_pkt.block_id = 0;
    up_pkt.block_id = 0;
    gate_pkt.last_block = true;
    up_pkt.last_block = true;
    gate_pkt.last_stream = true;
    up_pkt.last_stream = true;
    for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
        #pragma HLS pipeline II=1
        gate_pkt.data[lane] = gate_in[lane];
        up_pkt.data[lane] = up_in[lane];
    }

    gate_stream.write(gate_pkt);
    up_stream.write(up_pkt);
    stream_silu_mul(silu_stream, gate_stream, up_stream, 1);

    cu_vec16_packet_t silu_pkt = silu_stream.read();
    rms_stream.write(silu_pkt);
    stream_rmsnorm(rms_out_stream, rms_stream, norm_weights, CU_VEC_LANES);
    cu_vec16_packet_t out_pkt = rms_out_stream.read();

    for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
        #pragma HLS pipeline II=1
        sink[lane] = out_pkt.data[lane];
    }
}

static fm_t read_fm_word_lane(
    const fm_word_t words[],
    unsigned int value_idx
) {
    #pragma HLS inline

    fm_word_t word = words[value_idx / FM_BLOCK_SIZE];
    return unpack_fm_word_lane(word, value_idx % FM_BLOCK_SIZE);
}

static void emit_vec_stream_from_fm_words(
    hls::stream<cu_vec16_packet_t>& out_stream,
    const fm_word_t words[CU_STREAM_MAX_FM_WORDS],
    unsigned int token_count,
    unsigned int elem_count,
    unsigned int block_id
) {
    #pragma HLS inline off

    unsigned int elem_packets = ceildiv(elem_count, CU_VEC_LANES);
    for (unsigned int token = 0; token < LINEAR_TOKEN_TILE_ACTIVE; token++) {
        for (unsigned int elem_packet = 0; elem_packet < MAX_LINEAR_OUT_BLOCKS; elem_packet++) {
            #pragma HLS pipeline II=1
            if (elem_packet < elem_packets) {
                unsigned int packet_linear = token * elem_packets + elem_packet;
                cu_vec16_packet_t packet;
                packet.valid_mask = 0;
                packet.token_lane = token;
                packet.elem_base = elem_packet * CU_VEC_LANES;
                packet.block_id = block_id;
                packet.last_block =
                    token == LINEAR_TOKEN_TILE_ACTIVE - 1 &&
                    elem_packet == elem_packets - 1;
                packet.last_stream = packet.last_block;

                for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
                    #pragma HLS unroll
                    unsigned int elem = packet.elem_base + lane;
                    unsigned int value_idx = packet_linear * CU_VEC_LANES + lane;
                    bool valid = token < token_count && elem < elem_count;
                    packet.data[lane] = valid ? read_fm_word_lane(words, value_idx) : fm_t(0);
                    packet.valid_mask[lane] = valid;
                }
                out_stream.write(packet);
            }
        }
    }
}

static fm_accum_t drain_vec_stream_checksum(
    hls::stream<cu_vec16_packet_t>& in_stream,
    unsigned int packet_count
) {
    #pragma HLS inline off

    fm_accum_t checksum = fm_accum_t(0);
    for (unsigned int pkt = 0; pkt < CU_STREAM_MAX_PACKETS; pkt++) {
        if (pkt < packet_count) {
            cu_vec16_packet_t packet = in_stream.read();
            for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
                #pragma HLS pipeline II=1
                if (packet.valid_mask[lane]) {
                    checksum += fm_accum_t(packet.data[lane]);
                }
            }
        }
    }
    return checksum;
}

static void emit_q_stream_from_words(
    hls::stream<cu_vec16_packet_t>& q_stream,
    const fm_word_t q_words[CU_MV_Q_FM_WORDS],
    unsigned int head_dim
) {
    #pragma HLS inline off

    unsigned int block_count = ceildiv(head_dim, CU_VEC_LANES);
    for (unsigned int block = 0; block < MAX_LINEAR_IN_BLOCKS; block++) {
        #pragma HLS pipeline II=1
        if (block < block_count) {
            cu_vec16_packet_t packet;
            packet.valid_mask = 0;
            packet.token_lane = 0;
            packet.elem_base = block * CU_VEC_LANES;
            packet.block_id = block;
            packet.last_block = block + 1 == block_count;
            packet.last_stream = packet.last_block;
            for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
                #pragma HLS unroll
                unsigned int elem = packet.elem_base + lane;
                bool valid = elem < head_dim;
                packet.data[lane] = valid ? read_fm_word_lane(q_words, elem) : fm_t(0);
                packet.valid_mask[lane] = valid;
            }
            q_stream.write(packet);
        }
    }
}

static void emit_kv_streams_from_words(
    hls::stream<cu_vec16_packet_t>& k_stream,
    hls::stream<cu_vec16_packet_t>& v_stream,
    const fm_word_t k_words[CU_MV_CACHE_FM_WORDS],
    const fm_word_t v_words[CU_MV_CACHE_FM_WORDS],
    unsigned int tile_len,
    unsigned int head_dim
) {
    #pragma HLS inline off

    unsigned int block_count = ceildiv(head_dim, CU_VEC_LANES);
    for (unsigned int pos = 0; pos < MAX_SEQ_LEN; pos++) {
        for (unsigned int block = 0; block < MAX_LINEAR_IN_BLOCKS; block++) {
            #pragma HLS pipeline II=1
            if (pos < tile_len && block < block_count) {
                cu_vec16_packet_t k_pkt;
                cu_vec16_packet_t v_pkt;
                k_pkt.valid_mask = 0;
                v_pkt.valid_mask = 0;
                k_pkt.token_lane = pos;
                v_pkt.token_lane = pos;
                k_pkt.elem_base = block * CU_VEC_LANES;
                v_pkt.elem_base = block * CU_VEC_LANES;
                k_pkt.block_id = pos;
                v_pkt.block_id = pos;
                k_pkt.last_block = block + 1 == block_count;
                v_pkt.last_block = k_pkt.last_block;
                k_pkt.last_stream = pos + 1 == tile_len && k_pkt.last_block;
                v_pkt.last_stream = k_pkt.last_stream;

                for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
                    #pragma HLS unroll
                    unsigned int elem = k_pkt.elem_base + lane;
                    unsigned int value_idx = pos * head_dim + elem;
                    bool valid = elem < head_dim;
                    k_pkt.data[lane] = valid ? read_fm_word_lane(k_words, value_idx) : fm_t(0);
                    v_pkt.data[lane] = valid ? read_fm_word_lane(v_words, value_idx) : fm_t(0);
                    k_pkt.valid_mask[lane] = valid;
                    v_pkt.valid_mask[lane] = valid;
                }
                k_stream.write(k_pkt);
                v_stream.write(v_pkt);
            }
        }
    }
}

static fm_accum_t drain_mv_stream_checksum(
    hls::stream<cu_vec16_packet_t>& mv_out_stream,
    unsigned int head_dim
) {
    #pragma HLS inline off

    fm_accum_t checksum = fm_accum_t(0);
    unsigned int block_count = ceildiv(head_dim, CU_VEC_LANES);
    for (unsigned int block = 0; block < MAX_LINEAR_IN_BLOCKS; block++) {
        if (block < block_count) {
            cu_vec16_packet_t packet = mv_out_stream.read();
            for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
                #pragma HLS pipeline II=1
                if (packet.valid_mask[lane]) {
                    checksum += fm_accum_t(packet.data[lane]);
                }
            }
        }
    }
    return checksum;
}

void compute_stream_silu_mul_perf_stub(
    fm_word_t sink_words[1],
    const fm_word_t gate_words[CU_STREAM_MAX_FM_WORDS],
    const fm_word_t up_words[CU_STREAM_MAX_FM_WORDS],
    unsigned int token_count,
    unsigned int elem_count
) {
    #pragma HLS inline off
    #pragma HLS interface m_axi port=sink_words offset=slave bundle=sink depth=1 max_widen_bitwidth=512
    #pragma HLS interface m_axi port=gate_words offset=slave bundle=gate depth=CU_STREAM_MAX_FM_WORDS max_widen_bitwidth=512
    #pragma HLS interface m_axi port=up_words offset=slave bundle=up depth=CU_STREAM_MAX_FM_WORDS max_widen_bitwidth=512
    #pragma HLS interface s_axilite port=sink_words bundle=control
    #pragma HLS interface s_axilite port=gate_words bundle=control
    #pragma HLS interface s_axilite port=up_words bundle=control
    #pragma HLS interface s_axilite port=token_count bundle=control
    #pragma HLS interface s_axilite port=elem_count bundle=control
    #pragma HLS interface s_axilite port=return bundle=control

    hls::stream<cu_vec16_packet_t> gate_stream;
    hls::stream<cu_vec16_packet_t> up_stream;
    hls::stream<cu_vec16_packet_t> silu_stream;
    #pragma HLS stream variable=gate_stream depth=CU_STREAM_MAX_PACKETS
    #pragma HLS stream variable=up_stream depth=CU_STREAM_MAX_PACKETS
    #pragma HLS stream variable=silu_stream depth=CU_STREAM_MAX_PACKETS

    unsigned int elem_packets = ceildiv(elem_count, CU_VEC_LANES);
    unsigned int packet_count = LINEAR_TOKEN_TILE_ACTIVE * elem_packets;

    emit_vec_stream_from_fm_words(gate_stream, gate_words, token_count, elem_count, 0);
    emit_vec_stream_from_fm_words(up_stream, up_words, token_count, elem_count, 0);
    stream_silu_mul(silu_stream, gate_stream, up_stream, packet_count);
    fm_accum_t silu_checksum = drain_vec_stream_checksum(silu_stream, packet_count);

    fm_word_t sink_word = 0;
    set_fm_word_lane(sink_word, 0, fm_t(silu_checksum));
    set_fm_word_lane(sink_word, 1, fm_t(packet_count));
    set_fm_word_lane(sink_word, 2, fm_t(token_count));
    set_fm_word_lane(sink_word, 3, fm_t(elem_count));
    sink_words[0] = sink_word;
}

void compute_stream_mv_softmax_perf_stub(
    fm_word_t sink_words[1],
    const fm_word_t q_words[CU_MV_Q_FM_WORDS],
    const fm_word_t k_words[CU_MV_CACHE_FM_WORDS],
    const fm_word_t v_words[CU_MV_CACHE_FM_WORDS],
    unsigned int tile_len,
    unsigned int head_dim
) {
    #pragma HLS inline off
    #pragma HLS interface m_axi port=sink_words offset=slave bundle=sink depth=1 max_widen_bitwidth=512
    #pragma HLS interface m_axi port=q_words offset=slave bundle=q depth=CU_MV_Q_FM_WORDS max_widen_bitwidth=512
    #pragma HLS interface m_axi port=k_words offset=slave bundle=k depth=CU_MV_CACHE_FM_WORDS max_widen_bitwidth=512
    #pragma HLS interface m_axi port=v_words offset=slave bundle=v depth=CU_MV_CACHE_FM_WORDS max_widen_bitwidth=512
    #pragma HLS interface s_axilite port=sink_words bundle=control
    #pragma HLS interface s_axilite port=q_words bundle=control
    #pragma HLS interface s_axilite port=k_words bundle=control
    #pragma HLS interface s_axilite port=v_words bundle=control
    #pragma HLS interface s_axilite port=tile_len bundle=control
    #pragma HLS interface s_axilite port=head_dim bundle=control
    #pragma HLS interface s_axilite port=return bundle=control

    hls::stream<cu_vec16_packet_t> q_stream;
    hls::stream<cu_vec16_packet_t> k_stream;
    hls::stream<cu_vec16_packet_t> v_stream;
    hls::stream<cu_vec16_packet_t> mv_out_stream;
    #pragma HLS stream variable=q_stream depth=MAX_LINEAR_IN_BLOCKS
    #pragma HLS stream variable=k_stream depth=CU_MV_MAX_PACKETS
    #pragma HLS stream variable=v_stream depth=CU_MV_MAX_PACKETS
    #pragma HLS stream variable=mv_out_stream depth=MAX_LINEAR_IN_BLOCKS

    emit_q_stream_from_words(q_stream, q_words, head_dim);
    emit_kv_streams_from_words(k_stream, v_stream, k_words, v_words, tile_len, head_dim);
    stream_attention_mv_softmax(mv_out_stream, q_stream, k_stream, v_stream, tile_len, head_dim);
    fm_accum_t mv_checksum = drain_mv_stream_checksum(mv_out_stream, head_dim);

    fm_word_t sink_word = 0;
    set_fm_word_lane(sink_word, 0, fm_t(mv_checksum));
    set_fm_word_lane(sink_word, 1, fm_t(tile_len));
    set_fm_word_lane(sink_word, 2, fm_t(head_dim));
    sink_words[0] = sink_word;
}
