// XRT host for sw_emu/hw: control_cache_core(编排器) + 1×V8-2_s, operator_program QKV 三连
// stream (axis) 由 conn cfg 连接 (含 ctrl_stream), host 只 set m_axi + s_axilite args。
// 验证: 3 op (Q/K/V), 权重 1.0/2.0/0.5, input 全 1.0 → 输出 4.0/8.0/2.0。
#include "xcl2.hpp"
#include <vector>
#include <iostream>
#include <cstdint>
#include <cmath>

// ap_fixed bit patterns (host 用 int 存 bit pattern):
//   fm_t       = ap_fixed<16,8>:  1.0 = 1<<8  = 0x0100
//   wt_linear_t= ap_fixed<16,4>:  1.0 = 1<<12 = 0x1000, 2.0 = 0x2000, 0.5 = 0x0800
//   fm_accum_t = ap_fixed<32,16>: host 读 int32, float = raw / 65536.0
static void check_cl(cl_int err, const char* msg) {
    if (err != CL_SUCCESS) { std::cerr << "CL error: " << msg << " (" << err << ")\n"; exit(1); }
}
int main(int argc, char** argv) {
    std::cerr << "[dbg] host starting\n";
    if (argc < 2) { std::cerr << "usage: " << argv[0] << " <xclbin> [device_id]\n"; return 1; }
    std::string xclbin_path = argv[1];

    const unsigned NUM_TOKENS = 1;
    const unsigned NUM_TILES = 16, INPUT_DIM = 16, OUTPUT_DIM = 64;
    const unsigned PKT_PER_TOKEN = 32;        // 2 lane × 16 tile (D=16, TOTAL_LANES=2)
    const unsigned WT_PER_V82 = 64;           // 8 lane × 8 block
    const unsigned NUM_OPS = 7;               // 整层 matmul: Q/K/V/O/Gate/Up/Down
    const unsigned OUT_PER_OP = PKT_PER_TOKEN * OUTPUT_DIM;   // 8192
    const unsigned OP_TASK_STRIDE = 6;
    const unsigned SRC_HBM = 0, DST_HBM = 0;

    // ---- HBM 数据 ----
    std::vector<int16_t> hin(NUM_TOKENS * NUM_TILES * INPUT_DIM, 0x0100);  // input=1.0
    std::vector<int32_t> hout(NUM_OPS * OUT_PER_OP, 0);
    // 7 段权重, 不同值验证: Q=1.0(0x1000), K=2.0(0x2000), V=0.5(0x0800),
    //   O=1.0, Gate=1.0, Up=1.0, Down=1.0
    int16_t w_val[7] = {0x1000, 0x2000, 0x0800, 0x1000, 0x1000, 0x1000, 0x1000};
    // STEP 2: weight 拆 4 bank (每 bank WT_PER_V82/4 * 32 元素/op)
    unsigned WT_PER_BANK = WT_PER_V82 / 4;
    std::vector<int16_t> w0(NUM_OPS * WT_PER_BANK * 32, 0);
    std::vector<int16_t> w1(NUM_OPS * WT_PER_BANK * 32, 0);
    std::vector<int16_t> w2(NUM_OPS * WT_PER_BANK * 32, 0);
    std::vector<int16_t> w3(NUM_OPS * WT_PER_BANK * 32, 0);
    for (unsigned op = 0; op < NUM_OPS; op++) {
        for (unsigned b = 0; b < WT_PER_BANK * 32; b++) {
            w0[op * WT_PER_BANK * 32 + b] = w_val[op];
            w1[op * WT_PER_BANK * 32 + b] = w_val[op];
            w2[op * WT_PER_BANK * 32 + b] = w_val[op];
            w3[op * WT_PER_BANK * 32 + b] = w_val[op];
        }
    }
    // op_program: 7 op × 6 uint. op_ctrl: standalone ops 加 FINALIZE(0x80)
    unsigned char op_ctrls[7] = {0x80, 0x80, 0x80, 0x80, 0x02|0x80, 0x80, 0x80};  // NONE|FIN, GELU|FIN
    std::vector<uint32_t> op_program(NUM_OPS * OP_TASK_STRIDE, 0);
    for (unsigned op = 0; op < NUM_OPS; op++) {
        unsigned base = op * OP_TASK_STRIDE;
        op_program[base + 0] = op_ctrls[op];
        op_program[base + 1] = op * WT_PER_V82;      // weight_offset
        op_program[base + 2] = SRC_HBM;
        op_program[base + 3] = 0;                    // 都从 hin[0] (fan-out)
        op_program[base + 4] = DST_HBM;
        op_program[base + 5] = op * OUT_PER_OP;
    }
    std::vector<int16_t> param(OUTPUT_DIM, 0);

    // ---- OpenCL init ----
    cl_int err = CL_SUCCESS;
    auto devices = xcl::get_xil_devices();
    auto file_buf = xcl::read_binary_file(xclbin_path);
    cl::Program::Binaries bins{{file_buf.data(), file_buf.size()}};
    cl::Context context; cl::CommandQueue q; cl::Program program; bool valid = false;
    for (auto& d : devices) {
        context = cl::Context(d, nullptr, nullptr, nullptr, &err);
        q = cl::CommandQueue(context, d, CL_QUEUE_PROFILING_ENABLE, &err);
        program = cl::Program(context, {d}, bins, nullptr, &err);
        if (err == CL_SUCCESS) { valid = true; break; }
    }
    if (!valid) { std::cerr << "failed to program device\n"; return 1; }

    cl::Kernel krnl_cc(program, "control_cache_core", &err); check_cl(err, "cc");
    cl::Kernel krnl_v0(program, "qkv_tile_kernel_cc_qwen_small_core_v8_2_s", &err); check_cl(err, "v82");

    // ---- buffers (HBM bank from cfg: hin HBM0, hout HBM1, weight HBM2, op_program HBM3, param HBM0) ----
    auto make_ext = [](void* ptr, int bank) { cl_mem_ext_ptr_t e; e.obj = ptr; e.param = 0; e.flags = bank | XCL_MEM_TOPOLOGY; return e; };
    cl_mem_ext_ptr_t hin_ext = make_ext(hin.data(), 0);
    cl_mem_ext_ptr_t hout_ext = make_ext(hout.data(), 1);
    cl_mem_ext_ptr_t w0_ext = make_ext(w0.data(), 2);
    cl_mem_ext_ptr_t w1_ext = make_ext(w1.data(), 3);
    cl_mem_ext_ptr_t w2_ext = make_ext(w2.data(), 4);
    cl_mem_ext_ptr_t w3_ext = make_ext(w3.data(), 5);
    cl_mem_ext_ptr_t opp_ext = make_ext(op_program.data(), 6);
    cl_mem_ext_ptr_t param_ext = make_ext(param.data(), 0);
    cl::Buffer hin_buf(context, CL_MEM_EXT_PTR_XILINX|CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY,  hin.size()*2, &hin_ext, &err); check_cl(err,"hin");
    cl::Buffer hout_buf(context, CL_MEM_EXT_PTR_XILINX|CL_MEM_USE_HOST_PTR|CL_MEM_READ_WRITE, hout.size()*4, &hout_ext, &err); check_cl(err,"hout");
    cl::Buffer w0_buf(context, CL_MEM_EXT_PTR_XILINX|CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY, w0.size()*2, &w0_ext, &err); check_cl(err,"w0");
    cl::Buffer w1_buf(context, CL_MEM_EXT_PTR_XILINX|CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY, w1.size()*2, &w1_ext, &err); check_cl(err,"w1");
    cl::Buffer w2_buf(context, CL_MEM_EXT_PTR_XILINX|CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY, w2.size()*2, &w2_ext, &err); check_cl(err,"w2");
    cl::Buffer w3_buf(context, CL_MEM_EXT_PTR_XILINX|CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY, w3.size()*2, &w3_ext, &err); check_cl(err,"w3");
    cl::Buffer opp_buf(context, CL_MEM_EXT_PTR_XILINX|CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY, op_program.size()*4, &opp_ext, &err); check_cl(err,"op_program");
    cl::Buffer param_buf(context, CL_MEM_EXT_PTR_XILINX|CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY, param.size()*2, &param_ext, &err); check_cl(err,"param");

    // ---- set args ----
    // cc: 0=hidden_in, 1=hidden_out, 2=weight_hbm, 3-7=stream(skip: to_compute/weight/out_lo/out_hi/ctrl),
    //     8=op_program, 9=num_ops, 10=num_tokens
    // cc: 0=hidden_in, 1=hidden_out, 2-5=weight_hbm_0..3, 6-13=stream(skip), 14=op_program, 15=num_ops, 16=num_tokens
    krnl_cc.setArg(0, hin_buf); krnl_cc.setArg(1, hout_buf);
    krnl_cc.setArg(2, w0_buf); krnl_cc.setArg(3, w1_buf); krnl_cc.setArg(4, w2_buf); krnl_cc.setArg(5, w3_buf);
    krnl_cc.setArg(14, opp_buf); krnl_cc.setArg(15, NUM_OPS); krnl_cc.setArg(16, NUM_TOKENS);
    // v82: 0-7=stream(skip: in/out_lo/out_hi/weight0-3/ctrl), 8=param1, 9=param2, 10=num_tokens, 11=num_ops, 12=use_param1, 13=use_param2
    krnl_v0.setArg(8, param_buf); krnl_v0.setArg(9, param_buf);
    krnl_v0.setArg(10, NUM_TOKENS); krnl_v0.setArg(11, NUM_OPS); krnl_v0.setArg(12, 0u); krnl_v0.setArg(13, 0u);
    std::cerr << "[dbg] setArg done\n";

    // ---- migrate H2D ----
    q.enqueueMigrateMemObjects({hin_buf, w0_buf, w1_buf, w2_buf, w3_buf, opp_buf, param_buf}, 0, nullptr, nullptr);
    q.finish();
    std::cerr << "[dbg] migrate done\n";

    // ---- run: cc + v82 并发 (各独立 queue) ----
    cl::CommandQueue q_v0(context, devices[0], CL_QUEUE_PROFILING_ENABLE, &err);
    cl::CommandQueue q_cc(context, devices[0], CL_QUEUE_PROFILING_ENABLE, &err);
    q_v0.enqueueTask(krnl_v0); q_cc.enqueueTask(krnl_cc);
    q_v0.finish(); q_cc.finish();
    std::cerr << "[dbg] run complete\n";

    // ---- read hidden_out ----
    q.enqueueMigrateMemObjects({hout_buf}, CL_MIGRATE_MEM_OBJECT_HOST);
    q.finish();

    // ---- 校验: 整层 7 op (Q/K/V/O/Gate/Up/Down) ----
    // Q(w=1)=4, K(w=2)=8, V(w=0.5)=2, O(w=1)=4, Gate(w=1,GELU)=~3.96, Up(w=1)=4, Down(w=1)=4
    const char* opnames[7] = {"Q","K","V","O","Gate(GELU)","Up","Down"};
    float w_float[7] = {1.0f, 2.0f, 0.5f, 1.0f, 1.0f, 1.0f, 1.0f};
    float expect[7];
    for (int i = 0; i < 7; i++) {
        float raw = w_float[i] * (float)INPUT_DIM;  // input=1, w, INPUT_DIM-element reduction
        if (op_ctrls[i] == 0x02) {  // GELU: x>3 → ~x
            expect[i] = (raw > 3.0f) ? raw : raw * 0.8f;
        } else {
            expect[i] = raw;
        }
    }
    bool pass = true; int bad = 0;
    for (unsigned op = 0; op < NUM_OPS; op++) {
        int miss = 0;
        for (unsigned a = 0; a < OUT_PER_OP; a++) {
            float v = hout[op * OUT_PER_OP + a] / 65536.0f;
            if (std::fabs(v - expect[op]) > 0.15f) { pass = false; miss++; if (bad < 6) std::cout<<"miss "<<opnames[op]<<" a="<<a<<" v="<<v<<" exp="<<expect[op]<<"\n"; bad++; }
        }
        float sample = hout[op * OUT_PER_OP] / 65536.0f;
        std::cout << opnames[op] << ": expect=" << expect[op] << " sample=" << sample
                  << " miss=" << miss << "/" << OUT_PER_OP << (miss==0?" ✅":" ❌") << "\n";
    }
    std::cout << "整层 7-op: " << (pass ? "✅ operator_program 整层编排正确" : "❌ 存在错误") << "\n";
    return pass ? 0 : 1;
}
