// host_accum: cc + V8-2_s 累积模式 sw_emu 测试
// 4 op 做一个 16→64 reduction (4 chunk × INPUT_DIM=4): op0=overwrite, op1/2=accumulate, op3=accumulate+finalize
// input=1, weight=1 → 每 chunk 贡献 4, 4 chunk 累积 = 16
#include "xcl2.hpp"
#include <vector>
#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <cmath>

static void check_cl(cl_int e, const char* m){ if(e!=CL_SUCCESS){std::cerr<<"CL err: "<<m<<" ("<<e<<")\n";exit(1);} }

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "usage: " << argv[0] << " <xclbin> [num_chunks]\n"; return 1; }

    const unsigned NUM_TOKENS = 1, NUM_TILES = 16, INPUT_DIM = 16, OUTPUT_DIM = 64;
    const unsigned PKT_PER_TOKEN = 32, WT_PER_V82 = 64;   // D=16: 2 lane × 16 tile
    unsigned NUM_CHUNKS = 16;  // reduction chunks (可配置 argv[2], 分段验证性能)
    if (argc >= 3) NUM_CHUNKS = (unsigned)atoi(argv[2]);
    if (NUM_CHUNKS < 1) NUM_CHUNKS = 1;
    if (NUM_CHUNKS > 512) NUM_CHUNKS = 512;  // cc MAX_OPS=512 上限
    const unsigned OUT_PER_OP = PKT_PER_TOKEN * OUTPUT_DIM;  // 8192
    const unsigned OP_TASK_STRIDE = 6;

    std::vector<int16_t> hin(NUM_TOKENS * NUM_TILES * INPUT_DIM, 0x0100);
    std::vector<int32_t> hout(OUT_PER_OP, 0);
    std::vector<int16_t> w0(NUM_CHUNKS * WT_PER_V82/4 * 32, 0x1000);
    std::vector<int16_t> w1(NUM_CHUNKS * WT_PER_V82/4 * 32, 0x1000);
    std::vector<int16_t> w2(NUM_CHUNKS * WT_PER_V82/4 * 32, 0x1000);
    std::vector<int16_t> w3(NUM_CHUNKS * WT_PER_V82/4 * 32, 0x1000);
    // op_program: 4 chunk 的累积序列
    // op0: ctrl=0x00 (NONE, overwrite, no output)
    // op1: ctrl=0x40 (NONE|ACCUMULATE, no output)
    // op2: ctrl=0x40 (NONE|ACCUMULATE, no output)
    // op3: ctrl=0xC0 (NONE|ACCUMULATE|FINALIZE, accumulate + output)
    // ctrl 序列: op0=overwrite(0x00), op1..14=accumulate(0x40), op15=accumulate+finalize(0xC0)
    std::vector<uint32_t> op_program(NUM_CHUNKS * OP_TASK_STRIDE, 0);
    for (unsigned op = 0; op < NUM_CHUNKS; op++) {
        unsigned base = op * OP_TASK_STRIDE;
        unsigned char ctrl;
        if (op == 0)              ctrl = 0x00;              // 首步: overwrite
        else if (op == NUM_CHUNKS-1) ctrl = 0x40 | 0x80;   // 末步: accumulate + finalize
        else                      ctrl = 0x40;              // 中间: accumulate
        op_program[base + 0] = ctrl;
        op_program[base + 1] = op * WT_PER_V82;   // weight_offset
        op_program[base + 2] = 0;                  // SRC_HBM
        op_program[base + 3] = 0;                  // input_hbm_offset (同输入)
        op_program[base + 4] = 0;                  // DST_HBM
        op_program[base + 5] = 0;                  // output_hbm_offset (最终步写这里)
    }
    std::vector<int16_t> param(OUTPUT_DIM, 0);

    cl_int err;
    auto devices = xcl::get_xil_devices();
    auto fb = xcl::read_binary_file(argv[1]);
    cl::Program::Binaries bins{{fb.data(), fb.size()}};
    cl::Context ctx; cl::CommandQueue q; cl::Program prog; bool ok=false;
    for (auto& d : devices) {
        ctx = cl::Context(d, nullptr, nullptr, nullptr, &err);
        q = cl::CommandQueue(ctx, d, CL_QUEUE_PROFILING_ENABLE, &err);
        prog = cl::Program(ctx, {d}, bins, nullptr, &err);
        if (err == CL_SUCCESS) { ok=true; break; }
    }
    if (!ok) { std::cerr << "failed to program\n"; return 1; }
    cl::Kernel krnl_cc(prog, "control_cache_core", &err); check_cl(err, "cc");
    cl::Kernel krnl_v0(prog, "qkv_tile_kernel_cc_qwen_small_core_v8_2_s", &err); check_cl(err, "v82");

    auto mk = [](void* p, int bank){ cl_mem_ext_ptr_t e; e.obj=p; e.param=0; e.flags=bank|XCL_MEM_TOPOLOGY; return e; };
    cl_mem_ext_ptr_t hin_e=mk(hin.data(),0), hout_e=mk(hout.data(),1),
        w0_e=mk(w0.data(),2), w1_e=mk(w1.data(),3), w2_e=mk(w2.data(),4), w3_e=mk(w3.data(),5),
        opp_e=mk(op_program.data(),6), param_e=mk(param.data(),0);
    cl::Buffer hin_b(ctx, CL_MEM_EXT_PTR_XILINX|CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY, hin.size()*2, &hin_e, &err); check_cl(err,"hin");
    cl::Buffer hout_b(ctx, CL_MEM_EXT_PTR_XILINX|CL_MEM_USE_HOST_PTR|CL_MEM_READ_WRITE, hout.size()*4, &hout_e, &err); check_cl(err,"hout");
    cl::Buffer w0_b(ctx, CL_MEM_EXT_PTR_XILINX|CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY, w0.size()*2, &w0_e, &err); check_cl(err,"w0");
    cl::Buffer w1_b(ctx, CL_MEM_EXT_PTR_XILINX|CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY, w1.size()*2, &w1_e, &err); check_cl(err,"w1");
    cl::Buffer w2_b(ctx, CL_MEM_EXT_PTR_XILINX|CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY, w2.size()*2, &w2_e, &err); check_cl(err,"w2");
    cl::Buffer w3_b(ctx, CL_MEM_EXT_PTR_XILINX|CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY, w3.size()*2, &w3_e, &err); check_cl(err,"w3");
    cl::Buffer opp_b(ctx, CL_MEM_EXT_PTR_XILINX|CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY, op_program.size()*4, &opp_e, &err); check_cl(err,"opp");
    cl::Buffer param_b(ctx, CL_MEM_EXT_PTR_XILINX|CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY, param.size()*2, &param_e, &err); check_cl(err,"param");

    // cc: 0=hidden_in, 1=hidden_out, 2-5=weight_hbm_0..3, 6-13=stream(skip), 14=op_program, 15=num_ops, 16=num_tokens
    krnl_cc.setArg(0, hin_b); krnl_cc.setArg(1, hout_b);
    krnl_cc.setArg(2, w0_b); krnl_cc.setArg(3, w1_b); krnl_cc.setArg(4, w2_b); krnl_cc.setArg(5, w3_b);
    krnl_cc.setArg(14, opp_b); krnl_cc.setArg(15, (unsigned)NUM_CHUNKS); krnl_cc.setArg(16, NUM_TOKENS);
    // v82: 0-4=stream(skip), 5=param1, 6=param2, 7=num_tokens, 8=num_ops, 9-10=use_param
    krnl_v0.setArg(8, param_b); krnl_v0.setArg(9, param_b);
    krnl_v0.setArg(10, NUM_TOKENS); krnl_v0.setArg(11, (unsigned)NUM_CHUNKS); krnl_v0.setArg(12, 0u); krnl_v0.setArg(13, 0u);

    q.enqueueMigrateMemObjects({hin_b, w0_b, w1_b, w2_b, w3_b, opp_b, param_b}, 0, nullptr, nullptr); q.finish();
    cl::CommandQueue q_v0(ctx, devices[0], CL_QUEUE_PROFILING_ENABLE, &err);
    cl::CommandQueue q_cc(ctx, devices[0], CL_QUEUE_PROFILING_ENABLE, &err);
    q_v0.enqueueTask(krnl_v0); q_cc.enqueueTask(krnl_cc);
    q_v0.finish(); q_cc.finish();
    q.enqueueMigrateMemObjects({hout_b}, CL_MIGRATE_MEM_OBJECT_HOST); q.finish();

    // 校验: NUM_CHUNKS chunk × 4 = NUM_CHUNKS*4 (input=1, weight=1, NUM_CHUNKS 次累积)
    float expect_val = NUM_CHUNKS * (float)INPUT_DIM;
    bool pass = true; int bad = 0;
    for (unsigned a = 0; a < OUT_PER_OP; a++) {
        float v = hout[a] / 65536.0f;
        if (std::fabs(v - expect_val) > 0.5f) { pass = false; if (bad < 5) std::cout << "miss a=" << a << " v=" << v << " (expect " << expect_val << ")\n"; bad++; }
    }
    std::cout << "累积 (" << NUM_CHUNKS << " chunk, " << expect_val << "-element reduction): sample=" << hout[0]/65536.0f << " (expect " << expect_val << ") " << (pass ? "✅" : "❌") << "\n";
    return pass ? 0 : 1;
}
