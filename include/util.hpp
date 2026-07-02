#ifndef LLM_FPGA_UTIL_HPP
#define LLM_FPGA_UTIL_HPP

#define _LABEL_FOR_EACH_HELPER(line, var) _ln##line##_for_each_##var
#define _LABEL_FOR_EACH(line, var) _LABEL_FOR_EACH_HELPER(line, var)
#define FOR_EACH(var, limit) _LABEL_FOR_EACH(__LINE__, var): for (unsigned int var = 0; var < (limit); var++)

#define _LABEL_FOR_BLOCK_HELPER(line, var) _ln##line##_for_block_##var
#define _LABEL_FOR_BLOCK(line, var) _LABEL_FOR_BLOCK_HELPER(line, var)
#define FOR_BLOCK(var, limit, block_size) \
    constexpr unsigned int var##_step = (block_size); \
    constexpr unsigned int var##_limit = (limit); \
    _LABEL_FOR_BLOCK(__LINE__, var): for ( \
        unsigned int var##_base = 0, var##_block = 0; \
        var##_base < var##_limit; \
        var##_base += var##_step, var##_block++ \
    )

#define _LABEL_FOR_OFFSET_HELPER(line, var) _ln##line##_for_offset_##var
#define _LABEL_FOR_OFFSET(line, var) _LABEL_FOR_OFFSET_HELPER(line, var)
#define FOR_OFFSET(var) \
    _LABEL_FOR_OFFSET(__LINE__, var): for ( \
        unsigned int var##_offset = 0, var = var##_base; \
        var##_offset < var##_step; \
        var##_offset++, var++ \
    ) \
    if (var##_limit % var##_step == 0 || var < var##_limit)

template <typename T>
constexpr T ceildiv(T dividend, T divisor) {
    return (dividend + divisor - 1) / divisor;
}

template <typename T>
constexpr T max_const(T a, T b) {
    return (a > b) ? a : b;
}

#endif
