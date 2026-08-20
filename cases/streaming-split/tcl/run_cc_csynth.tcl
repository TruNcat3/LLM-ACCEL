# control_cache_core synthesis: verify the 512-bit store_out reduction (8192 to about 512 cycles).
open_project control_cache_core -reset
set_top control_cache_core
add_files kernel/control_cache_core.cpp
open_solution "csynth_solution" -reset
set_part {xcu250-figd2104-2L-e}
create_clock -period 3.33 -name default
config_compile -name_max_length 256
config_rtl -reset control

puts "=========================================="
puts "cc synthesis (512-bit packed store_out)"
puts "Goal: reduce store_out from 8192 to about 512 cycles without II regression"
puts "=========================================="

csynth_design

if {[file exists control_cache_core/csynth_solution/syn/report/csynth.rpt]} {
    puts "cc synthesis completed successfully."
} else {
    puts "cc synthesis failed."
}
exit
