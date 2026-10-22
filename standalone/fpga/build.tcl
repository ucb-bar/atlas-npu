# Build the VCU118 bitstream of the standalone Atlas SoC (non-project flow).
#   vivado -mode batch -source standalone/fpga/build.tcl -tclargs <rtl_dir> <out_dir> [clk_mhz] [jobs]
# rtl_dir is the Chisel output without assertions:
#   ./mill standalone.runMain atlas.standalone.EmitAtlasCore build/rtl-fpga
set rtl     [file normalize [lindex $argv 0]]
set out     [file normalize [lindex $argv 1]]
set clk_mhz [expr {[llength $argv] > 2 ? [lindex $argv 2] : 50}]
set jobs    [expr {[llength $argv] > 3 ? [lindex $argv 3] : 16}]

set root [file normalize [file join [file dirname [info script]] ../..]]
set sa   "$root/standalone"
set part xcvu9p-flga2104-2L-e
set top  vcu118_atlas_top

file mkdir $out
set_param general.maxThreads $jobs
# firtool emits each VMEM bank (8192 x 256) and IMEM (32768 x 32) as one array, past synthesis'
# default 1M-bit variable limit. (Chipyard's memory compiler splits them; this flow does not.)
set_param synth.elaboration.rodinMoreOptions "rt::set_parameter var_size_limit 8388608"

read_verilog -sv [list \
    $sa/rtl/tsi/tl_pkg.sv $sa/rtl/tsi/uart_rx.sv $sa/rtl/tsi/uart_tx.sv $sa/rtl/tsi/sync_fifo.sv \
    $sa/rtl/tsi/uart_tsi.sv $sa/rtl/tsi/tl_xbar.sv $sa/rtl/tsi/tl_widen.sv $sa/rtl/tsi/tl_narrow.sv \
    $sa/rtl/atlas_tl_port.sv $sa/rtl/atlas_dram.sv $sa/rtl/atlas_sys_regs.sv \
    $sa/rtl/atlas_soc.sv $sa/rtl/atlas_tsi_soc.sv $sa/rtl/vcu118_atlas_top.sv]
read_verilog -sv [glob $rtl/*.sv]
read_xdc $sa/fpga/vcu118_atlas.xdc
auto_detect_xpm

synth_design -top $top -part $part -generic CLK_MHZ=$clk_mhz
write_checkpoint -force $out/post_synth.dcp
report_utilization -file $out/utilization_synth.rpt

opt_design
place_design
phys_opt_design
route_design
if {[get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]] < 0} {
    phys_opt_design
}
write_checkpoint -force $out/post_route.dcp

report_utilization -file $out/utilization.rpt
report_utilization -hierarchical -hierarchical_depth 4 -file $out/utilization_hier.rpt
report_timing_summary -max_paths 20 -file $out/timing_summary.rpt
report_clock_utilization -file $out/clocks.rpt
report_drc -file $out/drc.rpt
report_io -file $out/io.rpt

set name atlas_standalone_vcu118_${clk_mhz}mhz
write_bitstream -force $out/$name.bit
set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
set whs [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -hold]]
puts "RESULT bit=$out/$name.bit wns_ns=$wns whs_ns=$whs"
