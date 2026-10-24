# Build the VCU118 bitstream of the standalone Atlas SoC (non-project flow).
#   vivado -mode batch -source standalone/fpga/build.tcl -tclargs <rtl_dir> <out_dir> [clk_mhz] [jobs] [config] \
#       [place_directive] [route_directive]
# DRAM is 2 GiB of the board's DDR4 C1 through the DDR4 MIG (top vcu118_atlas_top). Ethernet (TSI
# over UDP) goes through the SGMII PCS/PMA IP and verilog-ethernet (rtl/eth/vendor).
# rtl_dir is the Chisel output without assertions, after fpga_mems.py (build_bitstream.sh does both):
#   ./mill standalone.runMain atlas.standalone.EmitAtlasCore build/rtl-fpga
#   python3 standalone/fpga/fpga_mems.py build/rtl-fpga
set rtl     [file normalize [lindex $argv 0]]
set out     [file normalize [lindex $argv 1]]
set clk_mhz [expr {[llength $argv] > 2 ? [lindex $argv 2] : 50}]
set jobs    [expr {[llength $argv] > 3 ? [lindex $argv 3] : 16}]
# (not `config`: the DDR4 IP's scripts set a global of that name)
set atlas_config [expr {[llength $argv] > 4 ? [lindex $argv 4] : "default"}]
# ipt-nvfp4 routes only with SSI_SpreadLogic_high placement and Explore routing.
set place_dir [expr {[llength $argv] > 5 ? [lindex $argv 5] : "Default"}]
set route_dir [expr {[llength $argv] > 6 ? [lindex $argv 6] : "Default"}]

set root [file normalize [file join [file dirname [info script]] ../..]]
set sa   "$root/standalone"
set part xcvu9p-flga2104-2L-e
set top  vcu118_atlas_top

file mkdir $out
set_param general.maxThreads $jobs
# firtool emits each VMEM bank (8192 x 256) and IMEM (32768 x 32) as one array, past synthesis'
# default 1M-bit variable limit. (Chipyard's memory compiler splits them; this flow does not.)
set_param synth.elaboration.rodinMoreOptions "rt::set_parameter var_size_limit 8388608"

# The MIG and the PCS/PMA are IPs: generate and synthesize them out of context in an in-memory
# project.
create_project -in_memory -part $part
set ipdir $out/ip
file mkdir $ipdir
source $sa/fpga/vcu118_ddr4_mig.tcl
source $sa/fpga/vcu118_sgmii_pcs.tcl
foreach ip {vcu118mig gig_ethernet_pcs_pma_0} {
    generate_target {instantiation_template synthesis implementation} [get_ips $ip]
    synth_ip -force [get_ips $ip]
}
# atlas_dram.sv (the on-chip DRAM of the simulation model) is read for atlas_soc's unused branch.
read_verilog -sv [list \
    $sa/rtl/tsi/tl_pkg.sv $sa/rtl/tsi/uart_rx.sv $sa/rtl/tsi/uart_tx.sv $sa/rtl/tsi/sync_fifo.sv \
    $sa/rtl/tsi/uart_tsi.sv $sa/rtl/tsi/tl_xbar.sv $sa/rtl/tsi/tl_widen.sv $sa/rtl/tsi/tl_narrow.sv \
    $sa/rtl/tsi/tl_arbiter.sv $sa/rtl/atlas_tl_port.sv $sa/rtl/atlas_dram.sv $sa/rtl/atlas_sys_regs.sv \
    $sa/rtl/atlas_soc.sv $sa/rtl/atlas_tsi_soc.sv \
    $sa/rtl/eth/eth_tsi.sv $sa/rtl/eth/eth_udp_stack.sv $sa/rtl/eth/vcu118_sgmii.sv \
    $sa/rtl/async_fifo.sv $sa/rtl/atlas_ddr4_dram.sv $sa/rtl/vcu118_atlas_top.sv]
read_verilog [lsort [glob $sa/rtl/eth/vendor/*.v]]
# SystemVerilog packages (dotprod_pkg of the NVFP4 MXU) before their users.
set pkgs [glob -nocomplain $rtl/*_pkg.sv]
set rest [lsort [glob $rtl/*.sv]]
foreach f $pkgs { set rest [lsearch -all -inline -not -exact $rest $f] }
read_verilog -sv [concat $pkgs $rest]
read_xdc $sa/fpga/vcu118_atlas.xdc
read_xdc $sa/fpga/vcu118_ddr4.xdc
read_xdc $sa/fpga/vcu118_eth.xdc
auto_detect_xpm

synth_design -top $top -part $part -generic CLK_MHZ=$clk_mhz
# verilog-ethernet's constraint scripts (Tcl, they search the synthesized netlist): CDC of its
# async FIFOs, reset synchronizers.
foreach f {axis_async_fifo.tcl eth_mac_fifo.tcl sync_reset.tcl} {
    source $sa/rtl/eth/vendor/$f
}
# Clock domains: the design clock, the MIG user clock (atlas_ddr4_dram's gray-coded FIFOs) and
# the PCS/PMA clocks, all derived from the PHY's 625 MHz clock (eth_udp_stack's async FIFOs).
set_clock_groups -asynchronous \
    -group [get_clocks -of_objects [get_pins u_mmcm/CLKOUT0]] \
    -group [get_clocks -of_objects [get_pins u_mig/c0_ddr4_ui_clk]] \
    -group [get_clocks -include_generated_clocks -of_objects [get_ports phy_sgmii_clk_p]]
report_clocks -file $out/clocks_post_synth.rpt
write_checkpoint -force $out/post_synth.dcp
report_utilization -file $out/utilization_synth.rpt

opt_design
place_design -directive $place_dir
phys_opt_design
route_design -directive $route_dir
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

set name atlas_standalone_vcu118_${atlas_config}_ddr4_${clk_mhz}mhz
write_bitstream -force $out/$name.bit
set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
set whs [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -hold]]
puts "RESULT bit=$out/$name.bit wns_ns=$wns whs_ns=$whs"
