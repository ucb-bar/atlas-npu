# Re-implement a standalone bitstream from its post-synthesis checkpoint with other directives,
# for a design that fails to route (congestion) with the defaults of build.tcl.
#   vivado -mode batch -source standalone/fpga/impl_retry.tcl \
#       -tclargs <build_dir> <name> <tag> <place_directive> <route_directive> [jobs]
# Writes <build_dir>/retry-<tag>/ with the reports of build.tcl and <name>.bit.
set dir  [file normalize [lindex $argv 0]]
set name [lindex $argv 1]
set tag  [lindex $argv 2]
set pd   [lindex $argv 3]
set rd   [lindex $argv 4]
set jobs [expr {[llength $argv] > 5 ? [lindex $argv 5] : 16}]
set out  $dir/retry-$tag
file mkdir $out
set_param general.maxThreads $jobs
open_checkpoint $dir/post_synth.dcp
opt_design
place_design -directive $pd
report_design_analysis -congestion -file $out/congestion_placed.rpt
phys_opt_design
route_design -directive $rd
if {[get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]] < 0} {
    phys_opt_design
}
write_checkpoint -force $out/post_route.dcp
report_route_status -file $out/route_status.rpt
report_utilization -file $out/utilization.rpt
report_utilization -hierarchical -hierarchical_depth 4 -file $out/utilization_hier.rpt
report_timing_summary -max_paths 20 -file $out/timing_summary.rpt
write_bitstream -force $out/$name.bit
set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
set whs [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -hold]]
puts "RESULT bit=$out/$name.bit wns_ns=$wns whs_ns=$whs place=$pd route=$rd"
