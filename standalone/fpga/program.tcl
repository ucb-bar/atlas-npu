# Program the VCU118 over its JTAG USB port (volatile).
#   vivado -mode batch -source standalone/fpga/program.tcl -tclargs <bitfile> [jtag_serial]
# Other boards may share the hw_server: only the target with the VCU118's Digilent serial
# (210308A5E9B8 by default, or VCU118_JTAG) is opened, and only if it holds one xcvu9p.
set bitfile [file normalize [lindex $argv 0]]
if {![file exists $bitfile]} {
    puts "ERROR: $bitfile does not exist"
    exit 1
}
set serial [expr {[llength $argv] > 1 ? [lindex $argv 1] :
                  [info exists ::env(VCU118_JTAG)] ? $::env(VCU118_JTAG) : "210308A5E9B8"}]

open_hw_manager
connect_hw_server -url localhost:3121
set targets [get_hw_targets -quiet "*/Digilent/$serial"]
if {[llength $targets] != 1} {
    puts "ERROR: [llength $targets] JTAG targets match */Digilent/$serial. Targets: [get_hw_targets]"
    exit 1
}
open_hw_target $targets
set dev [get_hw_devices -quiet xcvu9p*]
if {[llength $dev] != 1} {
    puts "ERROR: target $targets holds [get_hw_devices], not one xcvu9p"
    close_hw_target
    exit 1
}
puts "TARGET $targets device $dev"
current_hw_device $dev
set_property PROGRAM.FILE $bitfile $dev
set_property PROBES.FILE {} $dev
program_hw_devices $dev
refresh_hw_device $dev
puts "RESULT done=[get_property REGISTER.CONFIG_STATUS.BIT14_DONE_PIN $dev] bit=$bitfile"
close_hw_target
