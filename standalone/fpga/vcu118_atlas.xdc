# VCU118 pins of the standalone Atlas SoC (standalone/rtl/vcu118_atlas_top.sv).
# Pin choices follow fpga-shells' VCU118 shell (sys_clock, CPU_RESET, CP2105 UART, GPIO LEDs).

set_property PACKAGE_PIN E12 [get_ports sys_clock_p]
set_property PACKAGE_PIN D12 [get_ports sys_clock_n]
set_property IOSTANDARD DIFF_SSTL12 [get_ports {sys_clock_p sys_clock_n}]
create_clock -period 4.000 -name sys_clock [get_ports sys_clock_p]

set_property PACKAGE_PIN L19 [get_ports reset]
set_property IOSTANDARD LVCMOS12 [get_ports reset]

set_property PACKAGE_PIN AW25 [get_ports uart_rxd]
set_property PACKAGE_PIN BB21 [get_ports uart_txd]
set_property PACKAGE_PIN BB22 [get_ports uart_rtsn]
set_property IOSTANDARD LVCMOS18 [get_ports {uart_rxd uart_txd uart_rtsn}]
set_property IOB TRUE [get_ports uart_txd]

set_property PACKAGE_PIN AT32 [get_ports {led[0]}]
set_property PACKAGE_PIN AV34 [get_ports {led[1]}]
set_property PACKAGE_PIN AY30 [get_ports {led[2]}]
set_property PACKAGE_PIN BB32 [get_ports {led[3]}]
set_property PACKAGE_PIN BF32 [get_ports {led[4]}]
set_property PACKAGE_PIN AU37 [get_ports {led[5]}]
set_property PACKAGE_PIN AV36 [get_ports {led[6]}]
set_property PACKAGE_PIN BA37 [get_ports {led[7]}]
set_property IOSTANDARD LVCMOS12 [get_ports {led[*]}]

# The pushbutton and the UART are asynchronous to the design clock (both are synchronized inside).
set_false_path -from [get_ports {reset uart_rxd}]
set_false_path -to [get_ports {uart_txd uart_rtsn led[*]}]

set_property BITSTREAM.GENERAL.COMPRESS TRUE [current_design]
