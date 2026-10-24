# VCU118 Ethernet: the DP83867 PHY over SGMII (standalone/rtl/eth/vcu118_sgmii.sv), bank 64 (HP,
# VCCO 1.8 V). Pins from verilog-ethernet example/VCU118/fpga_1g/fpga.xdc (MIT, upstream 77320a9).
# The SGMII lines are AC-coupled, so the RX data and 625 MHz clock inputs get the 100 ohm internal
# termination (as in taxi and the Xilinx VCU118 board file) and DQS_BIAS for their DC level (PG047).
# The PCS/PMA IP creates the 625 MHz clock on phy_sgmii_clk_p.

set_property PACKAGE_PIN AU24 [get_ports phy_sgmii_rx_p]
set_property PACKAGE_PIN AV24 [get_ports phy_sgmii_rx_n]
set_property PACKAGE_PIN AU21 [get_ports phy_sgmii_tx_p]
set_property PACKAGE_PIN AV21 [get_ports phy_sgmii_tx_n]
set_property PACKAGE_PIN AT22 [get_ports phy_sgmii_clk_p]
set_property PACKAGE_PIN AU22 [get_ports phy_sgmii_clk_n]
set_property IOSTANDARD LVDS [get_ports {phy_sgmii_rx_p phy_sgmii_rx_n phy_sgmii_tx_p phy_sgmii_tx_n}]
set_property IOSTANDARD LVDS [get_ports {phy_sgmii_clk_p phy_sgmii_clk_n}]
set_property -dict {DIFF_TERM_ADV TERM_100 DQS_BIAS TRUE} [get_ports {phy_sgmii_rx_p phy_sgmii_rx_n}]
set_property -dict {DIFF_TERM_ADV TERM_100 DQS_BIAS TRUE} [get_ports {phy_sgmii_clk_p phy_sgmii_clk_n}]

set_property PACKAGE_PIN BA21 [get_ports phy_reset_n]
set_property PACKAGE_PIN AR24 [get_ports phy_int_n]
set_property PACKAGE_PIN AR23 [get_ports phy_mdio]
set_property PACKAGE_PIN AV23 [get_ports phy_mdc]
set_property IOSTANDARD LVCMOS18 [get_ports {phy_reset_n phy_int_n phy_mdio phy_mdc}]
set_property SLEW SLOW [get_ports {phy_reset_n phy_mdio phy_mdc}]
set_property DRIVE 8 [get_ports {phy_reset_n phy_mdio phy_mdc}]

# Reset and MDIO are slow (MDC 2.5 MHz, data changes half an MDC period before the rising edge).
set_false_path -to [get_ports {phy_reset_n phy_mdio phy_mdc}]
set_false_path -from [get_ports {phy_int_n phy_mdio}]
