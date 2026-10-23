# The VCU118 SGMII PCS/PMA (1000BASE-X PCS/PMA or SGMII over LVDS, to the DP83867 PHY) as IP
# gig_ethernet_pcs_pma_0, in $ipdir (standalone/rtl/eth/vcu118_sgmii.sv).
# Settings from verilog-ethernet example/VCU118/fpga_1g/ip/gig_ethernet_pcs_pma_0.tcl (MIT,
# upstream 77320a9): 625 MHz reference clock from the PHY, shared logic in the core.
create_ip -vendor xilinx.com -library ip -name gig_ethernet_pcs_pma -module_name gig_ethernet_pcs_pma_0 \
    -dir $ipdir -force
set_property -dict [list \
    CONFIG.Standard             {SGMII} \
    CONFIG.Physical_Interface   {LVDS} \
    CONFIG.Management_Interface {false} \
    CONFIG.SupportLevel         {Include_Shared_Logic_in_Core} \
    CONFIG.LvdsRefClk           {625} \
    CONFIG.TxLane0_Placement    {DIFF_PAIR_2} \
    CONFIG.RxLane0_Placement    {DIFF_PAIR_0} \
    CONFIG.Tx_In_Upper_Nibble   {0} \
] [get_ips gig_ethernet_pcs_pma_0]
