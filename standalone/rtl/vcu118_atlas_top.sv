// VCU118 top of the standalone Atlas SoC.
//
//   sys_clock_p/n  250 MHz differential (E12/D12) -> MMCM (VCO 1000 MHz) -> CLK_MHZ design clock
//   reset          CPU_RESET pushbutton (L19), active high
//   uart_rxd/txd   CP2105 UART (AW25/BB21), the TSI link at BAUD 8N1; RTS# (BB22) held asserted
//   led[0]         heartbeat (about 1.5 Hz)
//   led[1]         TSI activity: toggles per completed command
//   led[2]         link error or denied access
//   led[3]         MMCM locked
//   led[4]         Atlas halted
//   led[7:5]       user LEDs (sys_regs LEDS)
module vcu118_atlas_top #(
    parameter int CLK_MHZ     = 50,
    parameter int BAUD        = 921_600,
    parameter int DRAM_LINE_W = 17
) (
    input  logic       sys_clock_p,
    input  logic       sys_clock_n,
    input  logic       reset,
    input  logic       uart_rxd,
    output logic       uart_txd,
    output logic       uart_rtsn,
    output logic [7:0] led
);
    logic clk_in, clk_fb, clk_mmcm, clk, locked;

    IBUFDS u_ibufds (.I(sys_clock_p), .IB(sys_clock_n), .O(clk_in));

    MMCME4_BASE #(
        .CLKIN1_PERIOD    (4.0),
        .DIVCLK_DIVIDE    (1),
        .CLKFBOUT_MULT_F  (4.0),
        .CLKOUT0_DIVIDE_F (1000.0 / CLK_MHZ)
    ) u_mmcm (
        .CLKIN1   (clk_in),
        .CLKFBIN  (clk_fb),
        .CLKFBOUT (clk_fb),
        .CLKOUT0  (clk_mmcm),
        .LOCKED   (locked),
        .RST      (1'b0),
        .PWRDWN   (1'b0),
        .CLKFBOUTB(), .CLKOUT0B(), .CLKOUT1(), .CLKOUT1B(), .CLKOUT2(), .CLKOUT2B(),
        .CLKOUT3(), .CLKOUT3B(), .CLKOUT4(), .CLKOUT5(), .CLKOUT6());

    BUFG u_bufg (.I(clk_mmcm), .O(clk));

    // Reset: held until the MMCM locks and for 256 cycles after, and while CPU_RESET is pressed.
    (* ASYNC_REG = "TRUE" *) logic [1:0] btn_sync;
    logic [8:0] por;
    logic       rst;
    always_ff @(posedge clk) btn_sync <= {btn_sync[0], reset};
    always_ff @(posedge clk or negedge locked) begin
        if (!locked)          por <= '0;
        else if (btn_sync[1]) por <= '0;
        else if (!por[8])     por <= por + 1'b1;
    end
    always_ff @(posedge clk) rst <= !por[8];

    logic       activity, error, halted;
    logic [2:0] user_leds;

    atlas_tsi_soc #(
        .CLK_HZ(CLK_MHZ * 1_000_000), .BAUD(BAUD), .DRAM_LINE_W(DRAM_LINE_W)
    ) u_soc (
        .clk, .rst, .rxd(uart_rxd), .txd(uart_txd),
        .activity, .error, .halted, .user_leds);

    logic [24:0] beat;
    always_ff @(posedge clk) beat <= beat + 1'b1;

    assign uart_rtsn = 1'b0;
    assign led       = {user_leds, halted, locked, error, activity, beat[24]};
endmodule
