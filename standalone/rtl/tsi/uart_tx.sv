// 8N1 UART transmitter. Each bit, including the stop bit, is held for CLKS_PER_BIT cycles.
module uart_tx #(
    parameter int CLKS_PER_BIT = 50
) (
    input  logic       clk,
    input  logic       rst,
    input  logic [7:0] data,
    input  logic       valid,
    output logic       ready,
    output logic       txd
);
    localparam int CW = $clog2(CLKS_PER_BIT);

    logic [9:0]    shreg;
    logic [3:0]    bits_left;
    logic [CW-1:0] cnt;

    assign ready = (bits_left == 0) && (cnt == 0);

    always_ff @(posedge clk) begin
        if (rst) begin
            txd       <= 1'b1;
            bits_left <= '0;
            cnt       <= '0;
        end else if (cnt != 0) begin
            cnt <= cnt - 1'b1;
        end else if (bits_left != 0) begin
            txd       <= shreg[0];
            shreg     <= {1'b1, shreg[9:1]};
            bits_left <= bits_left - 1'b1;
            cnt       <= CW'(CLKS_PER_BIT - 1);
        end else if (valid) begin
            shreg     <= {1'b1, data, 1'b0};
            bits_left <= 4'd10;
        end
    end
endmodule
