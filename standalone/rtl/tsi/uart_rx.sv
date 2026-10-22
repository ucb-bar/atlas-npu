// 8N1 UART receiver. Samples each bit at its midpoint.
module uart_rx #(
    parameter int CLKS_PER_BIT = 50
) (
    input  logic       clk,
    input  logic       rst,
    input  logic       rxd,
    output logic [7:0] data,
    output logic       valid
);
    localparam int CW = $clog2(CLKS_PER_BIT * 3 / 2 + 1);

    logic [2:0] sync;
    always_ff @(posedge clk) sync <= {sync[1:0], rxd};
    wire rx = sync[2];

    typedef enum logic [1:0] {IDLE, START, DATA, STOP} state_t;
    state_t        state;
    logic [CW-1:0] cnt;
    logic [2:0]    bit_idx;
    logic [7:0]    shreg;

    always_ff @(posedge clk) begin
        valid <= 1'b0;
        if (rst) begin
            state <= IDLE;
            cnt   <= '0;
        end else begin
            case (state)
                IDLE: if (!rx) begin
                    state <= START;
                    cnt   <= CW'(CLKS_PER_BIT / 2 - 1);
                end
                START: if (cnt == 0) begin
                    if (!rx) begin
                        state   <= DATA;
                        cnt     <= CW'(CLKS_PER_BIT - 1);
                        bit_idx <= '0;
                    end else begin
                        state <= IDLE;
                    end
                end else cnt <= cnt - 1'b1;
                DATA: if (cnt == 0) begin
                    shreg <= {rx, shreg[7:1]};
                    cnt   <= CW'(CLKS_PER_BIT - 1);
                    if (bit_idx == 3'd7) state <= STOP;
                    bit_idx <= bit_idx + 1'b1;
                end else cnt <= cnt - 1'b1;
                STOP: if (cnt == 0) begin
                    state <= IDLE;
                    if (rx) begin
                        data  <= shreg;
                        valid <= 1'b1;
                    end
                end else cnt <= cnt - 1'b1;
            endcase
        end
    end
endmodule
