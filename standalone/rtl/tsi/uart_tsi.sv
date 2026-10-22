// UART TSI target, per docs/uart-tsi-spec.md in ucb-bar/pyuartsi.
//
//   UART 8N1 -> byte queue -> 4 bytes to 1 word -> decoder (TL-UL client) -> 1 word to 4 bytes -> UART
//
// The decoder follows the spec's TSIToTileLink state machine. The bus port carries one 32-bit word,
// so every word of a command is its own Put or Get. Addresses at or above 4 GiB are not on the
// 32-bit bus: writes there are dropped and reads return 0, so the link never stalls.
module uart_tsi
    import tl_pkg::*;
#(
    parameter int CLKS_PER_BIT = 54,
    parameter int RX_DEPTH     = 32
) (
    input  logic       clk,
    input  logic       rst,
    input  logic       rxd,
    output logic       txd,

    output logic       a_valid,
    input  logic       a_ready,
    output a_t         a,
    input  logic       d_valid,
    output logic       d_ready,
    input  d_t         d,

    output logic       dropped,  // the receive queue overflowed (sticky)
    output logic       bad_cmd,  // a header had cmd other than 0 or 1 (sticky)
    output logic       done      // pulses when a command completes
);
    // Receive: UART -> byte queue -> word
    logic [7:0] rx_byte, q_byte;
    logic       rx_valid, rxq_ready, q_valid, q_ready;

    uart_rx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_rx (
        .clk, .rst, .rxd, .data(rx_byte), .valid(rx_valid));

    sync_fifo #(.W(8), .DEPTH(RX_DEPTH)) u_rxq (
        .clk, .rst,
        .wdata(rx_byte), .wvalid(rx_valid), .wready(rxq_ready),
        .rdata(q_byte), .rvalid(q_valid), .rready(q_ready));

    logic [31:0] in_word;
    logic [2:0]  in_cnt;
    logic        in_ready;
    wire         in_valid = in_cnt == 3'd4;
    wire         in_fire  = in_valid && in_ready;

    assign q_ready = !in_valid;

    always_ff @(posedge clk) begin
        if (rst) begin
            in_cnt  <= '0;
            dropped <= 1'b0;
        end else begin
            dropped <= dropped | (rx_valid && !rxq_ready);
            if (q_valid && q_ready) begin
                in_word <= {q_byte, in_word[31:8]};
                in_cnt  <= in_cnt + 1'b1;
            end else if (in_fire) begin
                in_cnt <= '0;
            end
        end
    end

    // Transmit: word -> 4 bytes -> UART
    logic [31:0] out_word;
    logic [2:0]  out_left;
    logic        out_valid, tx_ready;
    wire         out_ready = out_left == 0;

    uart_tx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_tx (
        .clk, .rst, .data(out_word[7:0]), .valid(out_left != 0), .ready(tx_ready), .txd);

    // Decoder
    typedef enum logic [3:0] {
        S_CMD, S_ADDR, S_LEN,
        S_WRITE_BODY, S_WRITE_DATA, S_WRITE_ACK,
        S_READ_REQ, S_READ_DATA, S_READ_BODY
    } state_e;

    state_e      state;
    logic [31:0] cmd, data;
    logic [63:0] addr, len, idx;
    logic        hi;  // second word of a 64-bit field

    wire on_bus = addr[63:32] == '0;

    assign in_ready  = state inside {S_CMD, S_ADDR, S_LEN, S_WRITE_BODY};
    assign a_valid   = on_bus && (state == S_WRITE_DATA || state == S_READ_REQ);
    assign a         = '{opcode:  state == S_WRITE_DATA ? A_PUT_FULL : A_GET,
                         address: addr[31:0], mask: 4'hF, data: data};
    assign d_ready   = state == S_WRITE_ACK || state == S_READ_DATA;
    assign out_valid = state == S_READ_BODY;

    wire last = idx == len;

    always_ff @(posedge clk) begin
        done <= 1'b0;
        if (rst) begin
            state   <= S_CMD;
            hi      <= 1'b0;
            bad_cmd <= 1'b0;
        end else begin
            case (state)
                S_CMD: if (in_fire) begin
                    cmd   <= in_word;
                    addr  <= '0;
                    len   <= '0;
                    state <= S_ADDR;
                end
                S_ADDR: if (in_fire) begin
                    if (hi) addr[63:32] <= in_word;
                    else    addr[31:0]  <= in_word;
                    hi <= !hi;
                    if (hi) state <= S_LEN;
                end
                S_LEN: if (in_fire) begin
                    if (hi) len[63:32] <= in_word;
                    else    len[31:0]  <= in_word;
                    hi <= !hi;
                    if (hi) begin
                        idx <= '0;
                        case (cmd)
                            32'd0:   state <= S_READ_REQ;
                            32'd1:   state <= S_WRITE_BODY;
                            default: begin
                                state   <= S_CMD;
                                bad_cmd <= 1'b1;
                            end
                        endcase
                    end
                end
                S_WRITE_BODY: if (in_fire) begin
                    data  <= in_word;
                    state <= S_WRITE_DATA;
                end
                S_WRITE_DATA: if (a_ready || !on_bus) state <= S_WRITE_ACK;
                S_WRITE_ACK: if (d_valid || !on_bus) begin
                    if (last) begin
                        state <= S_CMD;
                        done  <= 1'b1;
                    end else begin
                        idx   <= idx + 1'b1;
                        addr  <= addr + 64'd4;
                        state <= S_WRITE_BODY;
                    end
                end
                S_READ_REQ: if (a_ready || !on_bus) state <= S_READ_DATA;
                S_READ_DATA: if (d_valid || !on_bus) begin
                    data  <= on_bus ? d.data : '0;
                    state <= S_READ_BODY;
                end
                S_READ_BODY: if (out_ready) begin
                    if (last) begin
                        state <= S_CMD;
                        done  <= 1'b1;
                    end else begin
                        idx   <= idx + 1'b1;
                        addr  <= addr + 64'd4;
                        state <= S_READ_REQ;
                    end
                end
                default: state <= S_CMD;
            endcase
        end
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            out_left <= '0;
        end else if (out_valid && out_ready) begin
            out_word <= data;
            out_left <= 3'd4;
        end else if (out_left != 0 && tx_ready) begin
            out_word <= out_word >> 8;
            out_left <= out_left - 1'b1;
        end
    end
endmodule
