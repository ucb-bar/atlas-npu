// TSI over UDP (docs/tsi-over-udp.md): runs the TSI commands of each datagram on the beat fabric
// and sends a reply datagram. eth_udp_stack adds the prefix on rx and removes it on tx.
//
//   rx  peer IP, {length, peer port}, seq, flags, TSI commands
//   tx  peer IP, {16'h0, peer port}, seq, flags, expect, status, read data
//
// Datagrams run in sequence order, each once. SYNC opens a session for its peer (IP and port):
// the next seq to run (expect) is seq + 1. A copy of that SYNC (same peer and seq) gets SYNC with
// the current expect and does not reopen the session; any other SYNC opens a new one. In the
// session, seq == expect runs; an older seq gets DUP and does not run again (its reply was lost);
// a newer seq gets DROP (an earlier datagram was lost). Other peers get NOSYNC. Datagrams shorter
// than seq and flags, or not whole words, are dropped without a reply.
//
// A datagram that ends before its length says (rx_abort: eth_udp_stack cut it; or an early
// rx_last) never uses its last word. Cut in seq or flags, it is dropped without a reply. Cut in
// the commands, it replies CUT: the words before the cut were used. Words past the length are
// ignored.
//
// The reply goes out after every command completed, so it also acknowledges the writes. Until
// then the read data waits in a block RAM of REPLY_WORDS words. Reads past it set OVERFLOW and the
// rest of those reads is skipped.
//
// The decoder takes one word per cycle while the fabric keeps up. VMEM and DRAM (address >=
// 0x2000_0000) are accessed by line: the write words of one line merge into one masked Put, which
// goes out while the next line fills, and a read Gets each line once. The word devices below
// 0x2000_0000 get one transaction per word, as from tl_widen. Addresses at or above 4 GiB are not
// on the bus: writes are dropped and reads return 0, as in uart_tsi.
module eth_tsi
    import tl_pkg::*;
#(
    parameter int REPLY_WORDS = 364
) (
    input  logic        clk,
    input  logic        rst,

    input  logic [31:0] rx_data,
    input  logic        rx_valid,
    output logic        rx_ready,
    input  logic        rx_last,
    input  logic        rx_abort,   // with rx_last: the datagram was cut short

    output logic [31:0] tx_data,
    output logic        tx_valid,
    input  logic        tx_ready,
    output logic        tx_last,

    output logic        a_valid,
    input  logic        a_ready,
    output wa_t         a,
    input  logic        d_valid,
    output logic        d_ready,
    input  wd_t         d,

    output logic        synced,     // a session is open
    output logic [31:0] n_exec,     // datagrams run
    output logic [31:0] n_dup,      // datagrams acknowledged DUP
    output logic [31:0] n_drop,     // datagrams dropped or not run
    output logic        done        // pulses when a reply has been sent
);
    localparam int NW = $clog2(REPLY_WORDS + 1);  // read words in the reply
    localparam int OW = $clog2(REPLY_WORDS + 6);  // reply word index
    localparam int AW = $clog2(REPLY_WORDS);

    localparam logic [3:0] SYNC = 4'h1, DUP = 4'h2, NOSYNC = 4'h4, DROP = 4'h8;

    typedef enum logic [2:0] {F_IP, F_LEN, F_SEQ, F_FLAGS, F_SKIP, F_EXEC, F_REPLY} frame_e;
    typedef enum logic [3:0] {
        D_CMD, D_ADDR0, D_ADDR1, D_LEN0, D_LEN1, D_WR, D_RD, D_RD_WAIT, D_BAD
    } dec_e;

    // ── Datagram ────────────────────────────────────────────────────────────
    frame_e      fs;
    logic [31:0] peer_ip, seq, next_seq;
    logic [15:0] peer_port;
    logic [31:0] s_ip, s_seq;  // the session: peer and SYNC seq
    logic [15:0] s_port;
    logic [3:0]  rflags;
    logic [13:0] rx_left;    // words of the datagram from this one on, by its length
    logic        rx_more;    // the datagram has words after the last one taken
    logic        reply;      // F_SKIP: send a reply after the last word
    logic        frame_end;  // F_EXEC: the last word has been taken
    logic [15:0] n_cmds;
    logic        cut, bad_cmd, overflow, denied;
    logic [NW-1:0] n_rd;
    logic [OW-1:0] oi;

    // ── Decoder ─────────────────────────────────────────────────────────────
    dec_e        ds;
    logic        is_write;
    logic [63:0] addr, left;  // address of the current word, words left after it

    wire       on_bus = addr[63:32] == '0;
    wire       wide   = addr[31:29] != 3'd0;
    wire [2:0] lane   = addr[4:2];
    wire       last   = left == '0;

    // Fabric request, one in flight
    wa_t  req;
    logic req_valid, req_sent;
    assign a_valid = req_valid && !req_sent;
    assign a       = req;
    assign d_ready = req_sent;
    wire   d_fire  = d_valid && d_ready;

    // Write line being merged; closed when its last word is in
    logic [255:0] wbuf;
    logic [31:0]  wmask;
    logic [26:0]  wline;
    logic         wbuf_valid, wbuf_close;

    // Line of the last Get
    logic [255:0] rbuf;
    logic [26:0]  rline;
    logic         rbuf_valid;

    logic dec_ready;
    always_comb begin
        case (ds)
            D_CMD, D_ADDR0, D_ADDR1, D_LEN0, D_LEN1, D_BAD: dec_ready = 1'b1;
            D_WR: dec_ready = !on_bus ||
                              (wide ? !wbuf_valid || !wbuf_close || !req_valid
                                    : !wbuf_valid && !req_valid);
            default: dec_ready = 1'b0;
        endcase
        case (fs)
            F_IP, F_LEN, F_SEQ, F_FLAGS, F_SKIP: rx_ready = 1'b1;
            F_EXEC:  rx_ready = dec_ready && !frame_end;
            default: rx_ready = 1'b0;
        endcase
    end

    wire rx_fire  = rx_valid && rx_ready;
    wire len_last = rx_left == 14'd1;
    wire cut_end  = rx_last && (rx_abort || !len_last);  // the word is not used
    wire start    = fs == F_FLAGS && rx_fire;
    wire beat     = fs == F_EXEC && rx_fire;
    wire take     = beat && !cut_end;

    // The datagram is over: its last word is taken and no read is running.
    wire ending  = fs == F_EXEC && frame_end &&
                   ds inside {D_CMD, D_ADDR0, D_ADDR1, D_LEN0, D_LEN1, D_WR, D_BAD};
    wire finish  = ending && !wbuf_valid && !req_valid;

    wire put_word  = take && ds == D_WR && on_bus && !wide;
    wire put_line  = take && ds == D_WR && on_bus && wide;
    wire flush     = wbuf_valid && !req_valid && (wbuf_close || ending);

    wire full      = n_rd == NW'(REPLY_WORDS);
    wire hit       = rbuf_valid && rline == addr[31:5];
    wire get_now   = ds == D_RD && !full && on_bus && !(wide && hit) && !req_valid && !wbuf_valid;

    logic        push;
    logic [31:0] push_word;
    always_comb begin
        push      = 1'b0;
        push_word = '0;
        if (ds == D_RD && !full) begin
            push      = !on_bus || (wide && hit);
            push_word = on_bus ? rbuf[32 * lane +: 32] : '0;
        end else if (ds == D_RD_WAIT && d_fire) begin
            push      = 1'b1;
            push_word = d.data[32 * lane +: 32];
        end
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            req_valid <= 1'b0;
            req_sent  <= 1'b0;
        end else begin
            if (a_valid && a_ready) req_sent <= 1'b1;
            if (d_fire) begin
                req_valid <= 1'b0;
                req_sent  <= 1'b0;
            end
            if (flush || put_word || get_now) req_valid <= 1'b1;
        end
        if (flush)
            req <= '{opcode: A_PUT_FULL, address: {wline, 5'd0}, mask: wmask, data: wbuf};
        else if (put_word)
            req <= '{opcode: A_PUT_FULL, address: {addr[31:2], 2'b00},
                     mask: 32'hF << (4 * lane), data: {8{rx_data}}};
        else if (get_now)
            req <= '{opcode: A_GET, address: wide ? {addr[31:5], 5'd0} : {addr[31:2], 2'b00},
                     mask: wide ? '1 : 32'hF << (4 * lane), data: '0};
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            wbuf_valid <= 1'b0;
            wbuf_close <= 1'b0;
        end else begin
            if (flush) wbuf_valid <= 1'b0;
            if (put_line) begin
                wbuf_valid <= 1'b1;
                wbuf_close <= lane == 3'd7 || last;
            end
        end
        if (put_line) begin
            wbuf[32 * lane +: 32] <= rx_data;
            wline                 <= addr[31:5];
            wmask                 <= (wbuf_valid && !flush ? wmask : '0) | (32'hF << (4 * lane));
        end
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            ds         <= D_CMD;
            rbuf_valid <= 1'b0;
        end else if (start) begin
            ds         <= D_CMD;
            rbuf_valid <= 1'b0;
            n_cmds     <= '0;
            n_rd       <= '0;
            cut        <= 1'b0;
            bad_cmd    <= 1'b0;
            overflow   <= 1'b0;
            denied     <= 1'b0;
        end else begin
            if (d_fire && d.denied) denied <= 1'b1;
            if (push) n_rd <= n_rd + 1'b1;
            if (beat && cut_end) cut <= 1'b1;
            if (finish) begin
                if (!(ds inside {D_CMD, D_BAD})) cut <= 1'b1;
                ds <= D_CMD;
            end else begin
                case (ds)
                    D_CMD: if (take) begin
                        is_write <= rx_data[0];
                        if (rx_data[31:1] == '0) begin
                            ds <= D_ADDR0;
                        end else begin
                            bad_cmd <= 1'b1;   // skip the rest of the datagram
                            ds      <= D_BAD;
                        end
                    end
                    D_ADDR0: if (take) begin addr[31:0]  <= rx_data; ds <= D_ADDR1; end
                    D_ADDR1: if (take) begin addr[63:32] <= rx_data; ds <= D_LEN0;  end
                    D_LEN0:  if (take) begin left[31:0]  <= rx_data; ds <= D_LEN1;  end
                    D_LEN1:  if (take) begin
                        left[63:32] <= rx_data;
                        rbuf_valid  <= 1'b0;
                        ds          <= is_write ? D_WR : D_RD;
                    end
                    D_WR: if (take) begin
                        if (last) begin
                            n_cmds <= n_cmds + 1'b1;
                            ds     <= D_CMD;
                        end else begin
                            addr <= addr + 64'd4;
                            left <= left - 1'b1;
                        end
                    end
                    D_RD, D_RD_WAIT: begin
                        if (ds == D_RD && full) begin
                            overflow <= 1'b1;
                            n_cmds   <= n_cmds + 1'b1;
                            ds       <= D_CMD;
                        end else if (push) begin
                            if (last) begin
                                n_cmds <= n_cmds + 1'b1;
                                ds     <= D_CMD;
                            end else begin
                                addr <= addr + 64'd4;
                                left <= left - 1'b1;
                                ds   <= D_RD;
                            end
                        end else if (get_now) begin
                            ds <= D_RD_WAIT;
                        end
                        if (ds == D_RD_WAIT && d_fire && wide) begin
                            rbuf       <= d.data;
                            rline      <= addr[31:5];
                            rbuf_valid <= 1'b1;
                        end
                    end
                    default: ;
                endcase
            end
        end
    end

    // ── Frame ───────────────────────────────────────────────────────────────
    wire [31:0] diff    = next_seq - seq;
    wire        peer_ok = synced && peer_ip == s_ip && peer_port == s_port;
    wire        tx_fire = tx_valid && tx_ready;

    always_ff @(posedge clk) begin
        done <= 1'b0;
        if (rx_fire) begin
            rx_left <= fs == F_LEN ? rx_data[31:18] : rx_left - 1'b1;
            rx_more <= !rx_last;
        end
        if (rst) begin
            fs       <= F_IP;
            synced   <= 1'b0;
            next_seq <= '0;
            n_exec   <= '0;
            n_dup    <= '0;
            n_drop   <= '0;
            oi       <= '0;
        end else begin
            case (fs)
                F_IP: if (rx_fire) begin
                    peer_ip <= rx_data;
                    if (rx_last) n_drop <= n_drop + 1'b1;
                    else         fs     <= F_LEN;
                end
                F_LEN: if (rx_fire) begin
                    peer_port <= rx_data[15:0];
                    if (rx_last) begin
                        n_drop <= n_drop + 1'b1;
                        fs     <= F_IP;
                    end else if (rx_data[31:16] < 16'd8 || rx_data[17:16] != 2'd0) begin
                        n_drop <= n_drop + 1'b1;
                        reply  <= 1'b0;
                        fs     <= F_SKIP;
                    end else begin
                        fs <= F_SEQ;
                    end
                end
                F_SEQ: if (rx_fire) begin
                    seq <= rx_data;
                    if (rx_last) begin
                        n_drop <= n_drop + 1'b1;
                        fs     <= F_IP;
                    end else begin
                        fs <= F_FLAGS;
                    end
                end
                F_FLAGS: if (rx_fire) begin
                    frame_end <= len_last;
                    reply     <= 1'b1;
                    fs        <= rx_last ? F_REPLY : F_SKIP;
                    if (cut_end) begin
                        n_drop <= n_drop + 1'b1;
                        fs     <= F_IP;
                    end else if (rx_data[0]) begin
                        rflags <= SYNC;
                        if (!peer_ok || seq != s_seq) begin
                            synced   <= 1'b1;
                            next_seq <= seq + 1'b1;
                            s_ip     <= peer_ip;
                            s_port   <= peer_port;
                            s_seq    <= seq;
                        end
                    end else if (!peer_ok) begin
                        rflags <= NOSYNC;
                        n_drop <= n_drop + 1'b1;
                    end else if (diff == '0) begin
                        rflags <= '0;
                        fs     <= F_EXEC;
                    end else if (!diff[31]) begin
                        rflags <= DUP;
                        n_dup  <= n_dup + 1'b1;
                    end else begin
                        rflags <= DROP;
                        n_drop <= n_drop + 1'b1;
                    end
                end
                F_SKIP: if (rx_fire && rx_last) fs <= reply ? F_REPLY : F_IP;
                F_EXEC: begin
                    if (beat && (rx_last || len_last)) frame_end <= 1'b1;
                    if (finish) begin
                        next_seq <= next_seq + 1'b1;
                        n_exec   <= n_exec + 1'b1;
                        fs       <= rx_more ? F_SKIP : F_REPLY;
                    end
                end
                F_REPLY: if (tx_fire) begin
                    oi <= oi + 1'b1;
                    if (tx_last) begin
                        oi   <= '0;
                        done <= 1'b1;
                        fs   <= F_IP;
                    end
                end
                default: fs <= F_IP;
            endcase
        end
    end

    // ── Reply ───────────────────────────────────────────────────────────────
    (* ram_style = "block" *) logic [31:0] rmem [REPLY_WORDS];
    logic [31:0] rmem_q;

    // Read one word ahead, so the word after a handshake is ready in the next cycle.
    wire [OW-1:0] oi_next = tx_fire ? oi + 1'b1 : oi;
    wire [OW-1:0] ra      = oi_next < OW'(6) ? '0 : oi_next - OW'(6);

    always_ff @(posedge clk) begin
        if (push) rmem[AW'(n_rd)] <= push_word;
        rmem_q <= rmem[AW'(ra)];
    end

    wire [31:0] status      = {12'd0, denied, overflow, bad_cmd, cut, n_cmds};

    assign tx_valid = fs == F_REPLY;
    assign tx_last  = oi == OW'(5) + OW'(n_rd);
    always_comb begin
        case (oi)
            OW'(0):  tx_data = peer_ip;
            OW'(1):  tx_data = {16'd0, peer_port};
            OW'(2):  tx_data = seq;
            OW'(3):  tx_data = {28'd0, rflags};
            OW'(4):  tx_data = next_seq;
            OW'(5):  tx_data = status;
            default: tx_data = rmem_q;
        endcase
    end
endmodule
