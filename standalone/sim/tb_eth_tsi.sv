// Unit tests of eth_tsi and tl_arbiter on a behavioral beat-fabric memory (random a_ready and
// response latency; denied outside the SoC's address map). Build and run: standalone/sim/tb_eth_tsi.sh
//
//   tb_eth_tsi     eth_tsi is host 1 of tl_arbiter, a random host 0 shares the fabric. A reference
//                  model of the protocol predicts every reply, the counters and the final memory.
//                  Directed cases, then random datagrams (+n=<count>), then cycles per 32-byte line
//                  for long DRAM writes and reads.
//   tb_tl_arbiter  tl_arbiter between two random hosts.
package tb_eth_pkg;
    import tl_pkg::*;

    // Contents of a line never written
    function automatic logic [255:0] init_line(logic [26:0] line);
        logic [255:0] v;
        for (int w = 0; w < 8; w++)
            v[32 * w +: 32] = ({5'd0, line} * 32'h9E37_79B1) ^ (32'(w) * 32'h85EB_CA6B) ^ 32'h5EED_0000;
        return v;
    endfunction

    // The SoC's address map: 1 word devices (sys_regs, IMEM, CSR), 2 VMEM and DRAM, 0 unmapped
    function automatic int region(logic [31:0] a);
        if (a < 32'h0000_1000) return 1;
        if (a >= 32'h0002_0000 && a < 32'h0004_1000) return 1;
        if (a >= 32'h2000_0000 && a < 32'h2018_0000) return 2;
        if (a >= 32'h8000_0000) return 2;
        return 0;
    endfunction

    function automatic logic [255:0] rand256();
        return {$urandom, $urandom, $urandom, $urandom, $urandom, $urandom, $urandom, $urandom};
    endfunction
endpackage

// One transaction at a time, like tl_xbar and its devices. Writes apply at the a handshake.
module tb_beat_mem
    import tl_pkg::*;
    import tb_eth_pkg::*;
(
    input  logic clk,
    input  logic rst,
    input  int   p_ready,           // percent of idle cycles with a_ready
    input  int   lat_min, lat_max,  // cycles from the a handshake to d_valid (>= 1)
    input  int   p_long,            // percent of responses that take lat_long cycles instead
    input  int   lat_long,
    input  logic a_valid,
    output logic a_ready,
    input  wa_t  a,
    output logic d_valid,
    input  logic d_ready,
    output wd_t  d
);
    logic [255:0] mem [logic [26:0]];
    logic busy;
    int   wait_left;

    always @(posedge clk) begin
        logic [255:0] v;
        logic [26:0]  l;
        int           r;
        if (rst) begin
            a_ready <= 1'b0;
            d_valid <= 1'b0;
            busy    <= 1'b0;
        end else begin
            if (a_valid && a_ready) begin
                l = a.address[31:5];
                r = region(a.address);
                // Insert, then read: Verilator 5.020 may read mem[l] before testing exists(l)
                // (even in an if/else), and that read adds the key with 0.
                if (!mem.exists(l)) mem[l] = init_line(l);
                v = mem[l];
                busy      <= 1'b1;
                a_ready   <= 1'b0;
                wait_left <= (($urandom % 100) < p_long ? lat_long
                              : lat_min + int'($urandom % (lat_max - lat_min + 1))) - 1;
                d.opcode  <= a.opcode == A_GET ? D_ACK_DATA : D_ACK;
                d.denied  <= r == 0;
                d.data    <= a.opcode == A_GET && r != 0 ? v : '0;
                if (a.opcode != A_GET && r != 0) begin
                    for (int b = 0; b < 32; b++) if (a.mask[b]) v[8 * b +: 8] = a.data[8 * b +: 8];
                    mem[l] = v;
                end
            end else if (!busy || (d_valid && d_ready)) begin
                a_ready <= ($urandom % 100) < p_ready;
            end
            if (busy && !d_valid) begin
                if (wait_left <= 0) d_valid <= 1'b1;
                else                wait_left <= wait_left - 1;
            end
            if (d_valid && d_ready) begin
                d_valid <= 1'b0;
                busy    <= 1'b0;
            end
        end
    end
endmodule

// Random single-beat Puts and Gets to LINES private lines at BASE, checked against its own copy.
module tb_rand_host
    import tl_pkg::*;
    import tb_eth_pkg::*;
#(
    parameter logic [31:0] BASE  = 32'hF000_0000,
    parameter int          LINES = 32
) (
    input  logic clk,
    input  logic rst,
    input  int   p_req,     // percent of idle cycles that start a request
    input  int   p_dready,
    output logic a_valid,
    input  logic a_ready,
    output wa_t  a,
    input  logic d_valid,
    output logic d_ready,
    input  wd_t  d,
    output int   n_req, n_resp, n_err, max_wait,
    output logic idle
);
    logic [255:0] ref_mem [LINES];
    logic [255:0] want;
    logic         waiting, get;
    int           cur_wait;

    assign idle = !a_valid && !waiting;

    initial for (int i = 0; i < LINES; i++) ref_mem[i] = init_line(BASE[31:5] + 27'(i));

    always @(posedge clk) begin
        int           i;
        logic [255:0] v;
        logic [31:0]  m;
        if (rst) begin
            a_valid  <= 1'b0;
            d_ready  <= 1'b0;
            waiting  <= 1'b0;
            n_req    <= 0;
            n_resp   <= 0;
            n_err    <= 0;
            max_wait <= 0;
            cur_wait <= 0;
        end else begin
            d_ready <= ($urandom % 100) < p_dready;
            if (a_valid && !a_ready) begin
                cur_wait <= cur_wait + 1;
                if (cur_wait + 1 > max_wait) max_wait <= cur_wait + 1;
            end
            if (a_valid && a_ready) begin
                a_valid  <= 1'b0;
                waiting  <= 1'b1;
                cur_wait <= 0;
                n_req    <= n_req + 1;
            end else if (!a_valid && (!waiting || d_valid && d_ready) && ($urandom % 100) < p_req) begin
                i = $urandom % LINES;
                a_valid <= 1'b1;
                if ($urandom % 2) begin
                    get  <= 1'b1;
                    want <= ref_mem[i];
                    a    <= '{opcode: A_GET, address: BASE + 32'(32 * i), mask: '1, data: '0};
                end else begin
                    m = ($urandom % 2) ? '1 : $urandom;
                    v = rand256();
                    get <= 1'b0;
                    a   <= '{opcode: A_PUT_FULL, address: BASE + 32'(32 * i), mask: m, data: v};
                    for (int b = 0; b < 32; b++) if (m[b]) ref_mem[i][8 * b +: 8] = v[8 * b +: 8];
                end
            end
            if (d_valid && !waiting) begin
                $display("ERROR host %h: response without a request", BASE);
                n_err <= n_err + 1;
            end
            if (d_valid && d_ready && waiting) begin
                if (d.denied || d.opcode != (get ? D_ACK_DATA : D_ACK) || (get && d.data != want)) begin
                    $display("ERROR host %h: bad response op %0d denied %0d data %h want %h",
                             BASE, d.opcode, d.denied, d.data[63:0], want[63:0]);
                    n_err <= n_err + 1;
                end
                waiting <= 1'b0;
                n_resp  <= n_resp + 1;
            end
        end
    end
endmodule

module tb_eth_tsi;
    import tl_pkg::*;
    import tb_eth_pkg::*;
    localparam int RW = 364;

    logic   clk = 0, rst = 1, dut_rst = 1;
    longint cyc = 0;
    always #5 clk = ~clk;
    always @(posedge clk) cyc <= cyc + 1;

    // Knobs
    int p_gap = 0, p_stall = 0;  // percent of cycles without an rx word, without tx_ready
    int m_p_ready = 70, m_lat_min = 1, m_lat_max = 6, m_p_long = 3, m_lat_long = 40;
    int h_p_req = 20;

    // ── DUT, host 0, arbiter, memory ────────────────────────────────────────
    logic [31:0] rx_data = '0, tx_data;
    logic        rx_valid = 0, rx_last = 0, rx_ready, tx_valid, tx_ready = 0, tx_last;
    logic        e_a_valid, e_a_ready, e_d_valid, e_d_ready;
    wa_t         e_a;
    wd_t         e_d;
    logic        synced, done;
    logic [31:0] n_exec, n_dup, n_drop;

    eth_tsi #(.REPLY_WORDS(RW)) dut (
        .clk, .rst(dut_rst),
        .rx_data, .rx_valid, .rx_ready, .rx_last, .tx_data, .tx_valid, .tx_ready, .tx_last,
        .a_valid(e_a_valid), .a_ready(e_a_ready), .a(e_a),
        .d_valid(e_d_valid), .d_ready(e_d_ready), .d(e_d),
        .synced, .n_exec, .n_dup, .n_drop, .done);

    logic [1:0] h_a_valid, h_a_ready, h_d_valid, h_d_ready;
    wa_t        h_a [2];
    wd_t        h_d;
    logic       f_a_valid, f_a_ready, f_d_valid, f_d_ready;
    wa_t        f_a;
    wd_t        f_d;
    int         h_req, h_resp, h_err, h_wait;
    logic       h_idle;

    tb_rand_host #(.BASE(32'hF000_0000)) u_h0 (
        .clk, .rst, .p_req(h_p_req), .p_dready(70),
        .a_valid(h_a_valid[0]), .a_ready(h_a_ready[0]), .a(h_a[0]),
        .d_valid(h_d_valid[0]), .d_ready(h_d_ready[0]), .d(h_d),
        .n_req(h_req), .n_resp(h_resp), .n_err(h_err), .max_wait(h_wait), .idle(h_idle));

    assign h_a_valid[1] = e_a_valid;
    assign h_a[1]       = e_a;
    assign e_a_ready    = h_a_ready[1];
    assign e_d_valid    = h_d_valid[1];
    assign h_d_ready[1] = e_d_ready;
    assign e_d          = h_d;

    tl_arbiter u_arb (
        .clk, .rst,
        .host_a_valid(h_a_valid), .host_a_ready(h_a_ready), .host_a(h_a),
        .host_d_valid(h_d_valid), .host_d_ready(h_d_ready), .host_d(h_d),
        .a_valid(f_a_valid), .a_ready(f_a_ready), .a(f_a),
        .d_valid(f_d_valid), .d_ready(f_d_ready), .d(f_d));

    tb_beat_mem u_mem (
        .clk, .rst, .p_ready(m_p_ready), .lat_min(m_lat_min), .lat_max(m_lat_max),
        .p_long(m_p_long), .lat_long(m_lat_long),
        .a_valid(f_a_valid), .a_ready(f_a_ready), .a(f_a),
        .d_valid(f_d_valid), .d_ready(f_d_ready), .d(f_d));

    int errors = 0;
    function automatic void fail(string s);
        if (errors < 20) $display("ERROR @%0d: %s", cyc, s);
        errors++;
    endfunction

    // ── Requests of eth_tsi: rules and a log ────────────────────────────────
    wa_t         e_a_q;
    bit          e_hold = 0, e_out = 0;
    a_op_e       log_op [$];
    logic [31:0] log_addr [$], log_mask [$];
    longint      log_t [$];

    function automatic void check_req(wa_t r);
        logic [2:0] w;
        w = r.address[4:2];
        if (r.opcode != A_PUT_FULL && r.opcode != A_GET)
            fail($sformatf("opcode %0d", r.opcode));
        if (r.address[31:29] != 3'd0) begin
            if (r.address[4:0] != 5'd0 || (r.opcode == A_GET ? r.mask != '1 : r.mask == '0))
                fail($sformatf("line request %h mask %h", r.address, r.mask));
        end else begin
            if (r.address[1:0] != 2'd0 || r.mask != 32'hF << (4 * w))
                fail($sformatf("word request %h mask %h", r.address, r.mask));
            if (r.opcode != A_GET && r.data != {8{r.data[32 * w +: 32]}})
                fail($sformatf("word request %h: data not replicated", r.address));
        end
    endfunction

    always @(posedge clk) begin
        if (dut_rst) begin
            e_hold <= 0;
            e_out  <= 0;
        end else begin
            if (e_hold && (!e_a_valid || e_a != e_a_q)) fail("a changed while waiting for a_ready");
            if (e_a_valid && e_out) fail("a second request in flight");
            if (e_d_valid && !e_out) fail("response without a request");
            if (e_a_valid && e_a_ready) begin
                check_req(e_a);
                log_op.push_back(e_a.opcode);
                log_addr.push_back(e_a.address);
                log_mask.push_back(e_a.mask);
                log_t.push_back(cyc);
                e_out <= 1;
            end
            if (e_d_valid && e_d_ready) e_out <= 0;
            e_hold <= e_a_valid && !e_a_ready;
            e_a_q  <= e_a;
        end
    end

    function automatic void log_clear();
        log_op.delete();
        log_addr.delete();
        log_mask.delete();
        log_t.delete();
    endfunction

    function automatic void log_is(int i, a_op_e op, logic [31:0] ad, logic [31:0] m);
        if (i >= log_op.size())
            fail($sformatf("log: request %0d missing (want %0d %h %h)", i, op, ad, m));
        else if (log_op[i] != op || log_addr[i] != ad || log_mask[i] != m)
            fail($sformatf("log: request %0d is %0d %h %h, want %0d %h %h",
                           i, log_op[i], log_addr[i], log_mask[i], op, ad, m));
    endfunction

    function automatic void log_size(int n);
        if (log_op.size() != n) fail($sformatf("log: %0d requests, want %0d", log_op.size(), n));
    endfunction

    // ── Reference model ─────────────────────────────────────────────────────
    logic [31:0]  dg [$];  // the datagram being built: prefix, seq, flags, commands
    logic [255:0] ref_mem [logic [26:0]];
    bit           m_synced = 0, m_den, m_reply;
    logic [31:0]  m_expect = 0, m_exec = 0, m_dup = 0, m_drop = 0, m_status;
    logic [3:0]   m_flags;
    int           m_ndata;
    logic [31:0]  expw [$];  // expected reply words
    bit           expl [$];  // ... and their tx_last
    logic [31:0]  expc [$];  // expected counters after each reply: n_exec, n_dup, n_drop, synced
    logic [31:0]  hist [$];  // the datagrams of the expected replies
    int           histn [$];
    int           n_replies = 0, n_sent = 0;
    int           st_kind [6] = '{0, 0, 0, 0, 0, 0};  // runs, SYNC, DUP, DROP, NOSYNC, malformed
    int           st_bit [4] = '{0, 0, 0, 0};         // runs with CUT, BADCMD, OVERFLOW, DENIED
    longint       st_cmds = 0, st_rd = 0;

    function automatic logic [255:0] ref_line(logic [26:0] l);
        if (!ref_mem.exists(l)) ref_mem[l] = init_line(l);
        return ref_mem[l];
    endfunction

    function automatic logic [31:0] m_read(logic [63:0] ad);
        logic [255:0] v;
        if (ad[63:32] != '0) return '0;
        if (region(ad[31:0]) == 0) begin
            m_den = 1;
            return '0;
        end
        v = ref_line(ad[31:5]);
        return v[32 * ad[4:2] +: 32];
    endfunction

    function automatic void m_write(logic [63:0] ad, logic [31:0] w);
        logic [255:0] v;
        if (ad[63:32] != '0) return;
        if (region(ad[31:0]) == 0) begin
            m_den = 1;
            return;
        end
        v = ref_line(ad[31:5]);
        v[32 * ad[4:2] +: 32] = w;
        ref_mem[ad[31:5]] = v;
    endfunction

    function automatic void push_exp(logic [31:0] w, bit l);
        expw.push_back(w);
        expl.push_back(l);
    endfunction

    // Runs dg through the model and queues the expected reply.
    function automatic void model();
        int          n, i;
        logic [31:0] seq, cmd, st, len_port;
        logic [63:0] ad, ln, k;
        logic [31:0] data [$];
        bit          cut, bad, ovf;
        logic [15:0] ncmd;
        data.delete();  // local queues keep their contents between calls in Verilator 5.020
        n = dg.size();
        m_reply = 0;
        len_port = n > 1 ? dg[1] : '0;
        if (n < 4 || len_port[31:16] < 16'd8 || len_port[17:16] != 2'd0) begin
            m_drop++;
            st_kind[5]++;
            return;
        end
        seq = dg[2];
        m_reply = 1;
        m_den = 0;
        cut = 0;
        bad = 0;
        ovf = 0;
        ncmd = 0;
        st = '0;
        if (dg[3][0]) begin
            m_synced = 1;
            m_expect = seq + 1;
            m_flags = 4'h1;
            st_kind[1]++;
        end else if (!m_synced) begin
            m_flags = 4'h4;
            m_drop++;
            st_kind[4]++;
        end else if (seq == m_expect) begin
            m_flags = 4'h0;
            i = 4;
            while (i < n && !cut) begin
                cmd = dg[i];
                if (cmd > 1) begin
                    bad = 1;
                    break;
                end
                if (i + 5 > n) begin
                    cut = 1;
                    break;
                end
                ad = {dg[i + 2], dg[i + 1]};
                ln = {dg[i + 4], dg[i + 3]};
                i += 5;
                k = 0;
                if (cmd == 1) begin
                    while (1) begin
                        if (i >= n) begin
                            cut = 1;
                            break;
                        end
                        m_write(ad, dg[i]);
                        i++;
                        ad += 4;
                        if (k == ln) break;
                        k++;
                    end
                end else begin
                    while (1) begin
                        if (data.size() == RW) begin
                            ovf = 1;
                            break;
                        end
                        data.push_back(m_read(ad));
                        ad += 4;
                        if (k == ln) break;
                        k++;
                    end
                end
                if (!cut) ncmd++;
            end
            st = {12'd0, m_den, ovf, bad, cut, ncmd};
            m_expect++;
            m_exec++;
            st_kind[0]++;
            for (int b = 0; b < 4; b++) st_bit[b] += int'(st[16 + b]);
            st_cmds += ncmd;
            st_rd += data.size();
        end else if ($signed(m_expect - seq) > 0) begin
            m_flags = 4'h2;
            m_dup++;
            st_kind[2]++;
        end else begin
            m_flags = 4'h8;
            m_drop++;
            st_kind[3]++;
        end
        m_status = st;
        m_ndata = data.size();
        push_exp(dg[0], 0);
        push_exp({16'd0, len_port[15:0]}, 0);
        push_exp(seq, 0);
        push_exp({28'd0, m_flags}, 0);
        push_exp(m_expect, 0);
        push_exp(st, data.size() == 0);
        foreach (data[j]) push_exp(data[j], j == data.size() - 1);
        expc.push_back(m_exec);
        expc.push_back(m_dup);
        expc.push_back(m_drop);
        expc.push_back({31'd0, m_synced});
        foreach (dg[j]) hist.push_back(dg[j]);
        histn.push_back(dg.size());
        n_replies++;
    endfunction

    // ── Datagram builder ────────────────────────────────────────────────────
    logic [31:0] rxq [$];
    bit          rxl [$];

    function automatic void dg_head(logic [31:0] seq, logic [31:0] flags);
        dg.delete();
        dg.push_back($urandom);                    // peer IP
        dg.push_back({16'd0, 16'($urandom)});      // peer port
        dg.push_back(seq);
        dg.push_back(flags);
    endfunction

    function automatic void dg_cmd(logic [31:0] cmd, logic [63:0] ad, logic [63:0] len);
        dg.push_back(cmd);
        dg.push_back(ad[31:0]);
        dg.push_back(ad[63:32]);
        dg.push_back(len[31:0]);
        dg.push_back(len[63:32]);
    endfunction

    function automatic void dg_write(logic [63:0] ad, int n);
        dg_cmd(1, ad, 64'(n - 1));
        for (int k = 0; k < n; k++) dg.push_back($urandom);
    endfunction

    function automatic void dg_read(logic [63:0] ad, longint n);
        dg_cmd(0, ad, 64'(n - 1));
    endfunction

    // Sets the length field (plen < 0: the words of dg), runs the model, queues the datagram.
    function automatic void dg_send(int plen = -1);
        logic [31:0] w;
        if (dg.size() > 1) begin
            w = dg[1];
            w[31:16] = plen < 0 ? 16'(4 * (dg.size() - 2)) : 16'(plen);
            dg[1] = w;
        end
        model();
        foreach (dg[k]) begin
            rxq.push_back(dg[k]);
            rxl.push_back(k == dg.size() - 1);
        end
        n_sent++;
    endfunction

    // ── rx driver, tx monitor ───────────────────────────────────────────────
    bit     in_dg = 0;
    longint t_rx0 = 0, t_tx0 = 0;
    int     n_done = 0;

    always @(posedge clk) begin
        if (rx_valid && rx_ready) begin
            if (!in_dg) t_rx0 <= cyc;
            in_dg <= !rx_last;
            void'(rxq.pop_front());
            void'(rxl.pop_front());
        end
        if (!rx_valid || rx_ready) begin
            if (rxq.size() > 0 && ($urandom % 100) >= p_gap) begin
                rx_valid <= 1;
                rx_data  <= rxq[0];
                rx_last  <= rxl[0];
            end else begin
                rx_valid <= 0;
            end
        end
    end

    logic [31:0] got [$];

    function automatic void check_reply();
        logic [31:0] e [$], c [4], h [$];
        bit          l;
        int          bad, nh;
        bad = -1;
        e.delete();
        h.delete();
        if (expw.size() == 0) begin
            fail("reply not expected");
            return;
        end
        do begin
            e.push_back(expw.pop_front());
            l = expl.pop_front();
        end while (!l);
        nh = histn.pop_front();
        for (int j = 0; j < nh; j++) h.push_back(hist.pop_front());
        for (int k = 0; k < e.size() || k < got.size(); k++)
            if (bad < 0 && (k >= e.size() || k >= got.size() || got[k] != e[k])) bad = k;
        if (bad >= 0) begin
            fail($sformatf("reply to seq %h: word %0d is %h, want %h; %0d words, want %0d; status %h, want %h",
                           e[2], bad, bad < got.size() ? got[bad] : 32'hx, bad < e.size() ? e[bad] : 32'hx,
                           got.size(), e.size(), got.size() > 5 ? got[5] : 32'hx, e[5]));
            if (errors <= 3) begin
                $display("  datagram, %0d words:", h.size());
                for (int j = 0; j < h.size(); j += 8)
                    $display("    %3d: %h %h %h %h %h %h %h %h", j, h[j], h[j + 1], h[j + 2], h[j + 3],
                             h[j + 4], h[j + 5], h[j + 6], h[j + 7]);
            end
        end
        for (int j = 0; j < 4; j++) c[j] = expc.pop_front();
        if (n_exec != c[0] || n_dup != c[1] || n_drop != c[2] || synced != c[3][0])
            fail($sformatf("counters exec %0d dup %0d drop %0d synced %0d, want %0d %0d %0d %0d",
                           n_exec, n_dup, n_drop, synced, c[0], c[1], c[2], c[3]));
    endfunction

    always @(posedge clk) begin
        tx_ready <= ($urandom % 100) >= p_stall;
        if (done) n_done++;
        if (tx_valid && tx_ready) begin
            if (got.size() == 0) t_tx0 <= cyc;
            got.push_back(tx_data);
            if (tx_last) begin
                check_reply();
                got.delete();
            end
        end
    end

    // ── Test sequences ──────────────────────────────────────────────────────
    task automatic drain();
        int idle;
        idle = 0;
        while (idle < 30) begin
            @(posedge clk);
            if (rxq.size() == 0 && !rx_valid && expw.size() == 0 && !tx_valid && !e_out && !e_a_valid)
                idle++;
            else
                idle = 0;
        end
    endtask

    task automatic dut_reset();
        drain();
        @(negedge clk);
        dut_rst = 1;
        repeat (3) @(negedge clk);
        dut_rst = 0;
        m_synced = 0;
        m_expect = 0;
        m_exec = 0;
        m_dup = 0;
        m_drop = 0;
    endtask

    // Checks the model's verdict on the last datagram (guards the model, the monitor checks the DUT).
    function automatic void want(string what, logic [3:0] flags, logic [31:0] st, int ndata = -1);
        if (!m_reply || m_flags != flags || m_status != st || (ndata >= 0 && m_ndata != ndata))
            fail($sformatf("%s: model gives reply %0d flags %h status %h data %0d", what,
                           m_reply, m_flags, m_status, m_ndata));
    endfunction

    localparam logic [31:0] CUT = 32'h1_0000, BAD = 32'h2_0000, OVF = 32'h4_0000, DEN = 32'h8_0000;

    task automatic directed();
        logic [31:0] s;
        // NOSYNC: nothing runs
        log_clear();
        dg_head(5, 0); dg_write(64'h8000_0000, 4); dg_send();
        want("nosync", 4'h4, 0);
        drain(); log_size(0);

        // Malformed: dropped without a reply
        dg.delete(); dg.push_back(1); dg.push_back(2); dg_send(0);       // prefix only
        dg_head(7, 0); void'(dg.pop_back()); dg_send(4);                  // seq only
        dg_head(7, 0); dg_send(6);                                         // not whole words
        dg_head(7, 0); dg.push_back(0); dg_send(10);
        dg.delete(); dg.push_back(1); dg_send(0);                          // one word
        dg_head(7, 0); void'(dg.pop_back()); dg_send(8);                  // length 8, 1 word
        dg_head(7, 0); dg_write(64'h8000_0000, 2); dg_send(4);            // length 4, more words
        if (m_drop != 8) fail("malformed: model");

        // SYNC, with words after it that must not run
        s = 32'h1234_5670;
        log_clear();
        dg_head(s, 1); dg_write(64'h8000_0000, 4); dg_send();
        want("sync", 4'h1, 0);
        drain(); log_size(0);

        // A full aligned line: one Put
        log_clear();
        dg_head(m_expect, 0); dg_write(64'h8000_0100, 8); dg_send();
        want("line", 4'h0, 1);
        drain(); log_size(1); log_is(0, A_PUT_FULL, 32'h8000_0100, '1);

        // Unaligned write over two lines, read back
        log_clear();
        dg_head(m_expect, 0); dg_write(64'h8000_020C, 10); dg_read(64'h8000_0208, 12); dg_send();
        want("unaligned", 4'h0, 2, 12);
        drain(); log_size(4);
        log_is(0, A_PUT_FULL, 32'h8000_0200, 32'hFFFF_F000);
        log_is(1, A_PUT_FULL, 32'h8000_0220, 32'h000F_FFFF);
        log_is(2, A_GET, 32'h8000_0200, '1);
        log_is(3, A_GET, 32'h8000_0220, '1);

        // Read, write, read one line: the second read Gets the line again
        log_clear();
        dg_head(m_expect, 0); dg_read(64'h8000_0900, 2); dg_write(64'h8000_0904, 2);
        dg_read(64'h8000_0900, 3); dg_send();
        want("read after write", 4'h0, 3, 5);
        drain(); log_size(3);
        log_is(2, A_GET, 32'h8000_0900, '1);

        // Partial line in VMEM; two commands on one line are two Puts
        log_clear();
        dg_head(m_expect, 0); dg_write(64'h2000_0344, 3); dg_write(64'h2000_0350, 2);
        dg_read(64'h2000_0340, 8); dg_send();
        want("partial", 4'h0, 3, 8);
        drain(); log_size(3);
        log_is(0, A_PUT_FULL, 32'h2000_0340, 32'h0000_FFF0);
        log_is(1, A_PUT_FULL, 32'h2000_0340, 32'h00FF_0000);
        log_is(2, A_GET, 32'h2000_0340, '1);

        // Word devices: one transaction per word
        log_clear();
        dg_head(m_expect, 0);
        dg_write(64'h0000_0010, 3); dg_write(64'h0002_0004, 2); dg_write(64'h0004_001C, 1);
        dg_read(64'h0000_0010, 3); dg_read(64'h0002_0004, 2); dg_read(64'h0004_001C, 1);
        dg_send();
        want("words", 4'h0, 6, 6);
        drain(); log_size(12);
        log_is(0, A_PUT_FULL, 32'h0000_0010, 32'h000F_0000);
        log_is(1, A_PUT_FULL, 32'h0000_0014, 32'h00F0_0000);
        log_is(2, A_PUT_FULL, 32'h0000_0018, 32'h0F00_0000);
        log_is(3, A_PUT_FULL, 32'h0002_0004, 32'h0000_00F0);
        log_is(5, A_PUT_FULL, 32'h0004_001C, 32'hF000_0000);
        log_is(6, A_GET, 32'h0000_0010, 32'h000F_0000);
        log_is(11, A_GET, 32'h0004_001C, 32'hF000_0000);

        // From the word path into VMEM; the words below 0x2000_0000 are unmapped
        log_clear();
        dg_head(m_expect, 0); dg_write(64'h1FFF_FFF8, 4); dg_read(64'h1FFF_FFF8, 4); dg_send();
        want("boundary", 4'h0, DEN | 2, 4);
        drain(); log_size(6);
        log_is(0, A_PUT_FULL, 32'h1FFF_FFF8, 32'h0F00_0000);
        log_is(1, A_PUT_FULL, 32'h1FFF_FFFC, 32'hF000_0000);
        log_is(2, A_PUT_FULL, 32'h2000_0000, 32'h0000_00FF);
        log_is(5, A_GET, 32'h2000_0000, '1);

        // At or above 4 GiB: writes dropped, reads 0, nothing on the bus
        log_clear();
        dg_head(m_expect, 0); dg_write(64'h1_8000_0400, 4); dg_read(64'h1_8000_0400, 4); dg_send();
        want("4 GiB", 4'h0, 2, 4);
        drain(); log_size(0);
        log_clear();
        dg_head(m_expect, 0); dg_write(64'hFFFF_FFF8, 4); dg_read(64'hFFFF_FFF8, 4); dg_send();
        want("4 GiB crossing", 4'h0, 2, 4);
        drain(); log_size(2);
        log_is(0, A_PUT_FULL, 32'hFFFF_FFE0, 32'hFF00_0000);
        log_is(1, A_GET, 32'hFFFF_FFE0, '1);

        // Denied: the VMEM hole and an unmapped word address
        dg_head(m_expect, 0); dg_read(64'h2018_0000, 4); dg_write(64'h1000_0000, 2); dg_send();
        want("denied", 4'h0, DEN | 2, 4);

        // DUP: an older seq does not run
        drain(); log_clear();
        dg_head(s + 1, 0); dg_write(64'h8000_0100, 8); dg_send();
        want("dup", 4'h2, 0);
        dg_head(s - 100, 0); dg_send();
        want("dup old", 4'h2, 0);
        // DROP: a gap
        dg_head(m_expect + 3, 0); dg_write(64'h8000_0100, 8); dg_send();
        want("drop", 4'h8, 0);
        drain(); log_size(0);

        // CUT: inside the data (the words received are written), inside a header, before the data
        log_clear();
        dg_head(m_expect, 0); dg_cmd(1, 64'h8000_0400, 9);
        for (int k = 0; k < 5; k++) dg.push_back($urandom);
        dg_send();
        want("cut data", 4'h0, CUT);
        drain(); log_size(1); log_is(0, A_PUT_FULL, 32'h8000_0400, 32'h000F_FFFF);
        dg_head(m_expect, 0); dg_write(64'h8000_0500, 2); dg.push_back(1); dg.push_back(32'h8000_0600);
        dg_send();
        want("cut header", 4'h0, CUT | 1);
        dg_head(m_expect, 0); dg_read(64'h8000_0500, 2); dg_cmd(1, 64'h8000_0600, 0); dg_send();
        want("cut no data", 4'h0, CUT | 1, 2);
        dg_head(m_expect, 0); dg_write(64'h0000_0004, 3); void'(dg.pop_back()); dg_send();
        want("cut word path", 4'h0, CUT);

        // BADCMD: the rest is skipped
        dg_head(m_expect, 0); dg_write(64'h8000_0700, 3); dg.push_back(7);
        dg_write(64'h8000_0800, 3); dg_send();
        want("badcmd", 4'h0, BAD | 1);
        dg_head(m_expect, 0); dg.push_back(2); dg_send();
        want("badcmd last", 4'h0, BAD);

        // OVERFLOW
        dg_head(m_expect, 0); dg_read(64'h8000_0000, 400); dg_send();
        want("overflow", 4'h0, OVF | 1, RW);
        dg_head(m_expect, 0); dg_read(64'h8000_0004, 300); dg_read(64'h0000_0000, 100);
        dg_read(64'h8000_0000, 1); dg_send();
        want("overflow 2", 4'h0, OVF | 3, RW);
        dg_head(m_expect, 0); dg_read(64'h8000_0000, 64'h1_0000_0005); dg_send();
        want("overflow huge", 4'h0, OVF | 1, RW);
        dg_head(m_expect, 0); dg_read(64'h8000_0000, RW); dg_send();
        want("exactly full", 4'h0, 1, RW);

        // No commands; then the largest write datagram
        dg_head(m_expect, 0); dg_send();
        want("empty", 4'h0, 0);
        dg_head(m_expect, 0); dg_write(64'h8000_0804, 361); dg_send();
        want("1472 bytes", 4'h0, 1);

        // A new SYNC restarts the session; seq wraps
        dg_head(32'hFFFF_FFFE, 1); dg_send();
        dg_head(32'hFFFF_FFFF, 0); dg_write(64'h8000_0040, 2); dg_send();
        want("wrap 1", 4'h0, 1);
        dg_head(32'h0000_0000, 0); dg_read(64'h8000_0040, 2); dg_send();
        want("wrap 2", 4'h0, 1, 2);
        dg_head(32'hFFFF_FFF0, 0); dg_send();
        want("wrap dup", 4'h2, 0);
        dg_head(32'h0000_0005, 0); dg_send();
        want("wrap drop", 4'h8, 0);
        drain();
    endtask

    // ── Random datagrams ────────────────────────────────────────────────────
    function automatic logic [63:0] rand_addr();
        int          r;
        logic [31:0] hi, lo;
        r = $urandom % 100;
        hi = '0;
        if      (r < 35) lo = 32'h8000_0000 + $urandom % 32'h1000;   // DRAM
        else if (r < 42) lo = 32'hFFFF_FF00 + $urandom % 32'h100;    // DRAM top, into 4 GiB
        else if (r < 52) lo = 32'h2000_0000 + $urandom % 32'h400;    // VMEM
        else if (r < 57) lo = 32'h2017_FF80 + $urandom % 32'h100;    // VMEM end, denied after it
        else if (r < 64) lo = $urandom % 32'h1100;                   // sys_regs, denied after it
        else if (r < 72) lo = 32'h0002_0000 + $urandom % 32'h200;    // IMEM
        else if (r < 77) lo = 32'h0004_0F80 + $urandom % 32'h100;    // CSR end
        else if (r < 82) lo = 32'h1FFF_FF80 + $urandom % 32'h100;    // unmapped, into VMEM
        else if (r < 85) lo = 32'h1000_0000 + $urandom % 32'h100;    // unmapped
        else if (r < 92) begin                                       // >= 4 GiB
            lo = 32'h8000_0000 + $urandom % 32'h1000;
            hi = ($urandom % 4 == 0) ? $urandom | 1 : 1 + $urandom % 3;
        end else lo = 32'h8000_0000 + $urandom % 32'h100;           // DRAM, small
        lo[1:0] = 2'd0;
        return {hi, lo};
    endfunction

    function automatic int rand_len();
        int r;
        r = $urandom % 100;
        if (r < 50) return 1 + $urandom % 12;
        if (r < 85) return 1 + $urandom % 64;
        return 1 + $urandom % 361;
    endfunction

    function automatic void add_cmds();
        int  ncmd, room, n;
        ncmd = ($urandom % 4 == 0) ? 1 : 1 + $urandom % 12;
        for (int c = 0; c < ncmd; c++) begin
            room = 368 - (dg.size() - 2);  // words left in a 1472-byte payload
            if (room < 6) break;
            if ($urandom % 2) begin
                n = rand_len();
                if (n > room - 5) n = room - 5;
                dg_write(rand_addr(), n);
            end else if ($urandom % 50 == 0) begin
                dg_read(rand_addr(), 64'h1_0000_0000 + $urandom % 16);
            end else begin
                dg_read(rand_addr(), rand_len());
            end
        end
        if ($urandom % 100 < 3) begin                     // a bad cmd and junk after it
            dg.push_back(2 + $urandom % 1000);
            repeat ($urandom % 8) dg.push_back($urandom);
        end
        if ($urandom % 100 < 6 && dg.size() > 5) begin     // cut anywhere after flags
            n = 4 + $urandom % (dg.size() - 4);
            while (dg.size() > n) void'(dg.pop_back());
        end
    endfunction

    function automatic void rand_dg();
        int r;
        r = $urandom % 1000;
        if (r < 40) begin                                  // SYNC
            dg_head($urandom, 1 | ($urandom % 4 == 0 ? $urandom & 32'hFFFF_FFFE : 0));
            if ($urandom % 4 == 0) add_cmds();
            dg_send();
        end else if (r < 70) begin                         // malformed
            dg_head(m_expect, 0);
            case ($urandom % 4)
                0: begin void'(dg.pop_back()); dg_send(4); end
                1: dg_send(4 * (dg.size() - 2) - 1 - $urandom % 3);
                2: begin add_cmds(); dg_send(4 * (dg.size() - 2) - 1 - $urandom % 3); end
                default: begin while (dg.size() > 2) void'(dg.pop_back()); dg_send(0); end
            endcase
        end else if (r < 130) begin                        // DUP
            dg_head(m_expect - 1 - $urandom % 20, 0);
            add_cmds();
            dg_send();
        end else if (r < 180) begin                        // DROP
            dg_head(m_expect + 1 + $urandom % 20, 0);
            add_cmds();
            dg_send();
        end else begin
            dg_head(m_expect, 0);
            add_cmds();
            dg_send();
        end
    endfunction

    task automatic random_phase(int n);
        for (int i = 0; i < n; i++) begin
            if (i % 500 == 0) begin
                case ((i / 500) % 4)
                    0: begin p_gap = 0;  p_stall = 0;  m_p_ready = 100; m_lat_min = 1; m_lat_max = 2;  m_p_long = 0; h_p_req = 0;  end
                    1: begin p_gap = 30; p_stall = 50; m_p_ready = 60;  m_lat_min = 1; m_lat_max = 8;  m_p_long = 3; h_p_req = 20; end
                    2: begin p_gap = 5;  p_stall = 10; m_p_ready = 80;  m_lat_min = 1; m_lat_max = 4;  m_p_long = 1; h_p_req = 80; end
                    default: begin p_gap = 60; p_stall = 80; m_p_ready = 30; m_lat_min = 1; m_lat_max = 20; m_p_long = 10; h_p_req = 40; end
                endcase
            end
            if ($urandom % 1000 < 2) dut_reset();
            if (log_op.size() > 10000) log_clear();
            while (rxq.size() > 4000) @(posedge clk);
            rand_dg();
        end
        drain();
    endtask

    // ── Cycles per 32-byte line for a long DRAM write and read ──────────────
    task automatic perf(int lat);
        real w_all, w_req, r_all, r_req;
        m_p_ready = 100; m_lat_min = lat; m_lat_max = lat; m_p_long = 0;
        h_p_req = 0; p_gap = 0; p_stall = 0;
        drain();
        dg_head($urandom, 1); dg_send();
        drain(); log_clear();
        dg_head(m_expect, 0); dg_write(64'h8001_0000, 360); dg_send();
        drain();
        w_all = real'(t_tx0 - t_rx0) / 45.0;
        w_req = real'(log_t[log_t.size() - 1] - log_t[0]) / 44.0;
        log_size(45);
        log_clear();
        dg_head(m_expect, 0); dg_read(64'h8001_0000, 360); dg_send();
        drain();
        r_all = real'(t_tx0 - t_rx0) / 45.0;
        r_req = real'(log_t[log_t.size() - 1] - log_t[0]) / 44.0;
        log_size(45);
        $display("PERF latency %2d: write %5.2f cycles/line (%5.2f between Puts, %5.1f MB/s at 50 MHz), read %5.2f cycles/line (%5.2f between Gets)",
                 lat, w_all, w_req, 1600.0 / w_all, r_all, r_req);
    endtask

    // ── Main ────────────────────────────────────────────────────────────────
    initial begin
        int n, seed;
        int l_lines, l_diff;
        logic [255:0] v;
        n = 3000;
        void'($value$plusargs("n=%d", n));
        if ($value$plusargs("seed=%d", seed)) void'($urandom(seed));
        repeat (5) @(negedge clk);
        rst = 0;
        dut_rst = 0;
        repeat (5) @(negedge clk);

        directed();
        $display("directed: %0d datagrams, %0d replies, %0d errors", n_sent, n_replies, errors);
        p_gap = 0; p_stall = 0;
        for (int i = 0; i < 20; i++) begin                // back to back, no gaps
            dg_head(m_expect, 0);
            add_cmds();
            dg_send();
        end
        drain();
        random_phase(n);
        $display("random: %0d datagrams: %0d run (%0d commands, %0d read words; CUT %0d BADCMD %0d OVERFLOW %0d DENIED %0d), SYNC %0d, DUP %0d, DROP %0d, NOSYNC %0d, malformed %0d; %0d errors",
                 n_sent, st_kind[0], st_cmds, st_rd, st_bit[0], st_bit[1], st_bit[2], st_bit[3],
                 st_kind[1], st_kind[2], st_kind[3], st_kind[4], st_kind[5], errors);
        perf(1);
        perf(8);
        perf(40);
        drain();
        repeat (100) @(posedge clk);

        // Final memory: every line the model or the fabric touched
        l_lines = 0;
        l_diff = 0;
        foreach (ref_mem[l]) begin
            if (!u_mem.mem.exists(l)) u_mem.mem[l] = init_line(l);
            v = u_mem.mem[l];
            l_lines++;
            if (v != ref_mem[l]) begin
                l_diff++;
                if (l_diff < 5) fail($sformatf("memory line %h: %h, want %h", {l, 5'd0}, v, ref_mem[l]));
            end
        end
        foreach (u_mem.mem[l]) begin
            if (l >= 27'(32'hF000_0000 >> 5) && l < 27'(32'hF000_1000 >> 5)) continue;  // host 0
            if (!ref_mem.exists(l) && u_mem.mem[l] != init_line(l)) begin
                l_diff++;
                fail($sformatf("memory line %h written, model did not", {l, 5'd0}));
            end
        end
        if (n_done != n_replies) fail($sformatf("done pulsed %0d times, %0d replies", n_done, n_replies));
        if (expw.size() != 0) fail($sformatf("%0d reply words missing", expw.size()));
        if (h_err != 0 || h_req != h_resp) fail($sformatf("host 0: %0d errors, %0d requests, %0d responses", h_err, h_req, h_resp));
        $display("RESULT datagrams %0d replies %0d memory lines %0d host0 requests %0d errors %0d cycles %0d",
                 n_sent, n_replies, l_lines, h_req, errors, cyc);
        if (errors != 0) $fatal(1, "FAIL");
        $display("PASS");
        $finish;
    end

    // Watchdog: no progress
    longint last_act = 0;
    always @(posedge clk) begin
        if ((rx_valid && rx_ready) || (tx_valid && tx_ready) || (e_a_valid && e_a_ready)) last_act <= cyc;
        if (cyc - last_act > 200000 && (rxq.size() > 0 || expw.size() > 0)) begin
            $display("ERROR @%0d: no progress (rx %0d words queued, %0d reply words expected)", cyc, rxq.size(), expw.size());
            $fatal(1, "FAIL");
        end
    end
endmodule

module tb_tl_arbiter;
    import tl_pkg::*;
    import tb_eth_pkg::*;

    logic   clk = 0, rst = 1;
    longint cyc = 0;
    always #5 clk = ~clk;
    always @(posedge clk) cyc <= cyc + 1;

    int p_req [2] = '{50, 50};
    int p_dready = 70;
    int m_p_ready = 70, m_lat_min = 1, m_lat_max = 6, m_p_long = 3;

    logic [1:0] h_a_valid, h_a_ready, h_d_valid, h_d_ready;
    wa_t        h_a [2];
    wd_t        h_d;
    logic       a_valid, a_ready, d_valid, d_ready;
    wa_t        a;
    wd_t        d;
    int         n_req [2], n_resp [2], n_err [2], max_wait [2];
    logic [1:0] idle;

    for (genvar k = 0; k < 2; k++) begin : g_host
        tb_rand_host #(.BASE(32'hF000_0000 + 32'(k) * 32'h0100_0000)) u_host (
            .clk, .rst, .p_req(p_req[k]), .p_dready,
            .a_valid(h_a_valid[k]), .a_ready(h_a_ready[k]), .a(h_a[k]),
            .d_valid(h_d_valid[k]), .d_ready(h_d_ready[k]), .d(h_d),
            .n_req(n_req[k]), .n_resp(n_resp[k]), .n_err(n_err[k]), .max_wait(max_wait[k]),
            .idle(idle[k]));
    end

    tl_arbiter u_arb (
        .clk, .rst,
        .host_a_valid(h_a_valid), .host_a_ready(h_a_ready), .host_a(h_a),
        .host_d_valid(h_d_valid), .host_d_ready(h_d_ready), .host_d(h_d),
        .a_valid, .a_ready, .a, .d_valid, .d_ready, .d);

    tb_beat_mem u_mem (
        .clk, .rst, .p_ready(m_p_ready), .lat_min(m_lat_min), .lat_max(m_lat_max),
        .p_long(m_p_long), .lat_long(30), .a_valid, .a_ready, .a, .d_valid, .d_ready, .d);

    int errors = 0;
    function automatic void fail(string s);
        if (errors < 20) $display("ERROR @%0d: %s", cyc, s);
        errors++;
    endfunction

    // Fabric rules, routing and round robin
    wa_t    a_q;
    bit     hold = 0, out = 0, owner = 0, last_g = 0, other_waited = 0;
    int     n_grant [2] = '{0, 0};
    longint last_grant_t = 0;
    always @(posedge clk) begin
        bit g;
        if (!rst) begin
            if (hold && (!a_valid || a != a_q)) fail("fabric a changed while waiting for a_ready");
            if (a_valid && out) fail("fabric request while one is in flight");
            if (h_d_valid != (out && d_valid ? 2'b01 << owner : 2'b00))
                fail($sformatf("response to %b, owner %0d", h_d_valid, owner));
            if (d_ready != (out && h_d_ready[owner])) fail("d_ready not from the owner");
            if (a_valid && a_ready) begin
                g = h_a_valid[1] && h_a_ready[1];
                if ((h_a_valid & h_a_ready) != 2'b01 << g) fail("host handshakes");
                if (a != h_a[g]) fail("fabric a is not the granted host's");
                if (n_grant[0] + n_grant[1] > 0 && g == last_g && other_waited)
                    fail($sformatf("host %0d granted twice while host %0d waited", g, !g));
                other_waited = h_a_valid[!g];
                last_g = g;
                owner <= g;
                out <= 1;
                n_grant[g]++;
                last_grant_t <= cyc;
            end else if ((h_a_valid & h_a_ready) != 0) begin
                fail("host handshake without a fabric handshake");
            end
            if (d_valid && d_ready) out <= 0;
            hold <= a_valid && !a_ready;
            a_q  <= a;
            if (h_a_valid != 0 && cyc - last_grant_t > 5000) begin
                fail("no grant for 5000 cycles");
                $fatal(1, "FAIL");
            end
        end
    end

    task automatic phase(string name, int p0, int p1, int pd, int mr, int lmin, int lmax, int pl, int cycles);
        int g0, g1;
        p_req[0] = p0; p_req[1] = p1; p_dready = pd;
        m_p_ready = mr; m_lat_min = lmin; m_lat_max = lmax; m_p_long = pl;
        g0 = n_grant[0];
        g1 = n_grant[1];
        repeat (cycles) @(posedge clk);
        g0 = n_grant[0] - g0;
        g1 = n_grant[1] - g1;
        $display("phase %-12s grants %6d %6d  max wait %0d %0d", name, g0, g1, max_wait[0], max_wait[1]);
        if (p0 == 100 && p1 == 100 && (g0 - g1 > 2 || g1 - g0 > 2))
            fail($sformatf("%s: unfair, %0d vs %0d", name, g0, g1));
    endtask

    initial begin
        int seed;
        if ($value$plusargs("seed=%d", seed)) void'($urandom(seed));
        repeat (5) @(negedge clk);
        rst = 0;
        phase("random",    50,  50,  70,  70, 1, 6,  3, 40000);
        phase("saturated", 100, 100, 70,  70, 1, 6,  3, 40000);
        phase("host 0",    60,  0,   50,  50, 1, 10, 5, 20000);
        phase("host 1",    0,   60,  50,  50, 1, 10, 5, 20000);
        phase("fast",      100, 100, 100, 100, 1, 1, 0, 20000);
        phase("slow d",    80,  30,  15,  40, 1, 20, 10, 40000);
        phase("sparse",    3,   3,   90,  90, 1, 3,  0, 40000);
        p_req[0] = 0;
        p_req[1] = 0;
        while (idle != 2'b11) @(posedge clk);
        repeat (10) @(posedge clk);
        for (int k = 0; k < 2; k++)
            if (n_err[k] != 0 || n_req[k] != n_resp[k] || n_req[k] != n_grant[k])
                fail($sformatf("host %0d: %0d errors, %0d requests, %0d responses, %0d grants",
                               k, n_err[k], n_req[k], n_resp[k], n_grant[k]));
        $display("RESULT requests %0d %0d errors %0d", n_req[0], n_req[1], errors);
        if (errors != 0) $fatal(1, "FAIL");
        $display("PASS");
        $finish;
    end
endmodule
