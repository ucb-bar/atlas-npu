// atlas_ddr4_dram against a randomized AXI4 memory model (user clock domain) and a reference
// memory, under random host and DMA traffic in the core clock domain. Build: standalone/sim/tb_ddr4.sh
module tb_ddr4_dram;
    import tl_pkg::*;
    localparam int SW = 6, AW = 31, LINES = 64, N = 20000;

    logic clk = 0, ui_clk = 0, rst = 1, ui_rst = 1, calib = 0;
    always #10 clk = ~clk;
    always #2.6 ui_clk = ~ui_clk;

    logic a_valid = 0, a_ready, d_valid, d_ready = 1;
    wa_t  a;
    wd_t  d;
    logic dma_a_valid = 0, dma_a_ready, dma_d_valid, dma_d_ready = 1;
    logic [2:0] dma_a_opcode, dma_d_opcode;
    logic [SW-1:0] dma_a_source, dma_d_source;
    logic [63:0] dma_a_address;
    logic [31:0] dma_a_mask;
    logic [255:0] dma_a_data, dma_d_data;
    logic calibrated, aresetn;
    logic [3:0] awid, arid, bid = 0, rid = 0;
    logic [AW-1:0] awaddr, araddr;
    logic [7:0] awlen, arlen;
    logic [2:0] awsize, arsize, awprot, arprot;
    logic [1:0] awburst, arburst, bresp = 0, rresp = 0;
    logic awlock, arlock, wlast, rlast = 1;
    logic [3:0] awcache, arcache, awqos, arqos;
    logic awvalid, awready = 0, wvalid, wready = 0, bvalid = 0, bready;
    logic arvalid, arready = 0, rvalid = 0, rready;
    logic [255:0] wdata, rdata;
    logic [31:0] wstrb;

    atlas_ddr4_dram #(.SW(SW), .AXI_AW(AW)) dut (.*, .calib_done(calib));

    // ── AXI memory model ────────────────────────────────────────────────────
    logic [255:0] mem [LINES];
    int unsigned rq_line [$], rq_due [$], wq_due [$], aw_lines [$];
    logic [255:0] w_data [$];
    logic [31:0]  w_strb [$];
    int unsigned  ucyc = 0;
    always @(posedge ui_clk) begin
        ucyc <= ucyc + 1;
        arready <= ($urandom % 4) != 0;
        awready <= ($urandom % 3) != 0;
        wready  <= ($urandom % 3) != 0;
        if (arvalid && arready) begin
            rq_line.push_back((araddr >> 5) % LINES);
            rq_due.push_back(ucyc + 5 + $urandom % 40);
        end
        if (awvalid && awready) aw_lines.push_back((awaddr >> 5) % LINES);
        if (wvalid && wready) begin
            w_data.push_back(wdata);
            w_strb.push_back(wstrb);
        end
        if (aw_lines.size() > 0 && w_data.size() > 0) begin
            int unsigned l = aw_lines.pop_front();
            logic [255:0] dd = w_data.pop_front();
            logic [31:0]  ss = w_strb.pop_front();
            for (int b = 0; b < 32; b++) if (ss[b]) mem[l][8*b +: 8] = dd[8*b +: 8];
            wq_due.push_back(ucyc + 3 + $urandom % 30);
        end
        if (rvalid && rready) rvalid <= 0;
        else if (!rvalid && rq_due.size() > 0 && rq_due[0] <= ucyc) begin
            void'(rq_due.pop_front());
            rdata  <= mem[rq_line.pop_front()];
            rvalid <= 1;
        end
        if (bvalid && bready) bvalid <= 0;
        else if (!bvalid && wq_due.size() > 0 && wq_due[0] <= ucyc) begin
            void'(wq_due.pop_front());
            bvalid <= 1;
        end
        if (!aresetn) begin
            rvalid <= 0;
            bvalid <= 0;
        end
    end

    // ── Traffic and reference ───────────────────────────────────────────────
    logic [255:0] ref_mem [LINES];
    logic [255:0] exp_h [$], exp_d [$];
    logic [SW-1:0] exp_src [$];
    logic exp_hw [$], exp_dw [$];
    int sent = 0, got = 0, errors = 0, host_sent = 0;

    function automatic logic [255:0] rand256();
        return {$urandom, $urandom, $urandom, $urandom, $urandom, $urandom, $urandom, $urandom};
    endfunction

    function automatic void commit(int line, logic write, logic [31:0] m, logic [255:0] dat);
        for (int b = 0; b < 32; b++) if (write && m[b]) ref_mem[line][8*b +: 8] = dat[8*b +: 8];
    endfunction

    always @(posedge clk) begin
        if (!rst) begin
            d_ready     <= ($urandom % 5) != 0;
            dma_d_ready <= ($urandom % 4) != 0;
            if (dma_a_valid && dma_a_ready) begin
                int l = (dma_a_address >> 5) % LINES;
                logic w = dma_a_opcode != 3'd4;
                commit(l, w, dma_a_mask, dma_a_data);
                exp_d.push_back(w ? 256'd0 : ref_mem[l]);
                exp_dw.push_back(w);
                exp_src.push_back(dma_a_source);
                dma_a_valid <= 0;
                sent++;
            end
            if (a_valid && a_ready) begin
                int l = (a.address >> 5) % LINES;
                logic w = a.opcode != A_GET;
                commit(l, w, a.mask, a.data);
                exp_h.push_back(w ? 256'd0 : ref_mem[l]);
                exp_hw.push_back(w);
                a_valid <= 0;
                host_sent++;
            end
            if ((!dma_a_valid || dma_a_ready) && sent + host_sent < N && ($urandom % 3) != 0) begin
                dma_a_valid   <= 1;
                dma_a_opcode  <= ($urandom % 2) ? 3'd4 : 3'd0;
                dma_a_source  <= SW'($urandom);
                dma_a_address <= {$urandom, $urandom} & ~64'h1F;
                dma_a_mask    <= ($urandom % 2) ? 32'hFFFF_FFFF : $urandom;
                dma_a_data    <= rand256();
            end
            if ((!a_valid || a_ready) && ($urandom % 40) == 0 && sent + host_sent < N) begin
                a_valid   <= 1;
                a.opcode  <= ($urandom % 2) ? A_GET : A_PUT_PARTIAL;
                a.address <= 32'h8000_0000 | ($urandom & 32'h7FFF_FFE0);
                a.mask    <= $urandom;
                a.data    <= rand256();
            end
            if (dma_d_valid && dma_d_ready) begin
                logic [255:0] e = exp_d.pop_front();
                logic w = exp_dw.pop_front();
                logic [SW-1:0] s = exp_src.pop_front();
                if (dma_d_source != s || dma_d_opcode != (w ? 3'd0 : 3'd1) || (!w && dma_d_data != e)) begin
                    if (errors < 5) $display("DMA mismatch: src %0d/%0d op %0d data %h want %h", dma_d_source, s, dma_d_opcode, dma_d_data[63:0], e[63:0]);
                    errors++;
                end
                got++;
            end
            if (d_valid && d_ready) begin
                logic [255:0] e = exp_h.pop_front();
                logic w = exp_hw.pop_front();
                if (d.opcode != (w ? D_ACK : D_ACK_DATA) || (!w && d.data != e)) begin
                    if (errors < 5) $display("host mismatch");
                    errors++;
                end
                got++;
            end
        end
    end

    initial begin
        for (int i = 0; i < LINES; i++) begin
            mem[i] = '0;
            ref_mem[i] = '0;
        end
        #200 ui_rst = 0;
        #300 rst = 0;
        #2000 calib = 1;
        wait (got == N || $time > 64'd200_000_000);
        #100;
        $display("RESULT sent %0d host %0d answered %0d errors %0d time %0t", sent, host_sent, got, errors, $time);
        if (got != N || errors != 0) $fatal(1, "FAIL");
        $display("PASS");
        $finish;
    end
endmodule
