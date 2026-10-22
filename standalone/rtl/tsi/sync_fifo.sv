// Single-clock FIFO with valid/ready on both sides.
module sync_fifo #(
    parameter int W     = 8,
    parameter int DEPTH = 32
) (
    input  logic         clk,
    input  logic         rst,
    input  logic [W-1:0] wdata,
    input  logic         wvalid,
    output logic         wready,
    output logic [W-1:0] rdata,
    output logic         rvalid,
    input  logic         rready
);
    localparam int AW = $clog2(DEPTH);

    logic [W-1:0]  mem [DEPTH];
    logic [AW-1:0] wptr, rptr;
    logic [AW:0]   count;

    assign wready = count != (AW + 1)'(DEPTH);
    assign rvalid = count != 0;
    assign rdata  = mem[rptr];

    wire push = wvalid && wready;
    wire pop  = rvalid && rready;

    always_ff @(posedge clk) begin
        if (rst) begin
            wptr  <= '0;
            rptr  <= '0;
            count <= '0;
        end else begin
            if (push) begin
                mem[wptr] <= wdata;
                wptr      <= wptr + 1'b1;
            end
            if (pop) rptr <= rptr + 1'b1;
            count <= count + (AW + 1)'(push) - (AW + 1)'(pop);
        end
    end
endmodule
