// Fused accumulate of fp4_fedp for one output, after the block sums:
//   d = C + sum over blocks g = 0..3 of sigma_g x sA_g x sB_g
// with the arithmetic of NVIDIA tensor cores (see fp4_fedp.sv):
//   - Block g contributes sigma_g x s_a x s_b at exponent e_a + e_b, where (s, e) is the scale
//     normalized to 1 <= s < 2. UE4M3 exponents below -6 clamp to -6, and a zero scale gets -126.
//   - Every term, C included, is cut toward zero at 2^(emax - 35), where emax is the largest
//     term exponent. The sum is then cut toward zero to FP32 once.
//   - A NaN scale or a NaN C gives 0x7FFF_FFFF. An infinite C passes through. If all 64 products
//     are zero (input zero), d = C + (+0): C passes through and -0 becomes +0.
//
// sigma (exact sums of 16 products, units of 1/4) arrives one cycle after in_vld, from a register
// of the block-sum datapath. The other inputs arrive with in_vld. d is valid 5 cycles after in_vld.
module fp4_fedp_acc
    import dotprod_pkg::*;
(
    input  logic                            clk,
    input  logic                            rst,
    input  logic                            in_vld,
    input  logic [1:0]                      mode,
    input  logic [31:0]                     sfa,
    input  logic [31:0]                     sfb,
    input  logic [31:0]                     c,
    input  logic                            zero,
    input  logic signed [NVFP4_INNER_W-1:0] sigma [4],
    output logic                            out_vld,
    output logic [31:0]                     d
);
    localparam int TW = 21;  // block term sigma x Sa x Sb: |T| <= 2304 x 225 < 2^19
    localparam int BW = 48;  // aligned block term magnitude: |T| << 27 < 2^47
    localparam int CW = 37;  // aligned C magnitude: 24 bits << 12
    localparam int SW = 51;  // sum of 4 block terms and C
    localparam int EW = 11;  // signed exponents: scale sums reach -252 .. 254

    // Scale normalized to s x 2^e with s = S / 8. Returns S (4 bits), e, NaN.
    function automatic logic [3+9+1:0] dec_scale(input logic [7:0] x, input logic [1:0] m);
        logic [3:0]        s;
        logic signed [8:0] e;
        logic              nan;
        s = 4'd8; e = '0; nan = 1'b0;
        case (m)
            2'd0: begin  // UE4M3: bit 7 ignored, only 0x7F is NaN
                if (x[6:0] == 7'h7F)        nan = 1'b1;
                else if (x[6:3] != 4'd0) begin s = {1'b1, x[2:0]}; e = 9'(x[6:3]) - 9'sd7; end
                else if (x[2:0] != 3'd0) begin s = {1'b0, x[2:0]}; e = -9'sd6; end
                else                     begin s = 4'd0;           e = -9'sd126; end
            end
            2'd2: ;      // FP4: scale 1.0
            default: begin  // E8M0
                if (x == 8'hFF)      nan = 1'b1;
                else if (x == 8'h00) begin s = 4'd4; e = -9'sd126; end  // 2^-127 = 0.5 x 2^-126
                else                 begin s = 4'd8; e = 9'(x) - 9'sd127; end
            end
        endcase
        return {s, e, nan};
    endfunction

    // ---------------------------------------------------------------- stage 1: scales, specials
    logic [3:0]        sa_s [4], sb_s [4];
    logic signed [8:0] sa_e [4], sb_e [4];
    logic [3:0]        sa_nan, sb_nan;

    for (genvar g = 0; g < 4; g++) begin : g_blk
        wire [1:0] byte_sel = mode == 2'd1 ? 2'(g / 2) : 2'(g);
        assign {sa_s[g], sa_e[g], sa_nan[g]} = dec_scale(sfa[8*byte_sel +: 8], mode);
        assign {sb_s[g], sb_e[g], sb_nan[g]} = dec_scale(sfb[8*byte_sel +: 8], mode);
    end

    wire [7:0] c_exp   = c[30:23];
    wire       c_nan   = c_exp == 8'hFF && c[22:0] != '0;
    wire       c_inf   = c_exp == 8'hFF && c[22:0] == '0;
    wire       sf_nan  = mode == 2'd2 ? 1'b0 :
                         mode == 2'd1 ? |{sa_nan[2], sa_nan[0], sb_nan[2], sb_nan[0]} : |{sa_nan, sb_nan};

    logic                 s1_vld, s1_nan, s1_inf, s1_zero;
    logic [31:0]          s1_c;
    logic [7:0]           s1_p [4];
    logic signed [EW-1:0] s1_x [4];

    always_ff @(posedge clk) begin
        s1_vld  <= !rst && in_vld;
        s1_nan  <= sf_nan || c_nan;
        s1_inf  <= c_inf;
        s1_zero <= zero;
        s1_c    <= c;
        for (int g = 0; g < 4; g++) begin
            s1_p[g] <= sa_s[g] * sb_s[g];
            s1_x[g] <= EW'(sa_e[g]) + EW'(sb_e[g]);
        end
    end

    // ---------------------------------------------------------------- stage 2: exact block terms
    // Block g: T_g x 2^(X_g - 8). C: Sc x 2^(Xc - 23), with Xc = -126 for zero and subnormal C.
    wire [23:0]          s1_cs = {s1_c[30:23] != 8'd0, s1_c[22:0]};
    wire signed [EW-1:0] s1_cx = s1_c[30:23] == 8'd0 ? -EW'(126) : EW'(s1_c[30:23]) - EW'(127);

    logic signed [EW-1:0] emax;
    always_comb begin
        emax = s1_cx;
        for (int g = 0; g < 4; g++)
            if (s1_x[g] > emax) emax = s1_x[g];
    end

    logic                 s2_vld, s2_nan, s2_inf, s2_zero, s2_csign;
    logic [31:0]          s2_c;
    logic signed [TW-1:0] s2_t [4];
    logic [EW:0]          s2_dx [4];  // emax - X_g >= 0
    logic [EW:0]          s2_dc;
    logic [23:0]          s2_cs;
    logic signed [EW-1:0] s2_emax;

    always_ff @(posedge clk) begin
        s2_vld   <= !rst && s1_vld;
        s2_nan   <= s1_nan;
        s2_inf   <= s1_inf;
        s2_zero  <= s1_zero;
        s2_c     <= s1_c;
        s2_csign <= s1_c[31];
        s2_cs    <= s1_cs;
        s2_dc    <= (EW + 1)'(emax - s1_cx);
        s2_emax  <= emax;
        for (int g = 0; g < 4; g++) begin
            s2_t[g]  <= TW'(sigma[g]) * $signed({1'b0, s1_p[g]});
            s2_dx[g] <= (EW + 1)'(emax - s1_x[g]);
        end
    end

    // ---------------------------------------------------------------- stage 3: align, cut toward zero
    // Units of the aligned terms: 2^(emax - 35). Shifting the magnitude truncates toward zero.
    logic signed [SW-1:0] q [5];
    always_comb begin
        for (int g = 0; g < 4; g++) begin
            logic [TW-1:0] abs_t;
            logic [BW-1:0] mag;
            abs_t = s2_t[g] < 0 ? -s2_t[g] : s2_t[g];
            mag   = BW'(abs_t) << 27;
            mag   = s2_dx[g] >= (EW + 1)'(BW) ? '0 : mag >> s2_dx[g];
            q[g]  = s2_t[g] < 0 ? -SW'(mag) : SW'(mag);
        end
        begin
            logic [CW-1:0] mag;
            mag  = CW'(s2_cs) << 12;
            mag  = s2_dc >= (EW + 1)'(CW) ? '0 : mag >> s2_dc;
            q[4] = s2_csign ? -SW'(mag) : SW'(mag);
        end
    end

    logic                 s3_vld, s3_nan, s3_inf, s3_zero;
    logic [31:0]          s3_c;
    logic signed [SW-1:0] s3_q [5];
    logic signed [EW-1:0] s3_emax;

    always_ff @(posedge clk) begin
        s3_vld  <= !rst && s2_vld;
        s3_nan  <= s2_nan;
        s3_inf  <= s2_inf;
        s3_zero <= s2_zero;
        s3_c    <= s2_c;
        s3_emax <= s2_emax;
        for (int i = 0; i < 5; i++) s3_q[i] <= q[i];
    end

    // ---------------------------------------------------------------- stage 4: sum
    logic                 s4_vld, s4_nan, s4_inf, s4_zero;
    logic [31:0]          s4_c;
    logic signed [SW-1:0] s4_sum;
    logic signed [EW-1:0] s4_emax;

    always_ff @(posedge clk) begin
        s4_vld  <= !rst && s3_vld;
        s4_nan  <= s3_nan;
        s4_inf  <= s3_inf;
        s4_zero <= s3_zero;
        s4_c    <= s3_c;
        s4_emax <= s3_emax;
        s4_sum  <= s3_q[0] + s3_q[1] + s3_q[2] + s3_q[3] + s3_q[4];
    end

    // ---------------------------------------------------------------- stage 5: cut to FP32, specials
    // The sum is M x 2^(emax - 35). Its leading one at bit p has weight 2^(p + emax - 35).
    wire [SW-2:0] mag = s4_sum < 0 ? (SW - 1)'(-s4_sum) : (SW - 1)'(s4_sum);

    logic [31:0] packed_res;
    always_comb begin
        int                   p, sh;
        logic signed [EW+1:0] e;
        logic [SW-2:0]        frac;
        p    = -1;
        sh   = int'(s4_emax) + 114;  // subnormal mantissa field = M x 2^(emax + 114)
        frac = '0;
        for (int i = 0; i < SW - 1; i++) if (mag[i]) p = i;
        e = (EW + 2)'(p) + (EW + 2)'(s4_emax) - (EW + 2)'(35);
        if (p < 0) begin
            packed_res = 32'h0000_0000;
        end else if (e > 127) begin
            packed_res = {s4_sum < 0, 8'hFF, 23'd0};
        end else if (e >= -126) begin
            frac = p >= 23 ? mag >> (p - 23) : mag << (23 - p);
            packed_res = {s4_sum < 0, 8'(e + (EW + 2)'(127)), frac[22:0]};
        end else begin  // subnormal, cut toward zero
            frac = sh >= 0 ? mag << sh : (-sh >= SW - 1 ? '0 : mag >> -sh);
            packed_res = frac[22:0] == '0 ? 32'h0000_0000 : {s4_sum < 0, 8'h00, frac[22:0]};
        end
    end

    always_ff @(posedge clk) begin
        out_vld <= !rst && s4_vld;
        if      (s4_nan)  d <= 32'h7FFF_FFFF;
        else if (s4_inf)  d <= s4_c;
        else if (s4_zero) d <= s4_c == 32'h8000_0000 ? 32'h0000_0000 : s4_c;
        else              d <= packed_res;
    end
endmodule
