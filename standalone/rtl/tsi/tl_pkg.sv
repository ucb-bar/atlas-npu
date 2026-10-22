// The TileLink-UL subset on the MMIO bus: 32-bit address, one transaction in flight, so no
// size/source/sink/param fields. Opcode values match the TileLink specification.
//
// a_t/d_t carry one 32-bit word: UART TSI and the register devices use them.
// wa_t/wd_t carry one 32-byte beat: the device fabric uses them. Word w of a beat is
// data[32w +: 32], at byte address {address[31:5], w, 2'b00}; mask has one bit per byte.
package tl_pkg;
    typedef enum logic [2:0] {
        A_PUT_FULL    = 3'd0,
        A_PUT_PARTIAL = 3'd1,
        A_GET         = 3'd4
    } a_op_e;

    typedef enum logic [2:0] {
        D_ACK      = 3'd0,
        D_ACK_DATA = 3'd1
    } d_op_e;

    typedef struct packed {
        a_op_e       opcode;
        logic [31:0] address;
        logic [3:0]  mask;
        logic [31:0] data;
    } a_t;

    typedef struct packed {
        d_op_e       opcode;
        logic        denied;
        logic [31:0] data;
    } d_t;

    typedef struct packed {
        a_op_e        opcode;
        logic [31:0]  address;
        logic [31:0]  mask;
        logic [255:0] data;
    } wa_t;

    typedef struct packed {
        d_op_e        opcode;
        logic         denied;
        logic [255:0] data;
    } wd_t;

    function automatic logic [31:0] apply_mask(logic [31:0] old, logic [31:0] data, logic [3:0] mask);
        for (int b = 0; b < 4; b++)
            if (mask[b]) old[8*b +: 8] = data[8*b +: 8];
        return old;
    endfunction

    // Words of a beat whose byte mask is not empty
    function automatic logic [7:0] words_of(logic [31:0] mask);
        for (int w = 0; w < 8; w++) words_of[w] = mask[4*w +: 4] != 4'h0;
    endfunction

    // Byte mask of n words (0..8) starting at word w
    function automatic logic [31:0] word_span(logic [2:0] w, logic [3:0] n);
        logic [63:0] bytes;
        bytes = ((64'd1 << (4 * n)) - 1) << (4 * w);
        return bytes[31:0];
    endfunction
endpackage
