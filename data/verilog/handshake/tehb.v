// tbs - slow transparent - break ready

`timescale 1ns/1ps
module tehb #(
	parameter DATA_TYPE = 32 // Default set to 32 bits
)(
	input  clk,
	input  rst,
  // Input Channel
	input  [DATA_TYPE - 1 : 0] ins,
	input  ins_valid,
  output ins_ready,
  // Output Channel
  output [DATA_TYPE - 1 : 0]	outs,
  output outs_valid,
	input  outs_ready
);
	wire en;
  reg full = 1'b0;
  reg [DATA_TYPE - 1 : 0] reg_outs = {DATA_TYPE{1'b0}};

  always @(posedge clk) begin
    if (rst) begin
      full <= 1'b0;
      reg_outs <= {DATA_TYPE{1'b0}};
    end
    else begin
      if (en) begin
        reg_outs <= ins;
      end

      full <= ~outs_ready & (full | ins_valid);
    end
  end

  assign outs_valid = full | ins_valid;
  assign ins_ready = ~full;
  assign en = ~outs_ready & ins_valid & ~full;
  assign outs = (~full ? ins : reg_outs);

endmodule