// tbs - slow transparent - break ready

`timescale 1ns/1ps
module tehb_dataless (
	input  clk,
	input  rst,
  // Input Channel
	input  ins_valid,
  output ins_ready,
  // Output Channel
  output outs_valid,	
	input  outs_ready
);
	reg full = 1'b0;

  always @(posedge clk) begin
    if (rst) begin
      full <= 1'b0;
    end
    else begin
      full <= ~outs_ready & (full | ins_valid);
    end
  end

  assign outs_valid = full | ins_valid;
  assign ins_ready = ~full;

endmodule