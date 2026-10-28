// obf - fast opaque - break valid

`timescale 1ns/1ps
module oehb_dataless (
  input  clk,
  input  rst,
  // Input channel
  input  ins_valid,
  output ins_ready,
  // Output channel
  output outs_valid,
  input  outs_ready
);
  wire en;
  reg full = 1'b0;
  
  always @(posedge clk) begin
    if (rst) begin
      full <= 1'b0;
    end
    else begin
      full <= ins_valid | (full & ~outs_ready);
    end
  end

  assign outs_valid = full;
  assign ins_ready = ~full | outs_ready;
  assign en = (~full | outs_ready) & ins_valid;
  
endmodule