// obf - fast opaque - break valid and data

`timescale 1ns/1ps
module oehb #(
  parameter DATA_TYPE = 32
) (
  input  clk,
  input  rst,
  // Input channel
  input  [DATA_TYPE - 1 : 0] ins,
  input  ins_valid,
  output ins_ready,
  // Output channel
  output [DATA_TYPE - 1 : 0] outs,
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

      full <= ins_valid | (full & ~outs_ready);
    end
  end

  assign outs_valid = full;
  assign ins_ready = ~full | outs_ready;
  assign en = (~full | outs_ready) & ins_valid;
  assign outs = reg_outs;

endmodule