// tbn - N-slot tbf - break none

`timescale 1ns/1ps
module tbf #(
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

      full <= (~outs_ready & ins_valid) | (full & ins_valid) | (full & ~outs_ready);
    end
  end

  assign outs_valid = full | ins_valid;
  assign ins_ready = ~full | outs_ready;
  assign en = ins_valid & (~full | outs_ready) & (full | ~outs_ready);
  assign outs = (~full ? ins : reg_outs);

endmodule


module tfifo #(
  parameter NUM_SLOTS = 2,
  parameter DATA_TYPE = 32
)(
  input  clk,
  input  rst,
  input  [DATA_TYPE - 1 : 0] ins,
  input  ins_valid,
  input  outs_ready,
  output [DATA_TYPE - 1 : 0] outs,
  output outs_valid,
  output ins_ready
);
  wire [NUM_SLOTS : 0] valid;
  wire [NUM_SLOTS : 0] ready;
  wire [(NUM_SLOTS + 1) * DATA_TYPE - 1 : 0] data;

  genvar gen;

  generate
    for (gen = NUM_SLOTS - 1; gen >= 0; gen = gen - 1) begin: slots
        tbf #(
            .DATA_TYPE(DATA_TYPE)
        ) slot (
            .clk(clk),
            .rst(rst),
            .ins(data[(gen + 2) * DATA_TYPE - 1 : (gen + 1) * DATA_TYPE]),
            .ins_valid(valid[gen + 1]),
            .ins_ready(ready[gen + 1]),
            .outs(data[(gen + 1) * DATA_TYPE - 1 : gen * DATA_TYPE]),
            .outs_valid(valid[gen]),
            .outs_ready(ready[gen])
        );
    end
  endgenerate

  assign valid[NUM_SLOTS] = ins_valid;
  assign ins_ready = ready[NUM_SLOTS];
  assign data[(NUM_SLOTS + 1) * DATA_TYPE - 1 : NUM_SLOTS * DATA_TYPE] = ins;

  assign outs_valid = valid[0];
  assign ready[0] = outs_ready;
  assign outs = data[DATA_TYPE - 1 : 0];


endmodule