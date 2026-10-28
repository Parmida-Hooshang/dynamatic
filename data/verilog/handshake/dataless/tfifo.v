// tbn - N-slot tbf - break none

`timescale 1ns/1ps
module tbf_dataless (
  input  clk,
  input  rst,
  // Input channel
  input  ins_valid,
  output ins_ready,
  // Output channel
  output outs_valid,
  input  outs_ready
);
  
  reg full = 1'b0;

  always @(posedge clk) begin
    if (rst) begin
      full <= 1'b0;
    end
    else begin
      full <= (~outs_ready & ins_valid) | (full & ins_valid) | (full & ~outs_ready);
    end
  end

  assign outs_valid = full | ins_valid;
  assign ins_ready = ~full | outs_ready;

endmodule


module tfifo_dataless #(
  parameter NUM_SLOTS = 2
)(
  input  clk,
  input  rst,
  // Input channel
  input  ins_valid,
  output ins_ready,
  // Output channel
  output outs_valid,
  input  outs_ready
);
  wire [NUM_SLOTS : 0] valid;
  wire [NUM_SLOTS : 0] ready;

  genvar gen;

  generate
    for (gen = NUM_SLOTS - 1; gen >= 0; gen = gen - 1) begin: slots
        tbf_dataless slot (
            .clk(clk),
            .rst(rst),
            .ins_valid(valid[gen + 1]),
            .ins_ready(ready[gen + 1]),
            .outs_valid(valid[gen]),
            .outs_ready(ready[gen])
        );
    end
  endgenerate

  assign valid[NUM_SLOTS] = ins_valid;
  assign ins_ready = ready[NUM_SLOTS];

  assign outs_valid = valid[0];
  assign ready[0] = outs_ready;

endmodule