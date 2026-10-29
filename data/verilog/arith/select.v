`timescale 1ns/1ps

module tbf_dataless (
  input  clk,
  input  rst,
  // Input channel
  input  ins_valid,
  output ins_ready,
  // Output channel
  output outs_valid,
  input  outs_ready,
  // Slot occupancy (not needed by tfifo_dataless, used by selector)
  output is_full
);

  reg full = 1'b0;
  assign is_full = full;

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

module selector #(
  parameter DATA_TYPE = 32
)(
  // inputs
  input  clk,
  input  rst,
  input  condition,
  input  condition_valid,
  input  [DATA_TYPE-1 : 0] trueValue,
  input  trueValue_valid,
  input  [DATA_TYPE-1 : 0] falseValue,
  input  falseValue_valid,
  input  result_ready,
  // outputs
  output  [DATA_TYPE-1 : 0] result,
  output  result_valid,
  output  condition_ready,
  output  trueValue_ready,
  output  falseValue_ready
);

  // Each data input has a one-slot antitoken store (a tbf). An antitoken is
  // pushed when the select fires without that input's token, and popped when
  // the discarded token arrives. Since the tbf can pop and push in the same
  // cycle, the select can fire again in the cycle the previous discarded token
  // arrives, instead of stalling one cycle to clear the antitoken.
  wire ee, validInternal, antitokenStop, g0, g1;
  wire kill0, kill1, full0, full1, atReady0, atReady1;
  wire trueValid, falseValid;

  // While an antitoken is pending on an input, the token arriving there is the
  // discarded one: it is killed and must not be used.
  assign trueValid = trueValue_valid & !full0;
  assign falseValid = falseValue_valid & !full1;

  // condition and one input
  assign ee = condition_valid & ( ( !condition & falseValid) | (condition & trueValid) );
  // Stop if firing would need to push an antitoken into a full store that is
  // not popped this cycle
  assign antitokenStop = !atReady0 | !atReady1;
  assign validInternal = ee & !antitokenStop;

  assign g0 = !trueValid & validInternal & result_ready;
  assign g1 = !falseValid & validInternal & result_ready;

  assign result_valid = validInternal;
  assign trueValue_ready = !trueValue_valid | (validInternal & result_ready) | kill0; // normal join or antitoken
  assign falseValue_ready = !falseValue_valid | (validInternal & result_ready) | kill1; // normal join or antitoken
  assign condition_ready = !condition_valid | (validInternal & result_ready); // normal join

  assign result = condition ? trueValue : falseValue;

  tbf_dataless antitoken0 (
    .clk(clk),
    .rst(rst),
    .ins_valid(g0),
    .ins_ready(atReady0),
    .outs_valid(kill0),
    .outs_ready(trueValue_valid),
    .is_full(full0)
  );

  tbf_dataless antitoken1 (
    .clk(clk),
    .rst(rst),
    .ins_valid(g1),
    .ins_ready(atReady1),
    .outs_valid(kill1),
    .outs_ready(falseValue_valid),
    .is_full(full1)
  );

endmodule
