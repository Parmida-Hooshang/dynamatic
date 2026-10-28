//===- HandshakeSetBufferingProperties.h - Set buf. props. ------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the --handshake-set-buffering-properties pass. For now there is
// only a single policy, but it is expected that more will be defined in the
// future. Similarly to the default "fpga20", specifying a new policy amounts to
// writing a simple function that will be called on each channel present in the
// dataflow circuit and modify, if necessary, the buffering properties
// associated with it. The logic for fetching and writing back that data to the
// IR is conveniently hidden to reduce the amount of boilerplate code and
// improve performance.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Transforms/BufferPlacement/Utils/BufferingSupport.h"
#include "dynamatic/Transforms/HandshakeMaterialize.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/STLExtras.h"

using namespace dynamatic;
using namespace dynamatic::buffer;
// [START Boilerplate code for the MLIR pass]

#include "dynamatic/Transforms/Passes.h" // IWYU pragma: keep
namespace dynamatic {
#define GEN_PASS_DEF_HANDSHAKESETBUFFERINGPROPERTIES
#include "dynamatic/Transforms/Passes.h.inc"
} // namespace dynamatic
// [END Boilerplate code for the MLIR pass]

/// End of error message when there is a conflict between the constraint-setting
/// logic and already existing buffering constraints.
static const llvm::StringLiteral
    ERR_CONFLICT("but previous channel constraints prevent buffering. "
                 "Resulting circuit may deadlock.");

/// Sets the ready-path-cut constraint on LSQ group-allocation control forks'
/// output channels that do not lead to another group allocation of the same
/// LSQ (not needed if --lazy-forks is used).
static void setLSQReadyPathConstraints(handshake::LSQOp lsqOp) {
  LSQPorts ports = lsqOp.getPorts();
  ValueRange lsqInputs = lsqOp.getOperands();

  for (LSQGroup &group : ports.getGroups()) {
    // Control signal must come from a fork for this constraint to apply
    Value ctrlVal = lsqInputs[group->ctrlPort->getCtrlInputIndex()];
    Operation *ctrlDefOp = ctrlVal.getDefiningOp();
    if (!mlir::isa_and_present<handshake::ForkOp, handshake::LazyForkOp>(
            ctrlDefOp))
      continue;

    // Fork output channels not on a control path to a group allocation of
    // this LSQ get a transparent buffer slot constraint
    SmallVector<Value> ctrlPaths = lsqOp.getControlPaths(ctrlDefOp);
    llvm::SmallSetVector<Value, 4> ctrlPathSet(ctrlPaths.begin(),
                                               ctrlPaths.end());
    for (OpResult forkRes : ctrlDefOp->getResults()) {
      if (ctrlPathSet.contains(forkRes))
        continue;

      // Channels connecting directly to LSQs should be left alone (group
      // allocation signals have already been marked unbufferizable in 
      // handshake materialization).
      if (isa<handshake::LSQOp>(*forkRes.getUsers().begin()))
        continue;

      Channel channel(forkRes, true);
      if (channel.props->maxTrans.value_or(1) > 0) {
        channel.props->minTrans = std::max(channel.props->minTrans, 1U);
      } else {
        OpOperand &oprd = channel.getOperand();
        ctrlDefOp->emitWarning()
            << "Fork result " << forkRes.getResultNumber() << " ("
            << getUniqueName(oprd)
            << ") is *not* on path to other LSQ group allocation and "
                "should have its ready path cut, "
            << ERR_CONFLICT;
      }
    }
  }
}

// The speculator requires specific buffers for correctness.
static LogicalResult setSpeculatorBufferingProperties(handshake::FuncOp funcOp) {
  auto specOps = funcOp.getOps<handshake::SpeculatorOp>();
  auto specCount = std::distance(specOps.begin(), specOps.end());
  if (specCount > 1) {
    funcOp.emitError() << "Expected at most one SpeculatorOp";
    return failure();
  }
  
  if (specCount == 0)
    return success();

  auto specOp = *specOps.begin();

  // historyCtrl is on a path from a lazy fork to a join and so needs 
  // a buffer to prevent deadlock (if --lazy-forks is not used).
  Value historyCtrl = specOp.getHistoryCtrl();
  Channel resolveChannel(historyCtrl, true);
  resolveChannel.props->minTrans = std::max(resolveChannel.props->minTrans, 1U);

  return success();
}

static LogicalResult setFPGA20Properties(handshake::FuncOp funcOp) {
  // See docs/Specs/Buffering.md
  // A merge with more than one input should have at least one
  // buffer slot at its output, and this is necessary only if
  // the merge is on a cycle.
  for (handshake::MergeOp mergeOp : funcOp.getOps<handshake::MergeOp>()) {
    if (mergeOp->getNumOperands() > 1) {
      for (OpResult mergeRes : mergeOp->getResults()) {
        Channel channel(mergeRes, true);
        if (isChannelOnCycle(mergeRes)) {
          channel.props->minSlots = std::max(channel.props->minSlots, 1U);
        }
      }
    }
  }

  // Init ops replace init-merge patterns on loop backedges, so their output
  // also needs at least one buffer slot when it participates in a cycle.
  for (handshake::InitOp initOp : funcOp.getOps<handshake::InitOp>()) {
    for (OpResult initRes : initOp->getResults()) {
      Channel channel(initRes, true);
      if (isChannelOnCycle(initRes)) {
        channel.props->minSlots = std::max(channel.props->minSlots, 1U);
      }
    }
  }

  // See docs/Specs/Buffering.md
  // Control paths to LSQs have specific properties
  for (handshake::LSQOp lsqOp : funcOp.getOps<handshake::LSQOp>())
    setLSQReadyPathConstraints(lsqOp);

  if (failed(setSpeculatorBufferingProperties(funcOp)))
    return failure();

  return success();
}

namespace {
/// Simple pass driver that runs a specific buffering properties setting
/// policy on each Handshake function in the IR.
struct HandshakeSetBufferingPropertiesPass
    : public dynamatic::impl::HandshakeSetBufferingPropertiesBase<
          HandshakeSetBufferingPropertiesPass> {

  using HandshakeSetBufferingPropertiesBase::
      HandshakeSetBufferingPropertiesBase;

  void runOnOperation() override {
    // Make sure all operation names are unique and haven't changed from what is
    // cached (if anything is cached)
    NameAnalysis &namer = getAnalysis<NameAnalysis>();
    if (!namer.isAnalysisValid())
      return signalPassFailure();

    mlir::ModuleOp modOp = getOperation();
    // Check that the provided version is valid
    if (version != "fpga20") {
      modOp->emitError() << "Unkwown version \"" << version
                         << "\", expected one of [fpga20]";
      return signalPassFailure();
    }

    // Add properties to channels inside each function
    for (handshake::FuncOp funcOp : modOp.getOps<handshake::FuncOp>()) {
      // Buffer placement requires that all values are used exactly once
      if (failed(verifyIRMaterialized(funcOp))) {
        funcOp.emitOpError() << ERR_NON_MATERIALIZED_FUNC;
        return signalPassFailure();
      }
      if (failed(setFPGA20Properties(funcOp)))
        return signalPassFailure();
    }

    // Make sure all operation names are still unique and name operations that
    // do not currently have a name
    if (failed(namer.walk(NameAnalysis::UnnamedBehavior::NAME)))
      return signalPassFailure();
    markAnalysesPreserved<NameAnalysis>();
  };
};

} // namespace
