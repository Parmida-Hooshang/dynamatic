//===- HandshakeMaterialize.h - Materialize Handshake IR --------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the Handshake materialization pass using a mix of simple rewrite
// steps and a couple rewrite patterns applied greedily on the IR.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/HandshakeMaterialize.h"
#include "dynamatic/Dialect/Handshake/HandshakeCanonicalize.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/MemoryInterfaces.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Transforms/BufferPlacement/Utils/BufferingSupport.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/ErrorHandling.h"
#include <iterator>

using namespace mlir;
using namespace dynamatic;

// [START Boilerplate code for the MLIR pass]
#include "dynamatic/Transforms/Passes.h" // IWYU pragma: keep
namespace dynamatic {
#define GEN_PASS_DEF_HANDSHAKEMATERIALIZE
#include "dynamatic/Transforms/Passes.h.inc"
} // namespace dynamatic
// [END Boilerplate code for the MLIR pass]

/// End of warning message when a fork output that needs its data/valid path
/// cut already carries a conflicting constraint (only used in
/// non-instantiating mode, see `insertLSQGroupAllocationBuffers`).
static const llvm::StringLiteral
    ERR_CONFLICT("but previous channel constraints prevent buffering. "
                 "Resulting circuit may deadlock.");

/// Determines whether the value should be concerned by materialization rules;
/// only SSA values with dataflow semantics must have a single use.
static inline bool eligibleForMaterialization(Value val) {
  return isa<handshake::ControlType, handshake::ChannelType>(val.getType());
}

LogicalResult dynamatic::verifyIRMaterialized(handshake::FuncOp funcOp) {
  auto checkUses = [&](Operation *op, Value val, StringRef desc,
                       unsigned idx) -> LogicalResult {
    if (!eligibleForMaterialization(val))
      return success();

    auto numUses = std::distance(val.getUses().begin(), val.getUses().end());
    if (numUses == 0)
      return op->emitError() << desc << " " << idx << " has no uses.";
    if (numUses > 1)
      return op->emitError() << desc << " " << idx << " has multiple uses.";
    return success();
  };

  // Check function arguments
  for (BlockArgument funcArg : funcOp.getBody().getArguments()) {
    if (failed(checkUses(funcOp, funcArg, "function argument",
                         funcArg.getArgNumber())))
      return failure();
  }

  // Check results of operations
  for (Operation &op : llvm::make_early_inc_range(funcOp.getOps())) {
    for (OpResult res : op.getResults()) {
      if (failed(checkUses(&op, res, "result", res.getResultNumber())))
        return failure();
    }
  }
  return success();
}

LogicalResult dynamatic::verifyIRMaterialized(mlir::ModuleOp modOp) {
  for (handshake::FuncOp funcOp : modOp.getOps<handshake::FuncOp>()) {
    if (failed(verifyIRMaterialized(funcOp)))
      return failure();
  }
  return success();
}

/// Replaces the first use of `oldVal` by `newVal` in the operation's operands.
/// Asserts if the operation's operands do not contain the old value.
static void replaceFirstUse(Operation *op, Value oldVal, Value newVal) {
  for (unsigned i = 0, e = op->getNumOperands(); i < e; ++i) {
    if (op->getOperand(i) == oldVal) {
      op->setOperand(i, newVal);
      return;
    }
  }
  llvm_unreachable("failed to find operation operand");
}

/// Materializes a value, potentially placing an eager fork (if it has more than
/// one uses) or a sink (if it has no uses) to ensure that it is used exactly
/// once.
static void materializeValue(Value val, OpBuilder &builder) {
  if (!eligibleForMaterialization(val))
    return;
  if (val.use_empty()) {
    builder.setInsertionPointAfterValue(val);
    builder.create<handshake::SinkOp>(val.getLoc(), val);
    return;
  }
  if (val.hasOneUse())
    return;

  // The value has multiple uses, collect its owners
  unsigned numUses = std::distance(val.getUses().begin(), val.getUses().end());
  SmallVector<Operation *> valUsers;
  for (OpOperand &oprd : val.getUses())
    valUsers.push_back(oprd.getOwner());

  // Insert a fork with as many results as the value has uses
  builder.setInsertionPointAfterValue(val);
  auto forkOp = builder.create<handshake::ForkOp>(val.getLoc(), val, numUses);
  if (Operation *defOp = val.getDefiningOp())
    inheritBB(defOp, forkOp);

  // Replace original uses of the value with the fork's results
  for (auto [user, forkRes] : llvm::zip_equal(valUsers, forkOp->getResults()))
    replaceFirstUse(user, val, forkRes);
}

/// Converts all eager forks to lazy forks (used for --lazy-forks).
static void promoteAllForksToLazy(handshake::FuncOp funcOp) {
  OpBuilder builder(funcOp->getContext());
  for (handshake::ForkOp forkOp :
       llvm::make_early_inc_range(funcOp.getOps<handshake::ForkOp>())) {
    builder.setInsertionPoint(forkOp);
    auto lazyForkOp = builder.create<handshake::LazyForkOp>(
        forkOp->getLoc(), forkOp.getOperand(), forkOp->getNumResults());
    inheritBB(forkOp, lazyForkOp);
    forkOp->replaceAllUsesWith(lazyForkOp->getResults());
    forkOp->erase();
  }

  // Tag every control merge so that RTL export instantiates the --lazy-forks
  // variant.
  for (handshake::ControlMergeOp cmergeOp :
       funcOp.getOps<handshake::ControlMergeOp>())
    cmergeOp->setAttr("handshake.lazy_forks", builder.getUnitAttr());
}

/// Promotes some eager forks to lazy forks to ensure that different group
/// allocations to the same LSQ do not arrive to the LSQ on the same cycle. This
/// assumes that the IR is materialized. If `lazyForks` is true, calls
/// `promoteAllForksToLazy` and converts all eager forks to lazy forks.
static void promoteEagerToLazyForks(handshake::FuncOp funcOp, bool lazyForks) {
  if (lazyForks) {
    promoteAllForksToLazy(funcOp);
    return;
  }

  // Associate all eager forks feeding group allocation signals to any LSQ to
  // the set of their results that must become lazy
  llvm::DenseMap<handshake::ForkOp, llvm::SetVector<Value>> lazyChannels;
  for (auto lsqOp : funcOp.getOps<handshake::LSQOp>()) {
    LSQPorts lsqPorts = lsqOp.getPorts();
    ValueRange lsqInputs = lsqOp.getOperands();
    for (LSQGroup &group : lsqPorts.getGroups()) {
      Value groupCtrl = lsqInputs[group->ctrlPort->getCtrlInputIndex()];
      Operation *ctrlDefOp = groupCtrl.getDefiningOp();

      // There can only be other control paths to the same LSQ if the defining
      // operation is a fork (eager or lazy). The defining operation is allowed
      // to be `nullptr` if the control value is a function argument
      if (isa_and_present<handshake::LazyForkOp>(ctrlDefOp)) {
        // If there is already a lazy fork in the IR for some reason, assume
        // someone knows what they are doing and do not demote any result
        continue;
      }
      auto forkOp = dyn_cast_if_present<handshake::ForkOp>(ctrlDefOp);
      if (!forkOp)
        continue;

      // Find outputs of the control fork which are part of the memory network
      // and add them to the set of fork results that must be lazy. A single
      // control path means that no other group allocation is reachable from the
      // fork, so the result can be eager
      SmallVector<Value> ctrlResults = lsqOp.getControlPaths(forkOp);
      assert(!ctrlResults.empty() && "at least one control path must exist");
      if (ctrlResults.size() > 1)
        lazyChannels[forkOp].insert(ctrlResults.begin(), ctrlResults.end());
    }
  }

  // Promote eager fork results to lazy where necessary
  OpBuilder builder(funcOp->getContext());
  for (auto &[forkOp, lazyResults] : lazyChannels) {
    unsigned numLazyForkOutputs = lazyResults.size();
    bool hasValueWithoutLazyConstr =
        numLazyForkOutputs < forkOp->getNumResults();
    if (hasValueWithoutLazyConstr) {
      // To minimize damage to performance, as many outputs of the control fork
      // as possible should remain "eager". We achieve this by creating an eager
      // fork after the lazy fork that handles token duplication outside the
      // memory control network. The lazy fork needs an extra output to feed the
      // eager fork
      ++numLazyForkOutputs;
    }

    builder.setInsertionPoint(forkOp);
    handshake::LazyForkOp lazyForkOp = builder.create<handshake::LazyForkOp>(
        forkOp->getLoc(), forkOp.getOperand(), numLazyForkOutputs);
    inheritBB(forkOp, lazyForkOp);

    // Replace the original fork's outputs that are part of the memory control
    // network with the first lazy fork's outputs
    for (auto [from, to] : llvm::zip(lazyResults, lazyForkOp->getResults())) {
      // The elements of a SetVector are const, while replaceAllUsesWith is a
      // non-const method of Value in newer LLVM versions. Value is a handle to
      // the IR value, so replacing the uses through a copy replaces them in the
      // IR
      Value(from).replaceAllUsesWith(to);
    }

    if (hasValueWithoutLazyConstr) {
      // If some of the control fork's result go outside the memory control
      // network, create an eager fork fed by the lazy fork's last result
      unsigned numValuesWithoutLazyConstr =
          forkOp->getNumResults() - lazyResults.size();

      if (numValuesWithoutLazyConstr == 1) {
        // If there is only one eager output channel, we just use the lazy
        // fork's last result to drive that one
        for (OpResult res : forkOp.getResults())
          if (!lazyResults.contains(res))
            res.replaceAllUsesWith(lazyForkOp->getResults().back());
      } else {
        handshake::ForkOp eagerForkOp = builder.create<handshake::ForkOp>(
            forkOp->getLoc(), lazyForkOp->getResults().back(),
            numValuesWithoutLazyConstr);
        inheritBB(forkOp, eagerForkOp);

        // Replace the control fork's outputs that do not belong to the memory
        // control network with the eager fork's results
        ValueRange eagerResults = eagerForkOp.getResult();
        auto eagerForkResIt = eagerResults.begin();
        for (OpResult res : forkOp.getResults()) {
          if (!lazyResults.contains(res))
            res.replaceAllUsesWith(*(eagerForkResIt++));
        }
        assert(eagerForkResIt == eagerResults.end() &&
               "did not exhaust iterator");
      }
    }

    // Erase the original fork whose results are now unused
    forkOp->erase();
  }
}

static void makeUnbufferizable(Value val) {
  assert(!val.use_empty() &&
         "Cannot treat a value without a use as a channel!");

  buffer::Channel channel(val, true);
  channel.props->maxOpaque = 0;
  channel.props->maxTrans = 0;
}

/// Marks memory reference function arguments and memory interface ports as
/// unbufferizable.
static void setMemoryUnbufferizableProperties(handshake::FuncOp funcOp) {
  // See docs/Specs/Buffering.md
  // Memrefs are not real edges in the graph and are therefore unbufferizable
  for (BlockArgument arg : funcOp.getArguments())
    makeUnbufferizable(arg);

  // Ports of memory interfaces are unbufferizable
  for (auto memOp : funcOp.getOps<handshake::MemoryOpInterface>()) {
    FuncMemoryPorts ports = getMemoryPorts(memOp);
    for (size_t i = 0, e = ports.getNumGroups(); i < e; ++i) {
      for (Value inputVal : ports.getGroupInputs(i))
        makeUnbufferizable(inputVal);
      for (Value outputVal : ports.getGroupResults(i))
        makeUnbufferizable(outputVal);
    }
    for (Value inputVal : ports.getInterfacesInputs())
      makeUnbufferizable(inputVal);
    for (Value outputVal : ports.getInterfacesResults())
      makeUnbufferizable(outputVal);
  }
}

/// Ensures the speculator's volatile channels (dataOut, issueCtrl, and
/// trigger) get at least one transparent buffer slot.
static void insertSpeculatorVolatilityBuffers(handshake::FuncOp funcOp,
                                              bool instantiate) {
  OpBuilder builder(funcOp.getContext());

  // Handles `val`, either by setting a constraint on it, or by inserting a
  // buffer between it and its consumer. Works regardless of whether `val` 
  // is conceptually an input or an output of the operation of interest.
  auto handleValue = [&](Value val) {
    if (!instantiate) {
      buffer::Channel channel(val, true);
      channel.props->minTrans = std::max(channel.props->minTrans, 1U);
      return;
    }

    // This pass can run twice on the same function (once before speculation, 
    // once more right after --handshake-speculation), so skip if there is a
    // buffer on either side of this value.
    if (isa_and_present<handshake::BufferOp>(val.getDefiningOp()) ||
        isa<handshake::BufferOp>(*val.getUsers().begin()))
      return;

    Operation *consumer = *val.getUsers().begin();
    builder.setInsertionPoint(consumer);
    auto bufOp = builder.create<handshake::BufferOp>(
        val.getLoc(), val, /*numSlots=*/1, handshake::BufferType::ONE_SLOT_BREAK_R);
    inheritBB(consumer, bufOp);
    consumer->replaceUsesOfWith(val, bufOp->getResult(0));
  };

  for (handshake::SpeculatorOp specOp :
       funcOp.getOps<handshake::SpeculatorOp>()) {
    handleValue(specOp.getDataOut());
    handleValue(specOp.getIssueCtrl());
    handleValue(specOp.getTrigger());
  }
}

/// Ensures every input channel of Store/Load operations that connect to an
/// LSQ has at least one slot.
static void insertLoadStoreBuffers(handshake::FuncOp funcOp, bool instantiate) {
  OpBuilder builder(funcOp.getContext());

  auto handleOperand = [&](Operation *memPortOp, unsigned oprdIdx) {
    Value operand = memPortOp->getOperand(oprdIdx);
    Operation *defOp = operand.getDefiningOp();
    if (!defOp)
      return;

    if (!instantiate) {
      buffer::Channel channel(operand, true);
      channel.props->minSlots = std::max(channel.props->minSlots, 1U);
      return;
    }

    // This pass can run twice on the same function (once before speculation, 
    // once more right after --handshake-speculation), so skip if operands 
    // are already fed by a buffer.
    if (isa<handshake::BufferOp>(defOp))
      return;

    builder.setInsertionPoint(memPortOp);
    auto bufOp = builder.create<handshake::BufferOp>(
        operand.getLoc(), operand, /*numSlots=*/1,
        handshake::BufferType::ONE_SLOT_BREAK_R);
    inheritBB(memPortOp, bufOp);
    memPortOp->setOperand(oprdIdx, bufOp->getResult(0));
  };

  for (handshake::StoreOp storeOp : funcOp.getOps<handshake::StoreOp>()) {
    if (!isa_and_present<handshake::LSQOp>(
            findMemInterface(storeOp.getAddressResult())))
      continue;
    for (unsigned i = 0, e = storeOp->getNumOperands(); i < e; ++i)
      handleOperand(storeOp, i);
  }

  for (handshake::LoadOp loadOp : funcOp.getOps<handshake::LoadOp>()) {
    if (!isa_and_present<handshake::LSQOp>(
            findMemInterface(loadOp.getAddressResult())))
      continue;
    for (unsigned i = 0, e = loadOp->getNumOperands(); i < e; ++i) {
      Operation *defOp = loadOp->getOperand(i).getDefiningOp();
      if (defOp &&
          isa<handshake::MemoryOpInterface, handshake::ConstantOp>(defOp))
        continue;
      handleOperand(loadOp, i);
    }
  }
}

/// Sets buffering constraints related to the LSQ's control path. Output
/// channels of group-allocation-signal-defining (lazy-)forks to the LSQ must be
/// buffered in a particular way:
/// - direct outputs between the fork and LSQs must remain unbuffered
/// - outputs going to next group allocations to the same LSQ must have their
/// data/valid paths cut
/// - all other outputs must have their ready path cut
static void insertLSQGroupAllocationBuffers(handshake::FuncOp funcOp,
                                            OpBuilder &builder,
                                            bool instantiate) {
  for (handshake::LSQOp lsqOp : funcOp.getOps<handshake::LSQOp>()) {
    LSQPorts ports = lsqOp.getPorts();
    ValueRange lsqInputs = lsqOp.getOperands();

    for (LSQGroup &group : ports.getGroups()) {
      // Control signal must come from a fork for this to apply
      Value ctrlVal = lsqInputs[group->ctrlPort->getCtrlInputIndex()];
      Operation *ctrlDefOp = ctrlVal.getDefiningOp();
      if (!mlir::isa_and_present<handshake::ForkOp, handshake::LazyForkOp>(
              ctrlDefOp))
        continue;

      // Fork output channels triggering group allocations to the same LSQ
      // need their data/valid paths cut
      SmallVector<Value> ctrlPaths = lsqOp.getControlPaths(ctrlDefOp);
      llvm::SmallSetVector<Value, 4> ctrlPathSet(ctrlPaths.begin(),
                                                 ctrlPaths.end());
      for (OpResult forkRes : ctrlDefOp->getResults()) {
        if (!ctrlPathSet.contains(forkRes))
          continue;

        // Channels connecting directly to LSQs should be left alone (group
        // allocation signals have already been rendered unbufferizable)
        Operation *consumer = *forkRes.getUsers().begin();
        if (isa<handshake::LSQOp>(consumer))
          continue;

        if (!instantiate) {
          buffer::Channel channel(forkRes, true);
          if (channel.props->maxOpaque.value_or(1) > 0) {
            channel.props->minOpaque = std::max(channel.props->minOpaque, 1U);
          } else {
            OpOperand &oprd = channel.getOperand();
            ctrlDefOp->emitWarning()
                << "Fork result " << forkRes.getResultNumber() << " ("
                << getUniqueName(oprd)
                << ") is on path to other LSQ group allocation and should "
                   "have its data/valid paths cut, "
                << ERR_CONFLICT;
          }
          continue;
        }

        // This pass can run twice on the same function. Skip if already
        // buffered.
        if (isa<handshake::BufferOp>(consumer))
          continue;

        builder.setInsertionPoint(consumer);
        auto bufOp = builder.create<handshake::BufferOp>(
            forkRes.getLoc(), forkRes, /*numSlots=*/1,
            handshake::BufferType::ONE_SLOT_BREAK_DV);
        inheritBB(consumer, bufOp);
        consumer->replaceUsesOfWith(forkRes, bufOp->getResult(0));
      }
    }
  }
}

namespace {

/// Removes outputs of forks that do not have real uses. This can result in the
/// size reduction or deletion of fork operations (the latter if none of the
/// fork results have real users) as well as sink users of fork results.
struct MinimizeForkSizes : OpRewritePattern<handshake::ForkOp> {
  using OpRewritePattern<handshake::ForkOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::ForkOp forkOp,
                                PatternRewriter &rewriter) const override {
    // Compute the list of fork results that are actually used (erase any sink
    // user along the way)
    SmallVector<Value> usedForkResults;
    for (OpResult res : forkOp.getResults()) {
      if (hasRealUses(res)) {
        usedForkResults.push_back(res);
      } else if (!res.use_empty()) {
        // The value has sink users, delete them as the fork producing their
        // operand will be removed
        for (Operation *sinkUser : llvm::make_early_inc_range(res.getUsers()))
          rewriter.eraseOp(sinkUser);
      }
    }
    // Fail if all fork results are used, since it means that no transformation
    // is requires
    if (usedForkResults.size() == forkOp->getNumResults())
      return failure();

    if (!usedForkResults.empty()) {
      // Create a new fork operation
      rewriter.setInsertionPoint(forkOp);
      handshake::ForkOp newForkOp = rewriter.create<handshake::ForkOp>(
          forkOp.getLoc(), forkOp.getOperand(), usedForkResults.size());
      inheritBB(forkOp, newForkOp);

      // Replace results with actual uses of the original fork with results from
      // the new fork
      ValueRange newResults = newForkOp.getResult();
      for (auto [oldRes, newRes] : llvm::zip(usedForkResults, newResults))
        rewriter.replaceAllUsesWith(oldRes, newRes);
    }
    rewriter.eraseOp(forkOp);
    return success();
  }
};

/// Eliminates forks feeding into other forks by replacing both with a single
/// fork operation.
struct EliminateForksToForks : OpRewritePattern<handshake::ForkOp> {
  using mlir::OpRewritePattern<handshake::ForkOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::ForkOp forkOp,
                                PatternRewriter &rewriter) const override {
    // The defining operation must be also be a fork for the pattern to apply
    Value forkOprd = forkOp.getOperand();
    auto defForkOp = forkOprd.getDefiningOp<handshake::ForkOp>();
    if (!defForkOp)
      return failure();

    // It is important to take into account whether the matched fork's operand
    // is the single use of the defining fork's corresponding result. If it is
    // not, the new combined fork needs an extra result to replace the defining
    // fork's result with in the other uses
    bool isForkOprdSingleUse = forkOprd.hasOneUse();

    // Create a new combined fork to replace the two others
    unsigned totalNumResults =
        forkOp->getNumResults() + defForkOp.getNumResults();
    if (isForkOprdSingleUse)
      --totalNumResults;
    rewriter.setInsertionPoint(defForkOp);
    handshake::ForkOp newForkOp = rewriter.create<handshake::ForkOp>(
        defForkOp.getLoc(), defForkOp.getOperand(), totalNumResults);
    inheritBB(defForkOp, newForkOp);

    // Replace the defining fork's results with the first results of the new
    // fork (skipping the result feeding the matched fork if it has a single
    // use)
    ValueRange newResults = newForkOp->getResults();
    auto newResIt = newResults.begin();
    for (OpResult defForkRes : defForkOp->getResults()) {
      if (!isForkOprdSingleUse || defForkRes != forkOprd)
        rewriter.replaceAllUsesWith(defForkRes, *(newResIt++));
    }

    // Replace the results of the matched fork with the corresponding results of
    // the new defining fork
    rewriter.replaceOp(forkOp, newResults.take_back(forkOp.getNumResults()));
    return success();
  }
};

/// Erases forks with a single result unless their operand originates from a
/// lazy fork, in which case they may exist to prevent a combinational cycle.
struct EraseSingleOutputForks : OpRewritePattern<handshake::ForkOp> {
  using mlir::OpRewritePattern<handshake::ForkOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::ForkOp forkOp,
                                PatternRewriter &rewriter) const override {
    // The fork must have a single result
    if (forkOp.getNumResults() != 1)
      return failure();

    // The defining operation must not be a lazy fork, otherwise the fork may be
    // here to avoid a combination cycle between the valid and ready wires
    if (forkOp.getOperand().getDefiningOp<handshake::LazyForkOp>())
      return failure();

    // Bypass the fork and succeed
    rewriter.replaceOp(forkOp, forkOp.getOperand());
    return success();
  }
};

/// Replace
/// - Before: source -> constant -> fork -> {use1, use2, use3, ...}
/// - After:  {source -> constant -> use1, source -> constant -> use2, ...}
struct ReplicateSourceIntoConstant : OpRewritePattern<handshake::ForkOp> {
  using mlir::OpRewritePattern<handshake::ForkOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::ForkOp forkOp,
                                PatternRewriter &rewriter) const override {

    // size = 2 because it is generally source -> constant
    SmallVector<Operation *, 2> opsToReplicate;

    // Check if the fork delivers constants like above:
    std::function<bool(Operation *)> isDrivenByConstRecursive =
        [&](Operation *op) {
          if (isa<
                  // clang-format off
                  handshake::TruncIOp,
                  handshake::ExtSIOp,
                  handshake::ExtUIOp,
                  handshake::ConstantOp
                  // clang-format on
                  >(op)) {
            opsToReplicate.push_back(op);
            Value inputOprd = op->getOperand(0);
            if (Operation *defOp = inputOprd.getDefiningOp(); defOp)
              return isDrivenByConstRecursive(defOp);
            assert(isa<BlockArgument>(inputOprd));
            return false;
          }
          return isa<handshake::SourceOp>(op);
        };

    Operation *definingOp = forkOp.getOperand().getDefiningOp();

    // e.g., function argument
    if (!definingOp)
      return failure();

    if (!isDrivenByConstRecursive(definingOp))
      return failure();

    for (auto res : forkOp.getResults()) {
      // Replicate the source -> constant network:
      auto srcOp = rewriter.create<handshake::SourceOp>(forkOp.getLoc());
      inheritBB(forkOp, srcOp);
      Value oprd = srcOp.getResult();
      for (auto *op : llvm::reverse(opsToReplicate)) {
        Operation *dupOp = rewriter.clone(*op);
        dupOp->setOperand(0, oprd);
        dupOp->removeAttr(NameAnalysis::ATTR_NAME);
        oprd = dupOp->getResult(0);
      }
      rewriter.replaceAllUsesWith(res, oprd);
    }

    return success();
  }
};

/// Driver for the materialization pass, materializing the IR in three
/// sequential steps.
/// 1. First, forks and sinks are inserted within Handshake functions to ensure
/// that every value is used exactly once.
/// 2. Then, unnecessary forks and sinks are erased from the IR in a greedy
/// fashion. If the input IR contained no forks or sinks, this step should do
/// nothing. If the input IR was partially materialized, this may optimize away
/// some of the forks and sinks.
/// 3. Finally, eager forks feeding group allocation signals to LSQs are turned
/// into lazy forks to ensure (together with a correct buffer placement) that
/// multiple group allocations never happen during the same cycle.
struct HandshakeMaterializePass
    : public dynamatic::impl::HandshakeMaterializeBase<
          HandshakeMaterializePass> {

  using HandshakeMaterializeBase::HandshakeMaterializeBase;

  void runOnOperation() override {
    // Make sure all operation names are unique and haven't changed from what is
    // cached (if anything is cached)
    NameAnalysis &namer = getAnalysis<NameAnalysis>();
    if (!namer.isAnalysisValid())
      return signalPassFailure();

    mlir::ModuleOp modOp = getOperation();
    MLIRContext *ctx = &getContext();

    // First make sure that every value within Handshake functions is used
    // exactly once
    OpBuilder builder(ctx);
    for (handshake::FuncOp funcOp : modOp.getOps<handshake::FuncOp>()) {
      for (BlockArgument funcArg : funcOp.getBody().getArguments())
        materializeValue(funcArg, builder);
      for (Operation &op : llvm::make_early_inc_range(funcOp.getOps())) {
        for (OpResult res : op.getResults())
          materializeValue(res, builder);
      }
    }

    // Then, greedily optimize forks
    mlir::GreedyRewriteConfig config;
    config.setUseTopDownTraversal(true);
    config.setRegionSimplificationLevel(
        mlir::GreedySimplifyRegionLevel::Disabled);
    RewritePatternSet patterns{ctx};
    patterns.add<
        // clang-format off
        MinimizeForkSizes,
        EliminateForksToForks,
        EraseSingleOutputForks
        // clang-format on
        >(ctx);

    if (this->replicateConstants)
      patterns.add<ReplicateSourceIntoConstant>(ctx);

    if (failed(applyPatternsGreedily(modOp, std::move(patterns), config)))
      return signalPassFailure();

    // Finally, promote forks to lazy wherever necessary
    for (handshake::FuncOp funcOp : modOp.getOps<handshake::FuncOp>())
      promoteEagerToLazyForks(funcOp, this->lazyForks);

    // The IR is now fully materialized. Only now can we safely attach buffering
    // constraints that are anchored to a specific channel: memory-interface
    // ports (and memrefs) remain unbufferizable, and the speculator's
    // volatile channels get a transparent buffer slot.
    for (handshake::FuncOp funcOp : modOp.getOps<handshake::FuncOp>()) {
      setMemoryUnbufferizableProperties(funcOp);
      insertSpeculatorVolatilityBuffers(funcOp, this->instantiateBuffers);
      insertLoadStoreBuffers(funcOp, this->instantiateBuffers);
      insertLSQGroupAllocationBuffers(funcOp, builder, this->instantiateBuffers);
    }

    assert(succeeded(verifyIRMaterialized(modOp)) && "IR is not materialized");

    // Make sure all operation names are still unique and name operations that
    // do not currently have a name
    if (failed(namer.walk(NameAnalysis::UnnamedBehavior::NAME)))
      return signalPassFailure();
    markAnalysesPreserved<NameAnalysis>();
  }
};
} // namespace
