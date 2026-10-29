//===- FPL22Buffers.cpp - FPL'22 buffer placement ---------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements FPL'22 smart buffer placement.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/BufferPlacement/FPL22Buffers.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/Attribute.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/TimingModels.h"
#include "dynamatic/Transforms/BufferPlacement/Utils/BufferingSupport.h"
#include "dynamatic/Transforms/BufferPlacement/Utils/CFDFC.h"
#include "llvm/ADT/TypeSwitch.h"
#include <iterator>
#include <limits>
#include <optional>

// NOTE: The code wrapped in LLVM_DEBUG(...) is executed when
// - Dynamatic is built in debug mode
// - dynamatic-opt is called with `--debug` or `--debug-only=<DEBUG_TYPE>`.
#define DEBUG_TYPE "fpl22-buffers"

using namespace llvm::sys;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;
using namespace dynamatic::buffer::fpl22;

void FPL22BuffersBase::extractResult(BufferPlacement &placement) {
  // Iterate over all channels in the circuit
  for (auto [channel, chVars] : vars.channelVars) {
    // Extract number and type of slots from the MILP solution, as well as
    // channel-specific buffering properties
    unsigned numSlotsToPlace =
        static_cast<unsigned>(model->getValue(chVars.bufNumSlots) + 0.5);
    if (numSlotsToPlace == 0)
      continue;

    bool forceBreakDV =
        model->getValue(chVars.signalVars[SignalType::DATA].bufPresent) > 0;
    bool forceBreakR =
        model->getValue(chVars.signalVars[SignalType::READY].bufPresent) > 0;

    PlacementResult result;
    // A `fast` buffer can accept a new token in the same clock cycle 
    // as the previous token is emitted. A `slow` buffer needs at least
    // one cycle in between. 

    // ONE_SLOT_BREAK_DV = opaque fast
    // ONE_SLOT_BREAK_R = transparent slow
    // FIFO_BREAK_NONE = transparent fast

    // We don't count ONE_SLOT_BREAK_R towards the number of slots
    // because it's slow and cannot accept/emit a token every cycle.

    // 1. If breaking DV & R:
    // ONE_SLOT_BREAK_R + ONE_SLOT_BREAK_DV + (numslot - 1) *
    //                                           FIFO_BREAK_NONE
    //
    // 2. If only breaking DV:
    // ONE_SLOT_BREAK_DV + (numslot - 1) * FIFO_BREAK_NONE.
    //
    // 3. If only breaking R:
    // ONE_SLOT_BREAK_R + numslot * FIFO_BREAK_NONE.
    //
    // 4. If breaking none:
    // numslot * FIFO_BREAK_NONE.
    if (forceBreakDV && forceBreakR) {
      result.numOneSlotR = 1;
      result.numOneSlotDV = 1;
      result.numFifoNone = numSlotsToPlace - 1;
    } else if (forceBreakDV) {
      result.numOneSlotDV = 1;
      result.numFifoNone = numSlotsToPlace - 1;
    } else if (forceBreakR) {
      result.numOneSlotR = 1;
      result.numFifoNone = numSlotsToPlace;
    } else {
      result.numFifoNone = numSlotsToPlace;
    }

    placement[channel] = result;
  }

  LLVM_DEBUG(logResults(placement));

  llvm::MapVector<size_t, double> cfdfcTPResult;
  for (auto [idx, cfdfcWithVars] : llvm::enumerate(vars.cfdfcVars)) {
    auto [cf, cfVars] = cfdfcWithVars;
    double tmpThroughput = model->getValue(cfVars.throughput);

    cfdfcTPResult[idx] = tmpThroughput;
  }

  // Create and add the handshake.tp attribute
  auto cfdfcTPMap = handshake::CFDFCThroughputAttr::get(
      funcInfo.funcOp.getContext(), cfdfcTPResult);
  setDialectAttr(funcInfo.funcOp, cfdfcTPMap);

  populateCFDFCThroughputAndOccupancy();
}

void FPL22BuffersBase::addCustomChannelConstraints(Value channel) {
  // Get channel-specific buffering properties and channel's variables
  handshake::ChannelBufProps &props = channelProps[channel];
  ChannelVars &chVars = vars.channelVars[channel];

  // Force buffer presence if at least one slot is requested
  unsigned minSlots =
      std::max(props.minOpaque + props.minTrans, props.minSlots);
  if (minSlots > 0) {
    model->addConstr(chVars.bufPresent == 1, "custom_forceBuffers");
    model->addConstr(chVars.bufNumSlots >= minSlots, "custom_minSlots");
  }

  // Set constraints based on minimum number of buffer slots
  CPVar &bufData = chVars.signalVars[SignalType::DATA].bufPresent;
  CPVar &bufReady = chVars.signalVars[SignalType::READY].bufPresent;
  if (props.minOpaque > 0) {
    // Force the MILP to place at least one opaque slot
    model->addConstr(bufData == 1, "custom_forceData");
    // If the MILP decides to also place a ready buffer, then we must reserve
    // an extra slot for it
    model->addConstr(chVars.bufNumSlots >= props.minOpaque + bufReady,
                     "custom_minData");
  }
  if (props.minTrans > 0) {
    // If the MILP decides to also place a data buffer, then we must reserve
    // an extra slot for it
    model->addConstr(chVars.bufNumSlots >= props.minTrans + bufData,
                     "custom_minReady");
  }

  // Set constraints based on maximum number of buffer slots
  if (props.maxOpaque && props.maxTrans) {
    unsigned maxSlots = *props.maxOpaque + *props.maxTrans;
    if (maxSlots == 0) {
      // Forbid buffer placement on the channel entirely when no slots are
      // allowed
      model->addConstr(chVars.bufPresent == 0, "custom_noBuffer");
      model->addConstr(chVars.bufNumSlots == 0, "custom_maxSlots");
    } else {
      // Restrict the maximum number of slots allowed. If both types are allowed
      // but the MILP decides to only place one type, then the maximum allowed
      // number is the maximum number of slots we can place for that type
      model->addConstr(chVars.bufNumSlots <=
                           maxSlots - *props.maxOpaque * (1 - bufData) -
                               *props.maxTrans * (1 - bufReady),
                       "custom_maxSlots");
    }
  }

  // Forbid placement of some buffer type based on maximum number of allowed
  // slots on each signal
  if (props.maxOpaque && *props.maxOpaque == 0) {
    // Force the MILP to use transparent slots only
    model->addConstr(bufData == 0, "custom_noData");
  }
  if (props.maxTrans && *props.maxTrans == 0) {
    // Force the MILP to use opaque slots only
    model->addConstr(bufReady == 0, "custom_noReady");
  }
}

namespace {

/// Represents a specific pin of a unit's input or output port. Used internally
/// by the mixed-domain unit path constraints logic.
struct Pin {
  /// The channel connected to the unit's port.
  Value channel;
  /// The pin's timing domain, denoted by a signal type.
  SignalType signalType;

  /// Simple member-by-member constructor.
  Pin(Value channel, SignalType signalType)
      : channel(channel), signalType(signalType) {};
};

/// Represents a mixed domain constraint between an input pin and an output pin,
/// with a combinational delay between the two.
struct MixedDomainConstraint {
  /// The input pin.
  Pin input;
  /// The output pin.
  Pin output;
  /// Combinational delay (in ns) on the path.
  double internalDelay;

  /// Simple member-by-member constructor.
  MixedDomainConstraint(Pin input, Pin output, double internalDelay)
      : input(input), output(output), internalDelay(internalDelay) {};
};

} // namespace

void FPL22BuffersBase::addUnitMixedPathConstraints(Operation *unit,
                                                   ChannelFilter filter) {
  std::vector<MixedDomainConstraint> constraints;
  const TimingModel *unitModel = timingDB.getModel(unit);

  // Adds constraints between the input ports' valid and ready pins of a unit
  // with two operands.
  auto addJoinedOprdConstraints = [&]() -> void {
    double vr = unitModel ? unitModel->validToReady : 0.0;
    Value oprd0 = unit->getOperand(0), oprd1 = unit->getOperand(1);
    constraints.emplace_back(Pin(oprd0, SignalType::VALID),
                             Pin(oprd1, SignalType::READY), vr);
    constraints.emplace_back(Pin(oprd1, SignalType::VALID),
                             Pin(oprd0, SignalType::READY), vr);
  };

  // Adds constraints between the data pin of the provided input channel and all
  // valid/ready output pins.
  auto addDataToAllValidReadyConstraints = [&](Value inputChannel) -> void {
    Pin input(inputChannel, SignalType::DATA);
    double cv = unitModel ? unitModel->condToValid : 0.0;
    for (OpResult res : unit->getResults())
      constraints.emplace_back(input, Pin(res, SignalType::VALID), cv);
    double cr = unitModel ? unitModel->condToReady : 0.0;
    for (Value oprd : unit->getOperands())
      constraints.emplace_back(input, Pin(oprd, SignalType::READY), cr);
  };

  llvm::TypeSwitch<Operation *, void>(unit)
      .Case<handshake::ConditionalBranchOp>(
          [&](handshake::ConditionalBranchOp condBrOp) {
            // There is a path between the data pin of the condition operand and
            // every valid/ready output pin
            addDataToAllValidReadyConstraints(condBrOp.getConditionOperand());

            // The two branch inputs are joined therefore there are cross
            // connections between the valid and ready pins
            addJoinedOprdConstraints();
          })
      .Case<handshake::ControlMergeOp>([&](handshake::ControlMergeOp cmergeOp) {
        // There is a path between the valid pin of the first operand and the
        // data pin of the index result
        Pin input(cmergeOp.getOperand(0), SignalType::VALID);
        Pin output(cmergeOp.getIndex(), SignalType::DATA);
        double vc = unitModel ? unitModel->validToCond : 0.0;
        constraints.emplace_back(input, output, vc);
      })
      .Case<handshake::MergeOp>([&](handshake::MergeOp mergeOp) {
        // There is a path between every valid input pin and the data output
        // pin
        double vd = unitModel ? unitModel->validToData : 0.0;
        Pin output(mergeOp.getResult(), SignalType::DATA);
        for (Value oprd : mergeOp->getOperands())
          constraints.emplace_back(Pin(oprd, SignalType::VALID), output, vd);
      })
      .Case<handshake::MuxOp>([&](handshake::MuxOp muxOp) {
        // There is a path between the data pin of the select operand and every
        // valid/ready output pin
        addDataToAllValidReadyConstraints(muxOp.getSelectOperand());

        // There is a path between every valid input pin and every data/ready
        // output pin
        double vd = unitModel ? unitModel->validToData : 0.0;
        double vr = unitModel ? unitModel->validToReady : 0.0;
        for (Value oprd : muxOp->getOperands()) {
          for (OpResult res : muxOp->getResults()) {
            constraints.emplace_back(Pin(oprd, SignalType::VALID),
                                     Pin(res, SignalType::DATA), vd);
          }
          for (Value readyOprd : muxOp->getOperands()) {
            constraints.emplace_back(Pin(oprd, SignalType::VALID),
                                     Pin(readyOprd, SignalType::READY), vr);
          }
        }
      })
      .Case<handshake::LazyForkOp>([&](handshake::LazyForkOp lazyForkOp) {
        // A lazy fork only raises an output's valid once every other
        // output's ready is high, so there is a path between the ready
        // pin of every output and the valid pin of every other output
        for (OpResult outValid : lazyForkOp->getResults()) {
          for (OpResult outReady : lazyForkOp->getResults()) {
            if (outValid == outReady)
              continue;
            // 0.0 should be replaced with the correct value
            constraints.emplace_back(Pin(outReady, SignalType::READY),
                                     Pin(outValid, SignalType::VALID), 0.0);
          }
        }
      })
      .Case<handshake::LoadOp, handshake::StoreOp, handshake::AddIOp,
            handshake::AddFOp, handshake::SubIOp, handshake::SubFOp,
            handshake::AndIOp, handshake::OrIOp, handshake::XOrIOp,
            handshake::MulIOp, handshake::MulFOp, handshake::DivUIOp,
            handshake::DivSIOp, handshake::DivFOp, handshake::ShRSIOp,
            handshake::ShLIOp, handshake::CmpIOp, handshake::CmpFOp>(
          [&](auto) { addJoinedOprdConstraints(); });

  StringRef unitName = getUniqueName(unit);
  unsigned idx = 0;
  for (MixedDomainConstraint &cons : constraints) {
    // The input/output channels must both be inside the CFDFC union
    if (!filter(cons.input.channel) || !filter(cons.output.channel))
      continue;

    // Find variables for arrival time at input/output pin
    CPVar &tPinIn = vars.channelVars[cons.input.channel]
                        .signalVars[cons.input.signalType]
                        .path.tOut;
    CPVar &tPinOut = vars.channelVars[cons.output.channel]
                         .signalVars[cons.output.signalType]
                         .path.tIn;

    // Arrival time at unit's output pin must be greater than arrival time at
    // unit's input pin plus the unit's internal delay on the path
    std::string consName =
        "path_mixed_" + unitName.str() + "_" + std::to_string(idx++);
    model->addConstr(tPinIn + cons.internalDelay <= tPinOut, consName);
  }
}

CFDFCUnionBuffers::CFDFCUnionBuffers(CPSolver::SolverKind solverKind,
                                     int timeout, FuncInfo &funcInfo,
                                     const TimingDatabase &timingDB,
                                     double targetPeriod, CFDFCUnion &cfUnion,
                                     const buffer::DependencyGraph &depGraph,
                                     StringRef writeTo)
    : FPL22BuffersBase(solverKind, timeout, funcInfo, timingDB, targetPeriod,
                       writeTo),
      cfUnion(cfUnion), depGraph(depGraph) {
  if (!unsatisfiable)
    setup();
}

void CFDFCUnionBuffers::setup() {
  // Signals for which we have variables
  SmallVector<SignalType, 4> signalTypes;
  signalTypes.push_back(SignalType::DATA);
  signalTypes.push_back(SignalType::VALID);
  signalTypes.push_back(SignalType::READY);

  /// NOTE: (lucas-rami) For each buffering group this should be the timing
  /// model of the buffer that will be inserted by the MILP for this group. We
  /// don't have models for these buffers at the moment therefore we provide a
  /// null-model to each group, but this hurts our placement's accuracy.
  const TimingModel *bufModel = nullptr;

  BufferingGroup dataValidGroup({SignalType::DATA, SignalType::VALID},
                                bufModel);
  BufferingGroup readyGroup({SignalType::READY}, bufModel);

  SmallVector<BufferingGroup> bufGroups;
  bufGroups.push_back(dataValidGroup);
  bufGroups.push_back(readyGroup);

  // Group signals by matching buffer type for elasticty constraints
  SmallVector<ArrayRef<SignalType>> signalGroups;
  SmallVector<SignalType> opaqueGroup{SignalType::DATA, SignalType::VALID};
  SmallVector<SignalType> transparentGroup{SignalType::READY};
  signalGroups.push_back(opaqueGroup);
  signalGroups.push_back(transparentGroup);

  // Create channel variables and add custom, path, and elasticity contraints
  // over all channels in the CFDFC union
  for (Value channel : cfUnion.channels) {
    // Create variables and add custom channel constraints
    addChannelVars(channel, signalTypes);
    addCustomChannelConstraints(channel);

    // Add single-domain path constraints
    addChannelTimingConstraints(channel, SignalType::DATA, bufModel, {},
                                readyGroup);
    addChannelTimingConstraints(channel, SignalType::VALID, bufModel, {},
                                readyGroup);
    addChannelTimingConstraints(channel, SignalType::READY, bufModel,
                                dataValidGroup, {});

    // Elasticity constraints
    addBufferPresenceConstraints(channel);
    addBufferingGroupConstraints(channel, bufGroups);
  }

  // For unit constraints, filter out ports that are not part of the CFDFC union
  ChannelFilter channelFilter = [&](Value channel) -> bool {
    return cfUnion.channels.contains(channel);
  };

  // Add single-domain and mixed-domain path constraints as well as elasticity
  // constraints over all units in the CFDFC union
  for (Operation *unit : cfUnion.units) {
    if (isa<handshake::SpeculatorOp, handshake::SpecSaveCommitOp>(unit)) {
      addSpecUnitConstraints(
          unit, {SignalType::DATA, SignalType::VALID, SignalType::READY},
          channelFilter);
      continue;
    }
    addUnitTimingConstraints(unit, SignalType::DATA, channelFilter);
    addUnitTimingConstraints(unit, SignalType::VALID, channelFilter);
    addUnitTimingConstraints(unit, SignalType::READY, channelFilter);
    addUnitMixedPathConstraints(unit, channelFilter);
  }

  // Create CFDFC variables and add throughput constraints for each CFDFC in the
  // union which was marked for optimization
  for (CFDFC *cfdfc : cfUnion.cfdfcs) {
    assert(funcInfo.cfdfcs.contains(cfdfc) && "unknown CFDFC");
    if (!funcInfo.cfdfcs[cfdfc])
      continue;
    addCFDFCVars(*cfdfc);
    addSteadyStateReachabilityConstraints(*cfdfc);
    addChannelThroughputConstraintsForBinaryLatencyChannel(*cfdfc);
    addUnitThroughputConstraints(*cfdfc);
  }

  // Rank (dependency order) constraints for dependency cycles contained in 
  // this union.
  buffer::SignalGraph unionSigGraph;
  buffer::buildSignalGraph(
      depGraph, unionSigGraph,
      [&](Operation *op) { return op && cfUnion.units.contains(op); },
      channelFilter);
  buffer::addDependencyRankConstraints(
      unionSigGraph, *model, [&](Value channel) -> CPVar & {
        return vars.channelVars[channel].bufNumSlots;
      });

  // Add the MILP objective and mark the MILP ready to be optimized
  std::vector<Value> allChannels;
  llvm::copy(cfUnion.channels, std::back_inserter(allChannels));
  addMaxThroughputObjective(allChannels, cfUnion.cfdfcs);
  markReadyToOptimize();
}

NonCFDFCBuffers::NonCFDFCBuffers(
    CPSolver::SolverKind solverKind, int timeout, FuncInfo &funcInfo,
    const TimingDatabase &timingDB, double targetPeriod,
    const buffer::SignalGraph &sigGraph,
    const BufferPlacement &priorPlacement, StringRef writeTo)
    : FPL22BuffersBase(solverKind, timeout, funcInfo, timingDB, targetPeriod,
                       writeTo), 
                       sigGraph(sigGraph), priorPlacement(priorPlacement) {
  if (!unsatisfiable)
    setup();
}

void NonCFDFCBuffers::setup() {
  // Signals for which we have variables
  SmallVector<SignalType, 4> signalTypes;
  signalTypes.push_back(SignalType::DATA);
  signalTypes.push_back(SignalType::VALID);
  signalTypes.push_back(SignalType::READY);

  /// NOTE: (lucas-rami) For each buffering group this should be the timing
  /// model of the buffer that will be inserted by the MILP for this group. We
  /// don't have models for these buffers at the moment therefore we provide a
  /// null-model to each group, but this hurts our placement's accuracy.
  const TimingModel *bufModel = nullptr;

  BufferingGroup dataValidGroup({SignalType::DATA, SignalType::VALID},
                                nullptr);
  BufferingGroup readyGroup({SignalType::READY}, nullptr);
  SmallVector<BufferingGroup> bufGroups;
  bufGroups.push_back(dataValidGroup);
  bufGroups.push_back(readyGroup);

  // Create the expression for the MILP objective
  LinExpr objective;

  ChannelFilter memFilter = [&](Value channel) -> bool {
    Operation *defOp = channel.getDefiningOp();
    return !isa_and_present<handshake::MemoryOpInterface>(defOp) &&
           !isa<handshake::MemoryOpInterface>(*channel.getUsers().begin());
  };

  // Create a CFDFC union from all CFDFCs in the function so that we can make
  // very fast queries of the kind: is this channel part of any CFDFC?
  SmallVector<CFDFC *> allCFDFCs;
  for (auto [cfdfc, _] : funcInfo.cfdfcs)
    allCFDFCs.push_back(cfdfc);
  CFDFCUnion allCFDFCsUnion(allCFDFCs);

  // Create variables and add custom and elasticity constraints for every
  // channel in the function. Unlike `CFDFCUnionBuffers,` there is no 
  // filtering here.
  for (auto &[channel, props] : channelProps) {
    addChannelVars(channel, signalTypes);
    addCustomChannelConstraints(channel);
    addBufferPresenceConstraints(channel);
    addBufferingGroupConstraints(channel, bufGroups);

    ChannelVars &chVars = vars.channelVars[channel];
    CPVar &dataBuf = chVars.signalVars[SignalType::DATA].bufPresent;
    CPVar &readyBuf = chVars.signalVars[SignalType::READY].bufPresent;

    // Read whatever earlier MILPs have decided for this channel.
    unsigned priorSlots = 0;
    bool priorForceData = false, priorForceReady = false;
    if (auto it = priorPlacement.find(channel); it != priorPlacement.end()) {
      const PlacementResult &prior = it->second;
      // We do not count numOneSlotR towards the number of buffer slots
      // because ONE_SLOT_BREAK_R is slow. See FPL22BuffersBase::extractResult.
      priorSlots = prior.numOneSlotDV + prior.numFifoDV +
                  prior.numFifoNone + prior.numOneSlotDVR + prior.numShiftRegDV;
      priorForceData = prior.numOneSlotDV > 0 || prior.numOneSlotDVR > 0 ||
                       prior.numFifoDV > 0 || prior.numShiftRegDV > 0;
      priorForceReady = prior.numOneSlotR > 0 || prior.numOneSlotDVR > 0;
    }

    bool insideCFDFC = allCFDFCsUnion.channels.contains(channel);
    auto priorBufPresentFor = [&](SignalType sig) -> LinExpr {
      return LinExpr(
          (sig == SignalType::READY ? priorForceReady : priorForceData) ? 1.0
                                                                        : 0.0);
    };
    // For a channel inside some CFDFC, the timing constraints use the prior
    // decision as a constant instead of a live `bufPresent` variable.
    llvm::function_ref<LinExpr(SignalType)> bufPresentOverride = nullptr;
    if (insideCFDFC)
      bufPresentOverride = priorBufPresentFor;

    if (memFilter(channel)) {
      addChannelTimingConstraints(channel, SignalType::DATA, bufModel, {},
                                  readyGroup, bufPresentOverride);
      addChannelTimingConstraints(channel, SignalType::VALID, bufModel, {},
                                  readyGroup, bufPresentOverride);
      addChannelTimingConstraints(channel, SignalType::READY, bufModel,
                                  dataValidGroup, {}, bufPresentOverride);
    }

    // Never let the total number of slots go below what an earlier MILP has
    // already decided.
    if (priorSlots > 0)
      model->addConstr(chVars.bufNumSlots >= priorSlots, "prior_minSlots");

    if (insideCFDFC) {
      // Fix the decision for channels inside CFDFCs.
      model->addConstr(dataBuf == (priorForceData ? 1.0 : 0.0),
                       "prior_pinData");
      model->addConstr(readyBuf == (priorForceReady ? 1.0 : 0.0),
                       "prior_pinReady");
    } else {
      // Set lowerbounds for channels outside CFDFCs.
      if (priorForceData)
        model->addConstr(dataBuf >= 1.0, "prior_forceData");
      if (priorForceReady)
        model->addConstr(readyBuf >= 1.0, "prior_forceReady");
    }

    objective -= dataBuf;
    objective -= readyBuf;
    objective -= 0.1 * chVars.bufNumSlots;
  }

  // Rank constraints enforcing every dependency cycle to have at least
  // one buffered ready edge. This one is allowed to place a buffer even
  // inside CFDFCs, because in that case it only places tfifo slots (fast
  // and transparent) which do not damage throughput.
  buffer::addDependencyRankConstraints(
      sigGraph, *model, [&](Value channel) -> CPVar & {
        return vars.channelVars[channel].bufNumSlots;
      });

  // Unit path constraints (single-domain and mixed-domain) over all three
  // signal domains.
  for (Operation &unit : funcInfo.funcOp.getOps()) {
    if (isa<handshake::SpeculatorOp, handshake::SpecSaveCommitOp>(&unit)) {
      addSpecUnitConstraints(
          &unit, {SignalType::DATA, SignalType::VALID, SignalType::READY},
          memFilter);
      continue;
    }
    addUnitTimingConstraints(&unit, SignalType::DATA, memFilter);
    addUnitTimingConstraints(&unit, SignalType::VALID, memFilter);
    addUnitTimingConstraints(&unit, SignalType::READY, memFilter);
    addUnitMixedPathConstraints(&unit, memFilter);
  }

  // Set MILP objective and mark it ready to be optimized
  model->setMaximizeObjective(objective);
  markReadyToOptimize();
}
