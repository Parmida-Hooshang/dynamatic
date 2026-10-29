//===- DependencyCycles.cpp - Circuit dependency graph -------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements DependencyCycles.h.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/BufferPlacement/Utils/DependencyCycles.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/STLExtras.h"
#include <cctype>

using namespace mlir;
using namespace dynamatic;

namespace dynamatic {
namespace buffer {

DependencyNode &DependencyGraph::addNode(std::string id, Operation *op) {
  nodes.push_back(DependencyNode{std::move(id), op, {}, {}});
  return nodes.back();
}

static std::string getNodeId(Operation *op) { return getUniqueName(op).str(); }

/// Classifies internal valid/ready dependency patterns.
static ComponentKind getComponentKind(Operation *op) {
  if (isa<handshake::JoinOp, handshake::AddIOp, handshake::SubIOp,
         handshake::AndIOp, handshake::OrIOp, handshake::XOrIOp,
         handshake::ShLIOp, handshake::ShRSIOp, handshake::ShRUIOp,
         handshake::CmpIOp, handshake::MinSIOp, handshake::MinUIOp,
         handshake::MaxSIOp, handshake::MaxUIOp>(op))
    return ComponentKind::Join;
  // `handshake::EndOp` itself is not a real module, but the testbench's
  // `tb_join` instantiates a join.
  if (isa<handshake::EndOp>(op))
    return ComponentKind::Join;
  // cmpf's module is a register-free join when its latency is 0
  // (flopoco single precision), and a join followed by a pipeline 
  // otherwise (flopoco double precision, vivado).
  if (isa<handshake::CmpFOp>(op)) {
    auto latencyIface = dyn_cast<handshake::LatencyInterface>(op);
    if (latencyIface) {
      FailureOr<int64_t> latency = latencyIface.getLatency();
      if (succeeded(latency) && *latency == 0)
        return ComponentKind::Join;
    }
    return ComponentKind::PipelinedJoin;
  }
  if (isa<handshake::AddFOp, handshake::SubFOp, handshake::MulFOp,
          handshake::DivFOp, handshake::MaximumFOp,
          handshake::MinimumFOp, handshake::MulIOp, handshake::DivSIOp,
          handshake::DivUIOp, handshake::RemSIOp>(op))
    return ComponentKind::PipelinedJoin;
  if (isa<handshake::LazyForkOp>(op))
    return ComponentKind::LazyFork;
  if (isa<handshake::ConditionalBranchOp>(op))
    return ComponentKind::CondBranch;
  if (isa<handshake::BufferOp, handshake::ForkOp, handshake::LoadOp,
         handshake::MemoryControllerOp, handshake::LSQOp,
         handshake::SinkOp, handshake::SourceOp,
         // sitofp/fptosi/uitofp are unary casts that each wrap their result
         // directly in an OEHB.
         handshake::SIToFPOp, handshake::FPToSIOp, handshake::UIToFPOp,
         handshake::NDWireOp>(op))
    return ComponentKind::FullBreak;
  if (isa<handshake::StoreOp>(op))
    return ComponentKind::IndependentChannels;
  if (isa<handshake::MuxOp>(op))
    return ComponentKind::Mux;
  if (auto cmergeOp = dyn_cast<handshake::ControlMergeOp>(op))
    return ComponentKind::ControlMerge;
  if (isa<handshake::SelectOp>(op))
    return ComponentKind::Select;
  return ComponentKind::Generic;
}

/// Fills in `dstId`/`channel` for the consumer.
static void describeConsumer(OpOperand &oprd, std::string &dstId,
                             Value &channel) {
  Operation *dstOp = oprd.getOwner();
  channel = oprd.get();
  dstId = getNodeId(dstOp);
}

/// Adds intra-node dependencies not captured by `getComponentKind`.
static void addMixedDomainEdges(Operation *op,
                                std::vector<MixedDomainEdge> &edges) {
  if (auto lsqOp = dyn_cast<handshake::LSQOp>(op)) {
    // `ctrlEnd`'s ready depends on its own valid pin.
    if (!lsqOp.isConnectedToMC()) {
      unsigned ctrlEndIdx = op->getNumOperands() - 1;
      // Every group's own ctrl (group-allocation) valid depends on
      // `ctrlEnd`'s ready.
      LSQPorts ports = lsqOp.getPorts();
      for (LSQGroup &group : ports.getGroups()) {
        unsigned ctrlIdx = group->ctrlPort->getCtrlInputIndex();
        edges.push_back({{/*isInput=*/true, ctrlIdx, SignalKind::Valid},
                         {/*isInput=*/true, ctrlEndIdx, SignalKind::Ready}});
      }
    }
  }
}

void buildDependencyGraph(handshake::FuncOp funcOp, DependencyGraph &graph) {
  // Function arguments are the graph's entry points, so they only have
  // successors.
  for (auto [idx, arg] : llvm::enumerate(funcOp.getArguments())) {
    if (isa<MemRefType>(arg.getType()))
      continue;

    std::string argName = funcOp.getArgName(idx).str();
    DependencyNode &argNode = graph.addNode(argName);
    for (OpOperand &oprd : arg.getUses()) {
      std::string dstId;
      Value channel;
      describeConsumer(oprd, dstId, channel);
      argNode.successors.push_back(
          {static_cast<unsigned>(idx), dstId, oprd.getOperandNumber(),
           channel});
    }
  }

  for (Operation &op : funcOp.getOps()) {
    DependencyNode &node = graph.addNode(getNodeId(&op), &op);
    node.kind = getComponentKind(&op);
    addMixedDomainEdges(&op, node.mixedDomainEdges);

    for (unsigned idx = 0, e = op.getNumOperands(); idx < e; ++idx) {
      OpOperand &oprd = op.getOpOperand(idx);
      Value operand = oprd.get();
      std::string srcId;
      unsigned srcIdx;
      if (auto res = dyn_cast<OpResult>(operand)) {
        Operation *srcOp = res.getDefiningOp();
        srcIdx = res.getResultNumber();
        srcId = getNodeId(srcOp);
      } else {
        auto blockArg = cast<BlockArgument>(operand);
        srcIdx = blockArg.getArgNumber();
        srcId = funcOp.getArgName(srcIdx).str();
      }
      node.predecessors.push_back({idx, srcId, srcIdx, operand});
    }

    for (OpResult res : op.getResults()) {
      unsigned ownIdx = res.getResultNumber();
      for (OpOperand &oprd : res.getUses()) {
        std::string dstId;
        Value channel;
        describeConsumer(oprd, dstId, channel);
        node.successors.push_back(
            {ownIdx, dstId, oprd.getOperandNumber(), channel});
      }
    }
  }
}

void SignalGraph::addEdge(const std::string &src, const std::string &dst) {
  getOrAddNode(src);
  getOrAddNode(dst);
}

void SignalGraph::addInternalEdge(const std::string &src,
                                  const std::string &dst) {
  addEdge(src, dst);
  internalEdges.emplace_back(index.lookup(src), index.lookup(dst));
}

void SignalGraph::addChannelEdges(const std::string &producerValid,
                                  const std::string &consumerValid,
                                  const std::string &consumerReady,
                                  const std::string &producerReady,
                                  Value channel) {
  addEdge(producerValid, consumerValid);
  addEdge(consumerReady, producerReady);
  channelPairs.push_back(
      {{index.lookup(producerValid), index.lookup(consumerValid)},
       {index.lookup(consumerReady), index.lookup(producerReady)},
       channel});
}

size_t SignalGraph::getOrAddNode(const std::string &id) {
  auto [it, inserted] = index.try_emplace(id, nodes.size());
  if (inserted)
    nodes.push_back(id);
  return it->second;
}

std::string getSignalId(StringRef nodeId, bool isInput, bool isValid,
                        unsigned idx) {
  std::string id = nodeId.str();
  id += isInput ? ".ins_" : ".outs_";
  id += isValid ? "valid[" : "ready[";
  id += std::to_string(idx);
  id += ']';
  return id;
}

/// Resolves a `SignalPort` to its signal-graph node id.
static std::string resolveSignalPort(StringRef nodeId,
                                     const SignalPort &port) {
  switch (port.kind) {
  case SignalKind::Valid:
    return getSignalId(nodeId, port.isInput, /*isValid=*/true, port.idx);
  case SignalKind::Ready:
    return getSignalId(nodeId, port.isInput, /*isValid=*/false, port.idx);
  }
  llvm_unreachable("unknown SignalKind");
}

/// Every input valid to every output valid.
static void addInputOutputValidEdges(StringRef id, unsigned numIns,
                                     unsigned numOuts, SignalGraph &sig) {
  for (unsigned i = 0; i < numIns; ++i)
    for (unsigned j = 0; j < numOuts; ++j)
      sig.addInternalEdge(getSignalId(id, true, true, i),
                 getSignalId(id, false, true, j));
}

/// Every output ready to every input ready.
static void addOutputInputReadyEdges(StringRef id, unsigned numIns,
                                     unsigned numOuts, SignalGraph &sig) {
  for (unsigned i = 0; i < numOuts; ++i)
    for (unsigned j = 0; j < numIns; ++j)
      sig.addInternalEdge(getSignalId(id, false, false, i),
                 getSignalId(id, true, false, j));
}

/// Assuming Mux has no internal registers.
/// TODO: Remove the internal buffer in mux (python generator).
static void addMuxInternalEdges(StringRef id, unsigned numIns,
                                SignalGraph &sig) {
  std::string indexValid = getSignalId(id, true, true, 0);
  std::string indexReady = getSignalId(id, true, false, 0);
  for (unsigned i = 1; i < numIns; ++i) {
    sig.addInternalEdge(indexValid, getSignalId(id, true, false, i));
    sig.addInternalEdge(getSignalId(id, true, true, i), indexReady);
  }
}

/// Assuming ControlMerge has no internal registers.
/// TODO: Add a lazy implementation for control merge.
static void addControlMergeInternalEdges(StringRef id, SignalGraph &sig) {
  std::string resultValid = getSignalId(id, false, true, 0);
  std::string resultReady = getSignalId(id, false, false, 0);
  std::string indexValid = getSignalId(id, false, true, 1);
  std::string indexReady = getSignalId(id, false, false, 1);

  sig.addInternalEdge(indexReady, resultValid);
  sig.addInternalEdge(resultReady, indexValid);
}

static void addJoinInternalEdges(StringRef id, unsigned numIns,
                                 SignalGraph &sig) {
  for (unsigned i = 0; i < numIns; ++i)
    for (unsigned j = 0; j < numIns; ++j)
      if (i != j)
        sig.addInternalEdge(getSignalId(id, true, true, j),
                   getSignalId(id, true, false, i));
}

static void addLazyForkInternalEdges(StringRef id, unsigned numOuts,
                                     SignalGraph &sig) {
  for (unsigned i = 0; i < numOuts; ++i)
    for (unsigned j = 0; j < numOuts; ++j)
      if (i != j)
        sig.addInternalEdge(getSignalId(id, false, false, j),
                   getSignalId(id, false, true, i));
}

/// Unrelated pass-through channels (like `handshake::StoreOp`).
static void addIndependentChannelsInternalEdges(StringRef id, unsigned numIns,
                                                SignalGraph &sig) {
  for (unsigned i = 0; i < numIns; ++i) {
    sig.addInternalEdge(getSignalId(id, true, true, i), getSignalId(id, false, true, i));
    sig.addInternalEdge(getSignalId(id, false, false, i),
               getSignalId(id, true, false, i));
  }
}

static void addCondBranchInternalEdges(StringRef id, SignalGraph &sig) {
  std::string conditionValid = getSignalId(id, true, true, 0);
  std::string dataValid = getSignalId(id, true, true, 1);
  std::string conditionReady = getSignalId(id, true, false, 0);
  std::string dataReady = getSignalId(id, true, false, 1);

  sig.addInternalEdge(conditionValid, dataReady);
  sig.addInternalEdge(dataValid, conditionReady);
}

static void addSelectInternalEdges(StringRef id, SignalGraph &sig) {
  std::string conditionValid = getSignalId(id, true, true, 0);
  std::string trueValid = getSignalId(id, true, true, 1);
  std::string falseValid = getSignalId(id, true, true, 2);
  std::string conditionReady = getSignalId(id, true, false, 0);
  std::string trueReady = getSignalId(id, true, false, 1);
  std::string falseReady = getSignalId(id, true, false, 2);

  sig.addInternalEdge(conditionValid, falseReady);
  sig.addInternalEdge(trueValid, falseReady);
  sig.addInternalEdge(conditionValid, trueReady);
  sig.addInternalEdge(falseValid, trueReady);
  sig.addInternalEdge(falseValid, conditionReady);
  sig.addInternalEdge(trueValid, conditionReady);
}

void buildSignalGraph(const DependencyGraph &graph, SignalGraph &sig,
                      llvm::function_ref<bool(Operation *)> unitFilter,
                      llvm::function_ref<bool(Value)> channelFilter) {
  for (const DependencyNode &node : graph.getNodes()) {
    unsigned numIns = node.predecessors.size();
    unsigned numOuts = node.successors.size();

    if (unitFilter(node.op)) {
      if (node.kind != ComponentKind::IndependentChannels)
        addInputOutputValidEdges(node.id, numIns, numOuts, sig);

      // Output-to-input ready edges, only for kinds with no register between
      // their inputs and outputs.
      switch (node.kind) {
      case ComponentKind::Generic:
      case ComponentKind::Join:
      case ComponentKind::LazyFork:
      case ComponentKind::CondBranch:
      case ComponentKind::Mux:
      case ComponentKind::ControlMerge:
        addOutputInputReadyEdges(node.id, numIns, numOuts, sig);
        break;
      case ComponentKind::PipelinedJoin:
      case ComponentKind::FullBreak:
      case ComponentKind::IndependentChannels:
      case ComponentKind::Select:
        break;
      }

      // Kind-specific edges.
      switch (node.kind) {
      case ComponentKind::Join:
      case ComponentKind::PipelinedJoin:
        addJoinInternalEdges(node.id, numIns, sig);
        break;
      case ComponentKind::LazyFork:
        addLazyForkInternalEdges(node.id, numOuts, sig);
        break;
      case ComponentKind::CondBranch:
        addCondBranchInternalEdges(node.id, sig);
        break;
      case ComponentKind::FullBreak:
        // Only the input-to-output valid edges added above.
        break;
      case ComponentKind::IndependentChannels:
        addIndependentChannelsInternalEdges(node.id, numIns, sig);
        break;
      case ComponentKind::Generic:
        break;
      case ComponentKind::Mux:
        addMuxInternalEdges(node.id, numIns, sig);
        break;
      case ComponentKind::ControlMerge:
        addControlMergeInternalEdges(node.id, sig);
        break;
      case ComponentKind::Select:
        addSelectInternalEdges(node.id, sig);
        break;
      }

      // Intra-node dependencies not captured by `kind`.
      for (const MixedDomainEdge &edge : node.mixedDomainEdges)
        sig.addInternalEdge(resolveSignalPort(node.id, edge.src),
                   resolveSignalPort(node.id, edge.dst));
    }

    // Inter-node edges: valid flows downstream from an output port to the
    // input port it feeds, while ready flows upstream the other way.
    for (const PortConnection &c : node.successors) {
      if (!channelFilter(c.channel))
        continue;
      sig.addChannelEdges(
          getSignalId(node.id, false, true, c.ownIdx),
          getSignalId(c.neighborId, true, true, c.neighborIdx),
          getSignalId(c.neighborId, true, false, c.neighborIdx),
          getSignalId(node.id, false, false, c.ownIdx), c.channel);
    }
  }
}

void addDependencyRankConstraints(
    const SignalGraph &sigGraph, CPSolver &model,
    llvm::function_ref<CPVar &(Value)> getChannelBufNumSlots) {
  size_t numNodes = sigGraph.numNodes();

  // `rank` is a topological-order placeholder, not a real delay, so
  // any fixed positive increment works.
  const double eps = 1.0;
  // Large enough that a relaxed ready-edge constraint works regardless
  // of the two ranks involved, since both are bounded in [0, numNodes].
  const double bigM = static_cast<double>(numNodes) + 2.0;

  auto sanitize = [](StringRef id) {
    std::string s = id.str();
    for (char &c : s)
      if (!isalnum(static_cast<unsigned char>(c)))
        c = '_';
    return s;
  };

  std::vector<CPVar> rank;
  rank.reserve(numNodes);
  for (size_t i = 0; i < numNodes; ++i)
    rank.push_back(model.addVar("rank_" + sanitize(sigGraph.getNodeId(i)),
                                INTEGER, 0, static_cast<double>(numNodes)));

  // Valid and internal edges deliberately carry no `eps` because we
  // don't need to break dependency cycles that do not involve any
  // ready edge.
  unsigned idx = 0;
  for (const SignalGraph::ChannelEdgeGroup &group :
       sigGraph.getChannelEdgeGroups()) {
    model.addConstr(rank[group.validEdge.first] <= rank[group.validEdge.second],
                    "dep_rank_valid_" + std::to_string(idx++));

    // Ready edge is relaxed once this channel has at least one buffer slot of
    // any kind.
    CPVar &bufNumSlots = getChannelBufNumSlots(group.channel);
    model.addConstr(rank[group.readyEdge.first] + eps - bigM * bufNumSlots <=
                        rank[group.readyEdge.second],
                    "dep_rank_ready_" + std::to_string(idx++));
  }

  for (const std::pair<size_t, size_t> &edge : sigGraph.getInternalEdges())
    model.addConstr(rank[edge.first] <= rank[edge.second],
                    "dep_rank_internal_" + std::to_string(idx++));
}

} // namespace buffer
} // namespace dynamatic
