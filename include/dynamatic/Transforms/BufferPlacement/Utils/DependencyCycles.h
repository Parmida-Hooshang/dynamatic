//===- DependencyCycles.h - Circuit dependency graph ---------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Only considers handshake signals (ready and valid), and builds a graph
// (SignalGraph) with vertices representing the handshake ports of 
// components and edges representing dependencies between ports. It's as 
// if we are magnifying the components into their handshake ports and 
// signals. To resolve dependency cycles, it introduces a rank (dependency
// order) variable per signal, with every edge requiring the destination's 
// rank to be at least the source's. Additionally, ready edges require a 
// strict increase unless that channel is buffered. This forces every 
// dependency cycle to have at least one buffer (of any type) placed on 
// one of its ready edges.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_UTILS_DEPENDENCY_CYCLES_H
#define DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_UTILS_DEPENDENCY_CYCLES_H

#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/ConstraintProgramming/ConstraintProgramming.h"
#include "dynamatic/Support/LLVM.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringMap.h"
#include <string>
#include <utility>
#include <vector>

namespace dynamatic {
namespace buffer {

/// Classifies the nodes' internal valid/ready dependency patterns.
enum class ComponentKind {
  /// Every input's valid to every output's valid, and every output's 
  /// ready to every input's ready.
  Generic,
  Join,
  /// A pipelined binary op (floating-point `addf`, `subf`, `mulf`, `divf`,
  /// `maximumf`, `minimumf`, `cmpf` with nonzero latency, and integer `muli`,
  /// `divsi`, `divui`, `remsi`) whose module instantiates a join between its
  /// two operands followed by a pipeline before the result. The join's 
  /// rule between operands applies (e.g. `rhs_valid -> lhs_ready`), as well
  /// as the input-to-output valid, but not the output-to-input ready because
  /// the pipeline provides input ready.
  PipelinedJoin,
  LazyFork,
  CondBranch,
  FullBreak,
  /// Unrelated pass-through channels (like `handshake::StoreOp`). Not
  /// "every" input to "every" output unlike Generic.
  IndependentChannels,
  Mux,
  ControlMerge,
  Select,
};

enum class SignalKind { Valid, Ready };

struct SignalPort {
  bool isInput;
  unsigned idx;
  SignalKind kind;
};

/// A dependency not captured by the node's `ComponentKind`.
struct MixedDomainEdge {
  SignalPort src;
  SignalPort dst;
};

struct PortConnection {
  /// Index of the port on the node this connection belongs to.
  unsigned ownIdx;
  std::string neighborId;
  unsigned neighborIdx;
  /// This is the same `Channel` that buffer placement (e.g., FPL22) 
  /// reads/writes buffering properties on.
  Value channel;
};

struct DependencyNode {
  std::string id;
  Operation *op = nullptr;
  std::vector<PortConnection> predecessors;
  std::vector<PortConnection> successors;
  ComponentKind kind = ComponentKind::Generic;
  std::vector<MixedDomainEdge> mixedDomainEdges;
};

/// Graph mirroring the dataflow circuit's connectivity (each node
/// is a component, not a port yet).
class DependencyGraph {
public:
  DependencyNode &addNode(std::string id, Operation *op = nullptr);

  const std::vector<DependencyNode> &getNodes() const { return nodes; }

private:
  std::vector<DependencyNode> nodes;
};

/// Walks `funcOp` and populates `graph` with one node per operation.
void buildDependencyGraph(handshake::FuncOp funcOp, DependencyGraph &graph);

/// Graph obtained by "magnifying" every node of a `DependencyGraph` into one
/// node per valid/ready wire of its input and output ports, and connecting
/// them according to how the original nodes are wired to each other and
/// each component's internal valid/ready dependencies.
class SignalGraph {
public:
  struct ChannelEdgeGroup {
    std::pair<size_t, size_t> validEdge;
    std::pair<size_t, size_t> readyEdge;
    Value channel;
  };

  void addEdge(const std::string &src, const std::string &dst);

  void addInternalEdge(const std::string &src, const std::string &dst);

  void addChannelEdges(const std::string &producerValid,
                       const std::string &consumerValid,
                       const std::string &consumerReady,
                       const std::string &producerReady, Value channel);

  /// Number of signal nodes in the graph (one per valid/ready wire).
  size_t numNodes() const { return nodes.size(); }

  StringRef getNodeId(size_t idx) const { return nodes[idx]; }

  ArrayRef<ChannelEdgeGroup> getChannelEdgeGroups() const {
    return channelPairs;
  }

  /// Every internal dependency edge.
  ArrayRef<std::pair<size_t, size_t>> getInternalEdges() const {
    return internalEdges;
  }

private:
  /// Returns the node index for `id`, creating a new entry in `nodes`
  /// if it doesn't already exist.
  size_t getOrAddNode(const std::string &id);

  /// Maps a signal's id to its position in `nodes`.
  llvm::StringMap<size_t> index;
  std::vector<std::string> nodes;
  std::vector<ChannelEdgeGroup> channelPairs;
  std::vector<std::pair<size_t, size_t>> internalEdges;
};

std::string getSignalId(StringRef nodeId, bool isInput, bool isValid,
                        unsigned idx);

/// Builds the signal-level graph obtained by magnifying every node of
/// `graph` into its constituent valid/ready wires.
void buildSignalGraph(
    const DependencyGraph &graph, SignalGraph &sig,
    llvm::function_ref<bool(Operation *)> unitFilter =
        [](Operation *) { return true; },
    llvm::function_ref<bool(Value)> channelFilter =
        [](Value) { return true; });

/// Adds rank constraints. Every edge requires the destination's rank 
/// to be at least the source's. Additionally, ready edges require a 
/// strict increase unless that channel is buffered. This forces every
/// dependency cycle to have at least one buffer (of any type) placed 
/// on one of its ready edges.
void addDependencyRankConstraints(
    const SignalGraph &sigGraph, CPSolver &model,
    llvm::function_ref<CPVar &(Value)> getChannelBufNumSlots);

} // namespace buffer
} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_UTILS_DEPENDENCY_CYCLES_H
