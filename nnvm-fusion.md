# Note

Some notes about graph fusion in NNVM.

## GraphFindFusibleGroups

Input: source graph (`g: Graph`).

Output: decorated source graph (with attributes `group_root`, `group_master` and `pattern`).

Note for above attributes:

* `group_root`: vector of partitions to which the nodes belong in graph.
* `group_master`: indices of master nodes for each nodes in graph.
* `pattern`: pattern of operators of all nodes in graph.

Process:

1.  Get reference counter of graph nodes. **Output nodes will be countered twice**.
2.  Get operator pattern of each nodes. Mark input nodes as `kOpaque`.
3.  Get fuse rule of each nodes. Mark both input nodes and output nodes as `FuseRule::kRealize`.
4.  Get master index of each nodes. Mark output nodes as index of itself.
5.  Get group root (node partition).
6.  If optimization is enabled, fuse parent node to children nodes, and recalculate group root info.

## GraphFuse

Requires: `GraphFindFusibleGroups`.

Input: source graph (`g: Graph`).

Output: decorated source graph (with attributes `fused_entry`).

Note for about attribute:

* `fused_entry`: vector of fused entries (`struct FuseEntry`).

Process:

1.  Calculate input shape of all non-placeholder nodes, and then create input nodes and info (tensor which shape is calculated shape). Store above info into corresponding `FuseEntry`.
2.  Setup the subgraph of fused entries. Create subgraph node only if current node is group root node.
3.  Store fused entries into graph.

## GraphCompile

Requires: `GraphFuse`

Dependency: `CompileEngine::DoLower`, `DecorateMemoryPlan`

Input: source graph (`g: Graph`).

Output: fused graph (with attribute `module`).

Node for above attribute:

* `module`: compiled module of generated fused graph.

Process:

1.  Lower all group roots (fusable nodes) to a fused function using `CompileEngile::DoLower`. Store the result into a vector.
2.  Generate mapping between source graph nodes and new nodes in fused graph. Create all nodes in fused graph in this process.
3.  Construct fused graph using above mapping info. Specially handle assign operator.
4.  Setup module by calling `nnvm.compiler.build_target`. Handle assign by calling `DecorateMemoryPlan`.

## CompileEngine::DoLower

Used by: `GraphCompile`

Inputs:

* `graph: Graph`: subgraph of fused entry.
* `inputs: Array<Tensor>`: input placeholders of subgraph.
* `target: string`: target platform.
* `master_idx: int`: index of master node in subgraph.

Output: compiled fused function, used by fused node as a new operator.

Process:

1.  **TODO: need to debug**.
