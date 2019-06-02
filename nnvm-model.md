# NNVM Model

Some note about converting a TensorFlow model to NNVM symbol, and compile it to dynamic library.

## nnvm.frontend.from_tensorflow

Using this function to convert a TensorFlow `GraphDef` to NNVM symbol.

This function is a wrapper, after simple preprocess, it will call `GraphProto.from_tensorflow`.

## nnvm.frontend.tensorflow.GraphProto.from_tensorflow

Construct NNVM nodes from TensorFlow graph definition.

Inputs:

* `graph: tensorflow.GraphDef`: TensorFlow graph definition object.
* `layout`: target layout, for example, `'NHWC'`
* `shape`: input dimensions.

Outputs:

* `sym: nnvm.sym.Symbol`: NNVM symbol.
* `param: dict`: parameter dictionary to be used by NNVM.

Process:

* Check if there are unsupported operators of NNVM in TensorFlow's graph definition.
* Preprocess all `Placeholder` nodes, consider them as `Variable`.
* Preprocess all `Const` nodes, consider all as params.
* Assume last node of graph as output.
* Ignore `DecodeJpeg` and `ResizeBilinear`.
* Consider `CheckNumberics` as copy operators.

## nnvm.frontend.tensorflow._convert_map

A dictionary that stores mappings of TensorFlow node name to internal casting functions.

## nnvm.compiler.build

Build NNVM graph into runtime library.

Used Pass:

* `CorrectLayout`@`correct_layout.cc`: A simple layout infer & correct pass that will insert layout transform nodes automatically.
* `InferShape`@`infer_shape_type.cc`: Infer the shape of each node entries.
* `InferType`@`infer_shape_type.cc`: Infer the dtype of each node entries.
* `AlterOpLayout`@`alter_op_layout.cc`: Alter the operator layouts. Keep inferred layouts (if any) from previous stages.
* `SimplifyInference`@`simplify_inference.cc`: Simplify inference.
* `FoldScaleAxis`@`fold_scale_axis.cc`: Fold scaling parameter of axis into weight of conv/dense.
* `PrecomputePrune`@`precompute_prune.cc`: Split the graph into a pre-compute graph and a execution graph.
* `GraphFindFusibleGroups`, `GraphFuse`, `GraphCompile`: See [nnvm-fusion.md](nnvm-fusion.md).

## nnvm.compiler.save_param_dict

Save parameter dictionary to binary bytes.

This function is a wrapper of `nnvm.compiler._save_param_dict` in file `graph_runtime.cc`.

## nnvm.compiler._save_param_dict

Save parameter dictionary to binary bytes. This function is a `tvm::runtime::PackedFunc`.
