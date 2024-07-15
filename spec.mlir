#rocm_target = #hal.executable.target<"rocm", "rocm-hsaco-fb", {target_arch = "gfx942", ukernels = "none"}>

module attributes {transform.with_named_sequence} {

  util.func private @mm_external_func(%arg0: tensor<4864x8256xf16>, %arg1: tensor<8256x4096xf16>) -> tensor<4864x4096xf16> {
    %M = arith.constant 4864 : i32
    %N = arith.constant 4096 : i32
    %K = arith.constant 8256 : i32
    %stride_am = arith.constant 8256 : i32
    %stride_bn = arith.constant 8256 : i32
    %stride_cm = arith.constant 4096 : i32
    %total_full_tiles_streamk = arith.constant 258 : i32
    %total_partial_tiles_streamk = arith.constant 0 : i32
    %iters_per_tile = arith.constant 129 : i32
    %total_tiles_streamk = arith.constant 608 : i32
    %total_programs_streamk = arith.constant 304 : i32

    %cst_0 = arith.constant 0.000000e+00 : f16
    %out_empty = tensor.empty() : tensor<4864x4096xf16>
    %out_splat = linalg.fill ins(%cst_0 : f16) outs(%out_empty : tensor<4864x4096xf16>) -> tensor<4864x4096xf16>

    %5 = hal.dispatch.extern "streamk_gemm"(%M, %N, %K, %stride_am, %stride_bn, %stride_cm, %total_full_tiles_streamk,
    %total_partial_tiles_streamk, %iters_per_tile, %total_tiles_streamk, %total_programs_streamk, %arg0, %arg1, %out_splat) : 
        (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, tensor<4864x8256xf16>, tensor<8256x4096xf16>, tensor<4864x4096xf16>) -> %out_splat
      count(%device: !hal.device) -> (index, index, index) {
        %c1_0 = arith.constant 1 : index
        %c80_0 = arith.constant 304 : index
        hal.return %c80_0, %c1_0, %c1_0 : index, index, index
      }
      layout(#hal.pipeline.layout<push_constants = 11, sets = [
        <0, bindings = [
            <0, storage_buffer, ReadOnly>,
            <1, storage_buffer, ReadOnly>,
            <2, storage_buffer>
        ]>
      ]>)
      bindings([
        #hal.interface.binding<0, 0>,
        #hal.interface.binding<0, 1>,
        #hal.interface.binding<0, 2>
      ])
      objects({
        #rocm_target ordinal(0) = [
          #hal.executable.object<{
            path = "/home/nmeganat/triton_kernel_integration/streamk_gemm.hsaco"
          }>
        ]
      })
      attributes {subgroupSize = 64, workgroup_size = [256 : index, 1 : index, 1 : index]}
    util.return %5 : tensor<4864x4096xf16>
  }

  transform.named_sequence @cast_and_call_dag(%ins: !transform.any_value {transform.readonly},
                                              %out: !transform.any_value {transform.readonly}) {
    %root = transform.get_defining_op %out : (!transform.any_value) -> !transform.any_op
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @mm_external_func into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
      //  transform.type_conversion.tensor.cast_shape_dynamic_dims
      } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    // transform.print {name = "hi"}
    transform.yield
  }

  transform.named_sequence @match_mm(
    %root: !transform.any_op {transform.readonly}) -> (!transform.any_value, !transform.any_value) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %root {
      ^bb0(%lhs: tensor<4864x8256xf16>, %rhs : tensor<8256x4096xf16>):
        %init = tensor.empty() : tensor<4864x4096xf16>
        %cst_0 = arith.constant 0.000000e+00 : f16
        %3 = linalg.fill ins(%cst_0 : f16) outs(%init : tensor<4864x4096xf16>) -> tensor<4864x4096xf16>
        %matmul = linalg.matmul ins(%lhs, %rhs : tensor<4864x8256xf16>, tensor<8256x4096xf16>) outs(%3 : tensor<4864x4096xf16>) -> tensor<4864x4096xf16>
      } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    transform.yield %ins, %outs : !transform.any_value, !transform.any_value
  }

  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %funcs = transform.structured.match ops{["util.func"]} in %module : (!transform.any_op) -> !transform.any_op
    transform.foreach %funcs : !transform.any_op {
      ^bb1(%func: !transform.any_op):
        transform.foreach_match in %func
            @match_mm -> @cast_and_call_dag
          : (!transform.any_op) -> (!transform.any_op)
    }
    transform.apply_dce to %module : !transform.any_op
    transform.apply_registered_pass "inline" to %module : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}