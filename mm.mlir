func.func @matmul(%lhs : tensor<4864x8256xf16>, %rhs : tensor<4096x8256xf16>) -> tensor<4864x4096xf16> {
        %init = tensor.empty() : tensor<4864x4096xf16>
        %cst_0 = arith.constant 0.000000e+00 : f16
        %empty_tensor = tensor.empty() : tensor<8256x4096xf16>
        %2 = linalg.fill ins(%cst_0 : f16) outs(%empty_tensor : tensor<8256x4096xf16>) -> tensor<8256x4096xf16>
        %3 = linalg.fill ins(%cst_0 : f16) outs(%init : tensor<4864x4096xf16>) -> tensor<4864x4096xf16>
        %transposed = linalg.transpose ins(%rhs : tensor<4096x8256xf16>) outs(%2 : tensor<8256x4096xf16>) permutation = [1, 0]
        %matmul = linalg.matmul ins(%lhs, %transposed : tensor<4864x8256xf16>, tensor<8256x4096xf16>) outs(%3 : tensor<4864x4096xf16>) -> tensor<4864x4096xf16>
        return %matmul : tensor<4864x4096xf16>
}