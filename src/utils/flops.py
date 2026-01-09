import math
from typing import Any, List


def scaled_dot_product_attention_flop_jit(inputs: List[Any], outputs: List[Any]):
    """
    Count flops for the aten::scaled_dot_product_attention operator.
    """
    q, k, v = inputs[:3]
    q_shape = q.type().sizes()
    k_shape = k.type().sizes()
    v_shape = v.type().sizes()

    l_q = q_shape[-2]
    l_k = k_shape[-2]
    d_qk = k_shape[-1]
    d_v = v_shape[-1]
    batch_size = math.prod(q_shape[:-2]) #includinf number of head

    att_flops = l_q * l_k * d_qk
    softmax_flops = l_q * l_k * 2 # exp + (sum + div)
    out_flops = l_q * l_k * d_v
    return batch_size * (att_flops + softmax_flops + out_flops)

def point_wise_flop_jit(inputs: List[Any], outputs: List[Any]):
    x_shape = inputs[0].type().sizes()
    return math.prod(x_shape)

def double_point_wise_flop_jit(inputs: List[Any], outputs: List[Any]):
    x_shape = inputs[0].type().sizes()
    return math.prod(x_shape)

def two_input_point_wise_flop_jit(inputs: List[Any], outputs: List[Any]):
    x_shape = inputs[0].type().sizes()
    y_shape = inputs[0].type().sizes()
    return max(math.prod(x_shape),math.prod(y_shape))

def single_flop_jit(inputs: List[Any], outputs: List[Any]):
    return 1

def print_flop_jit(inputs: List[Any], outputs: List[Any]):
    print(inputs)
    return 0


def get_new_ops():
    return {
        'aten::scaled_dot_product_attention': scaled_dot_product_attention_flop_jit,
        'aten::gelu': point_wise_flop_jit,
        'aten::add': two_input_point_wise_flop_jit,
        'aten::add_': two_input_point_wise_flop_jit,
        'aten::mul': single_flop_jit,
        'aten::mul_': single_flop_jit,
        'aten::exp': point_wise_flop_jit,
        'aten::sigmoid': point_wise_flop_jit,
        'aten::softmax': double_point_wise_flop_jit,
        'aten::div': two_input_point_wise_flop_jit,
        'aten::rsub': single_flop_jit,
        'aten::linalg_vector_norm': point_wise_flop_jit,
    }
