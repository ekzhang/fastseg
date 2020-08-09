"""Command line script to optimize an ONNX model."""

import argparse

import onnx
from onnx import optimizer

parser = argparse.ArgumentParser(description=__doc__)

parser.add_argument('input', metavar='INPUT_FILENAME',
    help='filename of input model (e.g., mobilenetv3_large.onnx)')
parser.add_argument('output', metavar='OUTPUT_FILENAME',
    help='filename of output model (e.g., mobilenetv3_large.opt.onnx)')

args = parser.parse_args()

print(f'==> Loading model {args.input}')
original_model = onnx.load(args.input)

print('Number of nodes before optimization:', len(original_model.graph.node))

print('==> Optimizing')
passes = [
    'eliminate_identity',
    'eliminate_nop_dropout',
    'eliminate_nop_monotone_argmax',
    'eliminate_nop_pad',
    'eliminate_nop_transpose',
    'eliminate_unused_initializer',
    'extract_constant_to_initializer',
    'fuse_add_bias_into_conv',
    'fuse_bn_into_conv',
    'fuse_consecutive_concats',
    'fuse_consecutive_log_softmax',
    'fuse_consecutive_reduce_unsqueeze',
    'fuse_consecutive_squeezes',
    'fuse_consecutive_transposes',
    'fuse_pad_into_conv',
    'nop',
]

optimized_model = optimizer.optimize(original_model, passes)

print('Number of nodes after optimization:', len(optimized_model.graph.node))

print(f'Output file: {args.output}')
onnx.save(optimized_model, args.output)
