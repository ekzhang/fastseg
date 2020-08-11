"""Command line script to export a pretrained segmentation model to ONNX."""

import argparse
import sys

import torch
import geffnet
import fastseg

parser = argparse.ArgumentParser(description=__doc__)

parser.add_argument('output', metavar='OUTPUT_FILENAME',
    help='filename of output model (e.g., mobilenetv3_large.onnx)')
parser.add_argument('--model', '-m', default='MobileV3Large',
    help='the model to export (default MobileV3Large)')
parser.add_argument('--num_filters', '-F', type=int, default=128,
    help='the number of filters in the segmentation head (default 128)')
parser.add_argument('--size', '-s', default='1024,2048',
    help='the image dimensions to set as input (default 1024,2048)')
parser.add_argument('--checkpoint', '-c', default=None,
    help='filename of the weights checkpoint .pth file (uses pretrained by default)')

args = parser.parse_args()

print(f'==> Creating PyTorch {args.model} model')
if args.model == 'MobileV3Large':
    model_cls = fastseg.MobileV3Large
elif args.model == 'MobileV3Small':
    model_cls = fastseg.MobileV3Small
else:
    print(f'Unknown model name: {args.model}', file=sys.stderr)
    sys.exit(1)

geffnet.config.set_exportable(True)
model = model_cls.from_pretrained(args.checkpoint, num_filters=args.num_filters)
model.eval()

print('==> Exporting to ONNX')
height, width = [int(x) for x in args.size.split(',')]
print(f'Image dimensions: {height} x {width}')
print(f'Output file: {args.output}')

dummy_input = torch.randn(1, 3, height, width)
input_names = ['input0']
output_names = ['output0']

# Run model once, this is required by geffnet
model(dummy_input)

torch.onnx.export(model, dummy_input, args.output, verbose=True,
    input_names=input_names, output_names=output_names,
    opset_version=11, keep_initializers_as_inputs=True)

# Check the model
print(f'==> Finished export, loading and checking model: {args.output}')
import onnx
onnx_model = onnx.load(args.output)
onnx.checker.check_model(onnx_model)
print('==> Passed check')
