# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author   : henson.zhang
# @Company  : SHIQING TECH
# @Time     : 2022/4/18 9:58
# @File     : test_operations.py
from typing import Any, List, Optional, Sequence, Text, Union

import numpy as np
import onnx
import onnxruntime as rt
import torch
import torch.nn as nn
from onnx.onnx_pb import AttributeProto, FunctionProto, NodeProto, TypeProto

_TargetOpType = ""


def test_layernorm():
    def _extract_value_info(input: Union[List[Any], np.ndarray, None], name: Text, type_proto: Optional[TypeProto] = None) -> onnx.ValueInfoProto:
        if type_proto is None:
            if input is None:
                raise NotImplementedError(
                    "_extract_value_info: both input and type_proto arguments cannot be None.")
            elif isinstance(input, list):
                elem_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[input[0].dtype]
                shape = None
                tensor_type_proto = onnx.helper.make_tensor_type_proto(
                    elem_type, shape)
                type_proto = onnx.helper.make_sequence_type_proto(
                    tensor_type_proto)
            else:
                elem_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[input.dtype]
                shape = input.shape
                type_proto = onnx.helper.make_tensor_type_proto(
                    elem_type, shape)

        return onnx.helper.make_value_info(name, type_proto)

    # Layer normalization's reference implementation
    def _layer_normalization(X, W, B, axis=-1, epsilon=1e-5):  # type: ignore
        X_shape = X.shape
        X_rank = len(X_shape)
        if axis < 0:
            # If axis = -1 and rank of X is 4,
            # the axis is changed to -1 + 4 = 3,
            # which means the last axis.
            axis = axis + X_rank
        unsqueezed_rank = X_rank - axis
        reduction_shape = X_shape[0:axis] + (1,) * unsqueezed_rank

        # Parameter used to convert N-D tensor layer
        # normalization to equivalent 2-D matirx operations.
        row_number = 1
        col_number = 1
        for i in range(X_rank):
            if i < axis:
                row_number *= X_shape[i]
            else:
                col_number *= X_shape[i]

        # After reshaping input tensor X into a matrix,
        # layer normalization is equivalent to conducting
        # standardization on each column vector (s.t. each
        # column has zero mean and unit variance).
        x_mat = np.reshape(X, (row_number, col_number))
        # This computes mean for every x_mat's column.
        x_mean = np.sum(x_mat, axis=1, keepdims=True) / col_number
        x_diff = x_mat - x_mean
        x_squared_diff = x_diff * x_diff
        # This computes variance for every x_mat's column.
        variance = np.sum(x_squared_diff, axis=1, keepdims=True) / col_number
        variance_eps = variance + epsilon
        std_dev = np.sqrt(variance_eps)
        inv_std_dev = np.reciprocal(std_dev)
        # Standardization step. y_mat is zero-mean and unit-variance.
        y_mat = x_diff * inv_std_dev
        # Apply affine transform on normalization outcome.
        # W is linear coefficient while B is bias.
        Y = np.reshape(y_mat, X_shape) * W + B
        # Matrix-level operations' outputs should be reshaped
        # to compensate the initial tensor-to-matrix reshape.
        X_mean = np.reshape(x_mean, reduction_shape)
        X_inv_std_dev = np.reshape(inv_std_dev, reduction_shape)

        return Y, X_mean, X_inv_std_dev

    def calculate_normalized_shape(X_shape, axis):  # type: ignore
        X_rank = len(X_shape)
        if axis < 0:
            axis = axis + X_rank
        return X_shape[axis:]

    X = np.random.randn(1, 3, 224, 224).astype(np.float32)
    axis = 1
    normalized_shape = calculate_normalized_shape(X.shape, axis)
    W = np.random.randn(*normalized_shape).astype(np.float32)
    B = np.random.randn(*normalized_shape).astype(np.float32)
    Y, mean, inv_std_dev = _layer_normalization(X, W, B, axis)
    Y = Y.astype(np.float32)
    inputs = [X, W, B]
    outputs = [Y]

    x = torch.from_numpy(X)
    layer_norm = nn.LayerNorm([i for i in x.shape[axis:]])
    for name, param in layer_norm.named_parameters():
        if 'weight' in name:
            layer_norm.weight.data = torch.from_numpy(W)
        if 'bias' in name:
            layer_norm.bias.data = torch.from_numpy(B)
    y = layer_norm(x)
    y = y.detach().numpy()

    node = onnx.helper.make_node(
        'LayerNormalization',
        axis=axis,
        inputs=['X', 'W', 'B'],
        outputs=['Y']
    )

    if _TargetOpType and node.op_type != _TargetOpType:
        return
    present_inputs = [x for x in node.input if (x != '')]
    present_outputs = [x for x in node.output if (x != '')]
    input_type_protos = [None] * len(inputs)
    output_type_protos = [None] * len(outputs)
    inputs_vi = [_extract_value_info(arr, arr_name, input_type)
                 for arr, arr_name, input_type in zip(inputs, present_inputs, input_type_protos)]
    outputs_vi = [_extract_value_info(arr, arr_name, output_type)
                  for arr, arr_name, output_type in zip(outputs, present_outputs, output_type_protos)]

    graph = onnx.helper.make_graph(
        nodes=[node],
        name='test_layer_normalization_4d_axis',
        inputs=inputs_vi,
        outputs=outputs_vi)
    model = onnx.helper.make_model(graph)
    onnx.save_model(model, 'layernorm.onnx')

    sess = rt.InferenceSession('layernorm.onnx')
    x_name = sess.get_inputs()[0].name
    w_name = sess.get_inputs()[1].name
    b_name = sess.get_inputs()[2].name
    y_name = sess.get_outputs()[0].name

    pred_onx = sess.run([y_name], {x_name: X, w_name: W, b_name: B})[0]
    error = np.sum(np.abs(y - pred_onx)) / np.sum(np.abs(y))
    print(error)

    # print(model)

import os
from typing import Any, List, Optional, Sequence, Text, Union

import numpy as np
import onnx
import onnxruntime as rt
import torch
import torch.nn as nn
from onnx.onnx_pb import AttributeProto, FunctionProto, NodeProto, TypeProto

_TargetOpType = ""


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.slice_num = 4
        self.weight_ih = nn.Parameter(
            torch.randn(self.slice_num * hidden_size, input_size)
        )
        self.weight_hh = nn.Parameter(
            torch.randn(self.slice_num * hidden_size, hidden_size)
        )
        self.bias_ih = nn.Parameter(torch.randn(self.slice_num * hidden_size))
        self.bias_hh = nn.Parameter(torch.randn(self.slice_num * hidden_size))

        self.fc1 = nn.Linear(input_size, self.slice_num * hidden_size)
        self.fc2 = nn.Linear(hidden_size, self.slice_num * hidden_size)

    def forward(self, input, states, param=None):
        if param:
            self.fc1.weight = self.weight_ih = param[0]
            self.fc2.weight = self.weight_hh = param[1]
            self.fc1.bias = self.bias_ih = param[2]
            self.fc2.bias = self.bias_hh = param[3]

        hx, cx = states
        gates = self.fc1(input) + self.fc2(hx)
        # gates = (
        #     torch.mm(input, self.weight_ih.t())
        #     + self.bias_ih
        #     + torch.mm(hx, self.weight_hh.t())
        #     + self.bias_hh
        # )
        ingate, forgetgate, cellgate, outgate = torch.chunk(
            gates, self.slice_num, dim=1
        )
        # ingate, forgetgate, cellgate, outgate = torch.split(
        #     gates, [self.hidden_size for _ in range(self.slice_num)], dim=1
        # )

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm_cell = LSTMCell(
            input_size=self.input_size, hidden_size=self.hidden_size
        )

    def forward(self, inputs, states, params=None):
        time_step = inputs.shape[0]
        hx_outs = []
        cx_outs = []

        for layer_id in range(self.num_layers):
            if params:
                param = params[layer_id]
            else:
                param = None
            output = []
            hy, cy = states[0][layer_id], states[1][layer_id]
            for time in range(time_step):
                input = inputs[time]
                hy, (hy, cy) = self.lstm_cell(input, (hy, cy), param=param)
                output.append(hy.unsqueeze(dim=0))
            hx_outs.append(hy.unsqueeze(dim=0))
            cx_outs.append(cy.unsqueeze(dim=0))
            outputs = torch.concat(output, dim=0)
            inputs = outputs

        hx_outs = torch.concat(hx_outs, dim=0)
        cx_outs = torch.concat(cx_outs, dim=0)

        return outputs, (hx_outs, cx_outs)


def test_lstm(batch=1, time_step=9, input_size=64, hidden_size=32, bidirectional = True):

    lstm_torch = nn.LSTM(input_size, hidden_size, bidirectional=bidirectional)
    xi = torch.randn(
        time_step, batch, input_size
    )
    h_len = 2 if bidirectional else 1
    h0 = torch.randn(h_len, batch, hidden_size)
    c0 = torch.randn(h_len, batch, hidden_size)

    xo_t, (hn_t, cn_t) = lstm_torch(xi, (h0, c0))
    
    if not os.path.exists('{}/work_dir'.format('./')):
        os.makedirs('{}/work_dir'.format('./'))
    torch.onnx.export(lstm_torch,
                      (xi, (h0, c0)),
                      '{}/work_dir/lstm.onnx'.format('./'),
                      export_params=True,
                      opset_version=14,
                      do_constant_folding=True,
                      input_names=['input1', 'input2'],
                      output_names=['output1', 'output2'])

    sess = rt.InferenceSession('{}/work_dir/lstm.onnx'.format('./'))
    xi_name = sess.get_inputs()[0].name
    h0_name = sess.get_inputs()[1].name
    c0_name = sess.get_inputs()[2].name
    xo_name = sess.get_outputs()[0].name
    hn_name = sess.get_outputs()[1].name
    cn_name = sess.get_outputs()[2].name

    pred_onnx = sess.run([xo_name, hn_name, cn_name], {
        xi_name: xi.numpy(),
        h0_name: h0.numpy(),
        c0_name: c0.numpy()})
    print("=> lstm_torch: ", xo_t.mean().item(),
          hn_t.mean().item(), cn_t.mean().item())
    print("=> lstm_onnx: ", pred_onnx[0].mean(), pred_onnx[1].mean(), pred_onnx[2].mean())
    xo_onnx = pred_onnx[0]
    hn_onnx = pred_onnx[1]
    cn_onnx = pred_onnx[2]
    xo = xo.detach().numpy()
    hn = hn.detach().numpy()
    cn = cn.detach().numpy()
    W = lstm.state_dict()['lstm_cell.fc1.weight'].unsqueeze(0)
    Wb1 = lstm.state_dict()['lstm_cell.fc1.bias'].unsqueeze(0)
    R = lstm.state_dict()['lstm_cell.fc2.weight'].unsqueeze(0)
    Wb2 = lstm.state_dict()['lstm_cell.fc2.bias'].unsqueeze(0)
    Wb = torch.concat([Wb1, Wb2], dim=1)
    np.save('x.npy', xi.numpy())
    np.save('h0.npy', h0.numpy())
    np.save('c0.npy', c0.numpy())
    np.save('ot.npy', xo)
    np.save('hn.npy', hn)
    np.save('cn.npy', cn)
    np.save('w.npy', W.numpy())
    np.save('r.npy', R.numpy())
    np.save('b.npy', Wb.numpy())
    error_xo = np.sum(np.abs(xo_onnx - xo)) / np.sum(np.abs(xo))
    error_hn = np.sum(np.abs(hn_onnx - hn)) / np.sum(np.abs(hn))
    error_cn = np.sum(np.abs(cn_onnx - cn)) / np.sum(np.abs(cn))
    print('=> error: ', error_xo, error_hn, error_cn)


if __name__ == '__main__':
    # test_layernorm()
    test_lstm()
