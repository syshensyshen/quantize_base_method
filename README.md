<!--
 * @Author: WANG CHENG
 * @Date: 2024-04-24 20:36:59
 * @LastEditTime: 2024-04-24 20:58:00
-->
# Neural Network Quantization Test Suite

This Python package is a collection of scripts for testing and evaluating quantization algorithms on various neural network operations. It is designed to ensure the accuracy and performance of quantized models.

## Key Features

- **Quantization Algorithms**: Simulation of quantization for different operators with a focus on:
  - Symmetric and asymmetric quantization for convolution operations.
  - Fusing zero points into biases for simulation purposes.

- **Recurrent Operations**: Integer inference for RNN operations, potentially utilizing look-up tables (LUTs) for activation functions within RNN cells.

- **Per-Channel Convolution**: Implementation of a simulator for per-channel convolution operations.

- **Activation Functions**: Simulation of activation functions using LUTs for efficiency.

- **Multiplication Operations**: Integer inference for multiplication operations without the need for int32 transfers.

- **Pooling Operations**:
  - Max Pooling: Splitting the pooling operation into horizontal and vertical passes for potentially better performance on some hardware.
  - Average Pooling: Similar splitting approach, with considerations for using int32 for the divisor but int16 for the kernel sum.

## Usage

Each script is crafted for specific quantization tests. Users can execute these to assess the quantization effects on different parts of a neural network.

## Contribution

Contributions to expand the test coverage and improve quantization methods are encouraged. Please adhere to the contribution guidelines.
