/**
 * Activation Functions & Calculus for the Neural Network.
 * 
 * This module contains the non-linear activation functions used in the Forward Pass,
 * and their corresponding mathematical derivatives used in Backpropagation (Chain Rule)
 * to calculate the gradients for weight adjustments.
 */

// Sigmoid function: squashes numbers between 0 and 1.
// Used for probability outputs.
export function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

// The derivative of the Sigmoid function.
// Crucial for calculating the gradient during backpropagation.
// Math: f'(x) = f(x) * (1 - f(x))
export function dsigmoid(y: number): number {
  // Note: y is already the sigmoid output here.
  return y * (1 - y);
}

// ReLU (Rectified Linear Unit) function.
// Allows the network to learn non-linear patterns faster without the vanishing gradient problem.
export function relu(x: number): number {
  return Math.max(0, x);
}

// Derivative of ReLU.
// Math: 1 if x > 0 else 0
export function drelu(y: number): number {
  return y > 0 ? 1 : 0;
}

// Tanh (Hyperbolic Tangent) function.
// Squashes numbers between -1 and 1. Good for hidden layers.
export function tanh(x: number): number {
  return Math.tanh(x);
}

// Derivative of Tanh.
// Math: 1 - f(x)^2
export function dtanh(y: number): number {
  return 1 - (y * y);
}
