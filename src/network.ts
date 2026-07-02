import { Matrix } from './matrix';
import { sigmoid, dsigmoid } from './activations';

/**
 * The Core Neural Network Engine.
 * 
 * This class orchestrates the Forward Pass and Backpropagation algorithms.
 * It is designed to be lightweight enough to run non-blocking in a Web Worker.
 */
export class NeuralNetwork {
  inputNodes: number;
  hiddenNodes: number;
  outputNodes: number;
  
  weights_ih: Matrix; // Weights between Input and Hidden layer
  weights_ho: Matrix; // Weights between Hidden and Output layer
  
  bias_h: Matrix; // Bias for Hidden layer
  bias_o: Matrix; // Bias for Output layer
  
  learningRate: number;
  
  activationFunc: (x: number) => number;
  dActivationFunc: (y: number) => number;

  constructor(inputNodes: number, hiddenNodes: number, outputNodes: number) {
    this.inputNodes = inputNodes;
    this.hiddenNodes = hiddenNodes;
    this.outputNodes = outputNodes;

    // Initialize weights
    this.weights_ih = new Matrix(this.hiddenNodes, this.inputNodes);
    this.weights_ho = new Matrix(this.outputNodes, this.hiddenNodes);
    this.weights_ih.randomize();
    this.weights_ho.randomize();

    // Initialize biases
    this.bias_h = new Matrix(this.hiddenNodes, 1);
    this.bias_o = new Matrix(this.outputNodes, 1);
    this.bias_h.randomize();
    this.bias_o.randomize();

    this.learningRate = 0.1;
    
    // Default to Sigmoid activation (can be changed to ReLU or Tanh later)
    this.activationFunc = sigmoid;
    this.dActivationFunc = dsigmoid;
  }

  /**
   * The Forward Pass.
   * Feeds the input data through the network to generate a prediction.
   * Math: Output = Activation(Weights * Input + Bias)
   */
  predict(inputArray: number[]): number[] {
    // 1. Convert input array to Matrix
    const inputs = Matrix.from1DArray(inputArray);

    // 2. Calculate Hidden Layer signals
    const hidden = Matrix.dot(this.weights_ih, inputs);
    hidden.add(this.bias_h);
    hidden.map(this.activationFunc); // Apply non-linearity

    // 3. Calculate Output Layer signals
    const outputs = Matrix.dot(this.weights_ho, hidden);
    outputs.add(this.bias_o);
    outputs.map(this.activationFunc); // Apply non-linearity

    // Return the prediction array
    return outputs.toArray();
  }

  /**
   * The Backpropagation Algorithm (Training).
   * 1. Performs a Forward Pass.
   * 2. Calculates the Error (Target - Output).
   * 3. Calculates Gradients using the Chain Rule (Calculus).
   * 4. Updates weights and biases.
   */
  train(inputArray: number[], targetArray: number[]): void {
    // --- FORWARD PASS (Same as predict) ---
    const inputs = Matrix.from1DArray(inputArray);
    
    const hidden = Matrix.dot(this.weights_ih, inputs);
    hidden.add(this.bias_h);
    hidden.map(this.activationFunc);
    
    const outputs = Matrix.dot(this.weights_ho, hidden);
    outputs.add(this.bias_o);
    outputs.map(this.activationFunc);

    // --- ERROR CALCULATION ---
    const targets = Matrix.from1DArray(targetArray);
    // Output Error = Targets - Outputs
    const outputErrors = Matrix.subtract(targets, outputs);

    // Hidden Error = Transpose(Weights_HO) * OutputErrors
    const weights_ho_t = Matrix.transpose(this.weights_ho);
    const hiddenErrors = Matrix.dot(weights_ho_t, outputErrors);

    // --- CALCULATE GRADIENTS (The Calculus) ---
    
    // Gradients for Output Layer
    // gradient = learningRate * error * derivative(output) * transpose(hidden)
    const gradients = Matrix.map(outputs, this.dActivationFunc);
    const gradientsMult = Matrix.multiply(gradients, outputErrors);
    gradientsMult.multiply(this.learningRate);

    // Calculate weight deltas for Output Layer
    const hidden_T = Matrix.transpose(hidden);
    const weight_ho_deltas = Matrix.dot(gradientsMult, hidden_T);

    // Adjust weights and biases (Output Layer)
    this.weights_ho.add(weight_ho_deltas);
    this.bias_o.add(gradientsMult);

    // Gradients for Hidden Layer
    const hiddenGradients = Matrix.map(hidden, this.dActivationFunc);
    const hiddenGradientsMult = Matrix.multiply(hiddenGradients, hiddenErrors);
    hiddenGradientsMult.multiply(this.learningRate);

    // Calculate weight deltas for Hidden Layer
    const inputs_T = Matrix.transpose(inputs);
    const weight_ih_deltas = Matrix.dot(hiddenGradientsMult, inputs_T);

    // Adjust weights and biases (Hidden Layer)
    this.weights_ih.add(weight_ih_deltas);
    this.bias_h.add(hiddenGradientsMult);
  }

  /**
   * Serializes the model weights and biases for network transmission.
   * Used when sending the locally trained math back to the central server.
   */
  serialize(): string {
    return JSON.stringify({
      weights_ih: this.weights_ih.data,
      weights_ho: this.weights_ho.data,
      bias_h: this.bias_h.data,
      bias_o: this.bias_o.data
    });
  }

  /**
   * Merges incoming weights (used by the FedAvg algorithm).
   */
  loadWeights(data: { weights_ih: number[][]; weights_ho: number[][]; bias_h: number[][]; bias_o: number[][] }): void {
    this.weights_ih.data = data.weights_ih;
    this.weights_ho.data = data.weights_ho;
    this.bias_h.data = data.bias_h;
    this.bias_o.data = data.bias_o;
  }
}
