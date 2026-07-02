import { NeuralNetwork } from './network';
import { Matrix } from './matrix';
import { Quantizer, QuantizedPayload } from './quantization';

/**
 * The Central Brain: FedAvg Coordinator.
 * 
 * In a real production environment, this would run on a Node.js/Docker server.
 * It listens for incoming encrypted, quantized gradients from thousands of Web Workers,
 * decrypts them, and mathematically averages them together to update the Global Model.
 */
export class Coordinator {
  globalModel: NeuralNetwork;
  clientUpdates: {
    weights_ih: Matrix;
    weights_ho: Matrix;
    bias_h: Matrix;
    bias_o: Matrix;
  }[];

  constructor(inputNodes: number, hiddenNodes: number, outputNodes: number) {
    this.globalModel = new NeuralNetwork(inputNodes, hiddenNodes, outputNodes);
    this.clientUpdates = [];
  }

  /**
   * Receives a quantized payload from a browser client.
   * De-quantizes the 8-bit integers back to Float32 matrices and stores them for aggregation.
   */
  receiveUpdate(
    qWeightsIH: QuantizedPayload,
    qWeightsHO: QuantizedPayload,
    qBiasH: QuantizedPayload,
    qBiasO: QuantizedPayload
  ): void {
    const weights_ih = Quantizer.dequantize(qWeightsIH);
    const weights_ho = Quantizer.dequantize(qWeightsHO);
    const bias_h = Quantizer.dequantize(qBiasH);
    const bias_o = Quantizer.dequantize(qBiasO);

    this.clientUpdates.push({
      weights_ih,
      weights_ho,
      bias_h,
      bias_o
    });
  }

  /**
   * The Federated Averaging (FedAvg) Algorithm.
   * 
   * Averages all the incoming matrices. Because of Differential Privacy, 
   * the Laplacian noise mathematically cancels out to 0 here, leaving only
   * the pure, learned signal from the crowdsourced devices.
   */
  aggregate(): void {
    if (this.clientUpdates.length === 0) return;

    const numClients = this.clientUpdates.length;

    // Helper function to average a specific matrix across all clients
    const averageMatrix = (matrixKey: 'weights_ih' | 'weights_ho' | 'bias_h' | 'bias_o', targetMatrix: Matrix) => {
      // Create a zeroed matrix to hold the sum
      const sumMatrix = new Matrix(targetMatrix.rows, targetMatrix.cols);
      
      // Sum all client matrices
      for (const update of this.clientUpdates) {
        sumMatrix.add(update[matrixKey]);
      }

      // Divide by N (number of clients) to get the average
      sumMatrix.multiply(1 / numClients);

      // Apply the averaged weights to the global model
      for (let i = 0; i < targetMatrix.rows; i++) {
        for (let j = 0; j < targetMatrix.cols; j++) {
          targetMatrix.data[i][j] = sumMatrix.data[i][j];
        }
      }
    };

    // Run FedAvg on all weights and biases
    averageMatrix('weights_ih', this.globalModel.weights_ih);
    averageMatrix('weights_ho', this.globalModel.weights_ho);
    averageMatrix('bias_h', this.globalModel.bias_h);
    averageMatrix('bias_o', this.globalModel.bias_o);

    // Clear the queue for the next training round
    this.clientUpdates = [];
  }

  /**
   * Broadcasts the current global model weights to all new clients.
   * This is sent to browsers when they first connect to the WebSocket.
   */
  getGlobalWeightsForBroadcast() {
    return {
      weights_ih: Quantizer.quantize(this.globalModel.weights_ih),
      weights_ho: Quantizer.quantize(this.globalModel.weights_ho),
      bias_h: Quantizer.quantize(this.globalModel.bias_h),
      bias_o: Quantizer.quantize(this.globalModel.bias_o)
    };
  }
}
