import { Matrix } from './matrix';

/**
 * Differential Privacy Engine (Cryptographic Noise).
 * 
 * To guarantee that the central server cannot reverse-engineer a user's private data
 * from their uploaded neural network gradients, we inject Laplacian noise into the matrices.
 * 
 * According to Differential Privacy theorems, if enough users submit noisy gradients,
 * the random noise perfectly cancels out during Federated Averaging, leaving only the 
 * true signal (the learned patterns) intact.
 */

export class DifferentialPrivacy {
  
  /**
   * Generates a random number drawn from a Laplace distribution.
   * Laplace(mu=0, b=scale)
   * @param scale The 'b' parameter controlling the spread (variance) of the noise. Higher = more privacy, less accuracy.
   */
  static generateLaplaceNoise(scale: number): number {
    // Generate a uniform random variable u in the range [-0.5, 0.5]
    const u = Math.random() - 0.5;
    
    // Inverse cumulative distribution function for Laplace
    // x = mu - b * sgn(u) * ln(1 - 2|u|)
    // Since mu = 0:
    const sign = u < 0 ? -1 : 1;
    return -scale * sign * Math.log(1 - 2 * Math.abs(u));
  }

  /**
   * Mutates a Matrix by injecting Laplacian noise into every element.
   * This is called immediately before Quantization and Network Transmission.
   * 
   * @param matrix The gradients/weights to anonymize.
   * @param epsilon The privacy budget (e.g., 0.1 for high privacy, 10 for low privacy).
   * @param sensitivity The maximum possible change a single data point can cause (usually clipped in DL).
   */
  static applyNoise(matrix: Matrix, epsilon: number = 0.5, sensitivity: number = 1.0): void {
    // scale (b) = sensitivity / epsilon
    const scale = sensitivity / epsilon;

    for (let i = 0; i < matrix.rows; i++) {
      for (let j = 0; j < matrix.cols; j++) {
        const noise = this.generateLaplaceNoise(scale);
        matrix.data[i][j] += noise;
      }
    }
  }
}
