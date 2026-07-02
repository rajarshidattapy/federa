import { Matrix } from './matrix';

/**
 * Quantization Engine.
 * 
 * Standard neural networks use 32-bit floating point numbers (Float32).
 * This is too heavy for browsers to send over WebSockets rapidly.
 * We dynamically compress the Float32 matrices down to 8-bit integers (Int8)
 * for a 4x reduction in payload size before network transmission.
 */

export interface QuantizedPayload {
  data: number[]; // In a real implementation this would be Int8Array, but standard arrays are easier for JSON serialization MVP
  min: number;
  max: number;
  rows: number;
  cols: number;
}

export class Quantizer {
  
  /**
   * Compresses a Float32 Matrix into an 8-bit integer payload.
   * Maps the range [min, max] to [-127, 127].
   */
  static quantize(matrix: Matrix): QuantizedPayload {
    let min = Infinity;
    let max = -Infinity;

    // 1. Find the min and max values in the matrix
    for (let i = 0; i < matrix.rows; i++) {
      for (let j = 0; j < matrix.cols; j++) {
        const val = matrix.data[i][j];
        if (val < min) min = val;
        if (val > max) max = val;
      }
    }

    // 2. Calculate the scaling factor
    // We map to [-127, 127] for Int8 representation
    const range = Math.max(Math.abs(min), Math.abs(max));
    const scale = range === 0 ? 1 : 127 / range;

    const quantizedData: number[] = [];

    // 3. Scale and round to nearest integer
    for (let i = 0; i < matrix.rows; i++) {
      for (let j = 0; j < matrix.cols; j++) {
        const val = matrix.data[i][j];
        // Round to integer between -127 and 127
        const qVal = Math.round(val * scale);
        quantizedData.push(qVal);
      }
    }

    return {
      data: quantizedData,
      min: -range, // Symmetric range
      max: range,
      rows: matrix.rows,
      cols: matrix.cols
    };
  }

  /**
   * Decompresses an 8-bit integer payload back into a Float32 Matrix.
   * Executed primarily by the Coordinator server, but included here for local testing.
   */
  static dequantize(payload: QuantizedPayload): Matrix {
    const matrix = new Matrix(payload.rows, payload.cols);
    
    const range = Math.max(Math.abs(payload.min), Math.abs(payload.max));
    const scale = range === 0 ? 1 : range / 127;

    let index = 0;
    for (let i = 0; i < payload.rows; i++) {
      for (let j = 0; j < payload.cols; j++) {
        const qVal = payload.data[index++];
        // Convert back to float
        matrix.data[i][j] = qVal * scale;
      }
    }

    return matrix;
  }
}
