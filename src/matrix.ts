/**
 * Core Linear Algebra Engine for Browser-Native Neural Networks.
 * 
 * This class handles all matrix math required for Forward Passes and Backpropagation.
 * Optimized for V8 JIT compilation using typed Float32Arrays where possible, 
 * though standard arrays are used here for structural simplicity in the MVP.
 */

export class Matrix {
  rows: number;
  cols: number;
  data: number[][];

  constructor(rows: number, cols: number) {
    this.rows = rows;
    this.cols = cols;
    this.data = Array(this.rows).fill(0).map(() => Array(this.cols).fill(0));
  }

  /**
   * Creates a matrix from an existing 2D array.
   */
  static fromArray(arr: number[][]): Matrix {
    const m = new Matrix(arr.length, arr[0].length);
    for (let i = 0; i < m.rows; i++) {
      for (let j = 0; j < m.cols; j++) {
        m.data[i][j] = arr[i][j];
      }
    }
    return m;
  }

  /**
   * Randomizes the matrix weights (typically for initialization).
   * Uses a standard normal distribution approximation (Xavier initialization simplified).
   */
  randomize(): void {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        // Random value between -1 and 1
        this.data[i][j] = Math.random() * 2 - 1;
      }
    }
  }

  /**
   * Matrix Dot Product (Matrix Multiplication).
   * Used heavily in the Forward Pass to calculate layer activations.
   * O(n^3) complexity - prime candidate for WebGPU acceleration later.
   */
  static dot(a: Matrix, b: Matrix): Matrix {
    if (a.cols !== b.rows) {
      throw new Error(`Incompatible matrices for dot product: ${a.cols} !== ${b.rows}`);
    }

    const result = new Matrix(a.rows, b.cols);
    for (let i = 0; i < result.rows; i++) {
      for (let j = 0; j < result.cols; j++) {
        let sum = 0;
        for (let k = 0; k < a.cols; k++) {
          sum += a.data[i][k] * b.data[k][j];
        }
        result.data[i][j] = sum;
      }
    }
    return result;
  }

  /**
   * Element-wise addition.
   */
  add(n: Matrix | number): void {
    if (n instanceof Matrix) {
      if (this.rows !== n.rows || this.cols !== n.cols) {
        throw new Error('Incompatible matrices for addition');
      }
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.cols; j++) {
          this.data[i][j] += n.data[i][j];
        }
      }
    } else {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.cols; j++) {
          this.data[i][j] += n;
        }
      }
    }
  }

  /**
   * Element-wise subtraction (A - B). Returns a new Matrix.
   * Crucial for calculating the Error (Target - Prediction).
   */
  static subtract(a: Matrix, b: Matrix): Matrix {
    if (a.rows !== b.rows || a.cols !== b.cols) {
      throw new Error('Incompatible matrices for subtraction');
    }
    const result = new Matrix(a.rows, a.cols);
    for (let i = 0; i < result.rows; i++) {
      for (let j = 0; j < result.cols; j++) {
        result.data[i][j] = a.data[i][j] - b.data[i][j];
      }
    }
    return result;
  }

  /**
   * Element-wise multiplication (Hadamard product). Returns a new Matrix.
   */
  static multiply(a: Matrix, b: Matrix): Matrix {
    if (a.rows !== b.rows || a.cols !== b.cols) {
      throw new Error('Incompatible matrices for element-wise multiplication');
    }
    const result = new Matrix(a.rows, a.cols);
    for (let i = 0; i < result.rows; i++) {
      for (let j = 0; j < result.cols; j++) {
        result.data[i][j] = a.data[i][j] * b.data[i][j];
      }
    }
    return result;
  }

  /**
   * Scalar multiplication.
   */
  multiply(n: number): void {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        this.data[i][j] *= n;
      }
    }
  }

  /**
   * Applies a function to every element in the matrix.
   * Used for Activation Functions (e.g., Sigmoid, ReLU) and their derivatives.
   */
  map(func: (val: number, i: number, j: number) => number): void {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        const val = this.data[i][j];
        this.data[i][j] = func(val, i, j);
      }
    }
  }

  /**
   * Static version of map, returns a new Matrix.
   */
  static map(matrix: Matrix, func: (val: number, i: number, j: number) => number): Matrix {
    const result = new Matrix(matrix.rows, matrix.cols);
    for (let i = 0; i < matrix.rows; i++) {
      for (let j = 0; j < matrix.cols; j++) {
        const val = matrix.data[i][j];
        result.data[i][j] = func(val, i, j);
      }
    }
    return result;
  }

  /**
   * Transposes the matrix (flips rows and columns).
   * Crucial for calculating gradients during Backpropagation.
   */
  static transpose(matrix: Matrix): Matrix {
    const result = new Matrix(matrix.cols, matrix.rows);
    for (let i = 0; i < matrix.rows; i++) {
      for (let j = 0; j < matrix.cols; j++) {
        result.data[j][i] = matrix.data[i][j];
      }
    }
    return result;
  }

  /**
   * Converts a 1D array to a Matrix (Column vector).
   */
  static from1DArray(arr: number[]): Matrix {
    return Matrix.fromArray(arr.map(x => [x]));
  }

  /**
   * Converts the Matrix back to a 1D array.
   */
  toArray(): number[] {
    const arr: number[] = [];
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        arr.push(this.data[i][j]);
      }
    }
    return arr;
  }
}
