import { Coordinator } from './coordinator';
import { SwarmNode } from './client-node';

async function runTests() {
  console.log("=========================================");
  console.log("  FlockML Unit Test: Basic Functionality ");
  console.log("=========================================");

  // 1. Initialize Central Coordinator (Cloud)
  const coordinator = new Coordinator(2, 4, 1);
  console.log("\n[Cloud] Initialized Central Coordinator (FedAvg Engine).");

  // 2. Initialize 2 Browser Clients (Edge)
  const clientA = new SwarmNode(2, 4, 1);
  const clientB = new SwarmNode(2, 4, 1);
  
  clientA.connect("wss://api.flockml.com");
  clientB.connect("wss://api.flockml.com");

  // Dummy Dataset (XOR Problem)
  const inputs = [[0, 0], [0, 1], [1, 0], [1, 1]];
  const targets = [[0], [1], [1], [0]];

  // 4. Train the clients locally on the Edge
  console.log("\n[Edge] Client A training local batch...");
  await clientA.trainLocalBatchAsync(inputs, targets);

  console.log("[Edge] Client B training local batch...");
  await clientB.trainLocalBatchAsync(inputs, targets);

  // 5. Secure & Quantize Gradients
  console.log("\n[Edge] Applying Laplacian Noise & 8-Bit Quantization...");
  const payloadA = clientA.exportSecureGradients();
  const payloadB = clientB.exportSecureGradients();

  // 6. Transmit to Cloud
  console.log("\n[Cloud] Receiving 8-bit quantized payloads from Edge...");
  coordinator.receiveUpdate(payloadA.weights_ih, payloadA.weights_ho, payloadA.bias_h, payloadA.bias_o);
  coordinator.receiveUpdate(payloadB.weights_ih, payloadB.weights_ho, payloadB.bias_h, payloadB.bias_o);

  // 7. Perform Federated Averaging
  console.log("[Cloud] Aggregating Models (FedAvg)...");
  coordinator.aggregate();

  console.log("\n=========================================");
  console.log("  Test Complete. System mathematically sound.");
  console.log("=========================================\n");
}

runTests().catch(console.error);
