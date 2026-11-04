/**
 * @file labs/mean-square-boundary/mlp-network.ts
 * @description Real 2-layer non-linear MLP (`f_Theta(x) = W_2 tanh(W_1 x + b_1) + b_2`, using `fdlibmTanh`) with exact analytical backpropagation (`createTwoLayerMlpDatasetAndOps`) for the `P = 33` network of Phases 4–6, also the small test case of the Phase 2 tests.
 * @cold
 */
import { fdlibmTanh } from '../../src/math/index.ts';
import { createXorshift128Plus } from './prng-xorshift128p.ts';

function populateMlpCache0(
  theta0: Float64Array,
  xData: Float64Array,
  numSamples: number,
  dIn: number,
  dHidden: number,
  numParams: number,
  snapTheta0: Float64Array,
  actTable0: Float64Array,
  dtanhTable0: Float64Array,
  predTable0: Float64Array,
): void {
  snapTheta0.set(theta0);
  const b1Base = dIn * dHidden;
  const w2Base = b1Base + dHidden;
  for (let n = 0; n < numSamples; n = (n + 1) | 0) {
    const xOff = Math.imul(n, dIn);
    const hOff = Math.imul(n, dHidden);
    let pred = theta0[numParams - 1]!;
    for (let h = 0; h < dHidden; h = (h + 1) | 0) {
      const w1Row = Math.imul(h, dIn);
      let z = theta0[b1Base + h]!;
      for (let j = 0; j < dIn; j = (j + 1) | 0) z += theta0[w1Row + j]! * xData[xOff + j]!;
      const a = fdlibmTanh(z);
      actTable0[(hOff + h) | 0] = a;
      dtanhTable0[(hOff + h) | 0] = 1.0 - a * a;
      pred += theta0[w2Base + h]! * a;
    }
    predTable0[n] = pred;
  }
}

export function createTwoLayerMlpDatasetAndOps(seedLo = 0x67200001, seedHi = 0x9e3779b9): {
  readonly dIn: number;
  readonly dHidden: number;
  readonly numParams: number;
  readonly numSamples: number;
  readonly xData: Float64Array;
  readonly yData: Float64Array;
  readonly checkpointTheta0: Float64Array;
  readonly checkpointLoss0: number;
  evalFullLoss(theta: Float64Array): number;
  computeFullGradient(theta: Float64Array, outGrad: Float64Array): number;
  computeMinibatchGradient(
    theta: Float64Array,
    batchIndices: Int32Array,
    batchSize: number,
    outGrad: Float64Array,
  ): number;
  computeSampleOutputGradient(theta: Float64Array, sampleIdx: number, outGrad: Float64Array): void;
} {
  const dIn = 6;
  const dHidden = 4;
  // Layout of flat theta (length 33):
  // [0 .. 23]: W1 (4 x 6)
  // [24 .. 27]: b1 (4)
  // [28 .. 31]: W2 (1 x 4)
  // [32]: b2 (1)
  const numParams = dIn * dHidden + dHidden + dHidden + 1;
  const numSamples = 64;

  const prng = createXorshift128Plus(seedLo, seedHi);
  const teacherTheta = new Float64Array(numParams);
  for (let i = 0; i < dIn * dHidden; i = (i + 1) | 0) {
    teacherTheta[i] = prng.nextGaussian() * 0.85;
  }
  for (let h = 0; h < dHidden; h = (h + 1) | 0) {
    teacherTheta[dIn * dHidden + h] = prng.nextGaussian() * 0.25;
    teacherTheta[dIn * dHidden + dHidden + h] = (h % 2 === 0 ? 1.35 : -1.25) * (1.0 + 0.2 * h);
  }
  teacherTheta[numParams - 1] = 0.15;

  const xData = new Float64Array(numSamples * dIn);
  const yData = new Float64Array(numSamples);
  const anisoScale = [2.4, 1.3, 0.8, 0.5, 0.35, 0.25] as const;

  function forwardSingle(theta: Float64Array, xOffset: number): number {
    const b1Base = dIn * dHidden;
    const w2Base = b1Base + dHidden;
    let pred = theta[numParams - 1]!;
    for (let h = 0; h < dHidden; h = (h + 1) | 0) {
      const w1Row = Math.imul(h, dIn);
      let z = theta[b1Base + h]!;
      for (let j = 0; j < dIn; j = (j + 1) | 0) {
        z += theta[w1Row + j]! * xData[xOffset + j]!;
      }
      const a = fdlibmTanh(z);
      pred += theta[w2Base + h]! * a;
    }
    return pred;
  }

  for (let n = 0; n < numSamples; n = (n + 1) | 0) {
    const xOff = Math.imul(n, dIn);
    for (let j = 0; j < dIn; j = (j + 1) | 0) {
      xData[xOff + j] = prng.nextGaussian() * anisoScale[j]!;
    }
    yData[n] = forwardSingle(teacherTheta, xOff) + prng.nextGaussian() * 0.08;
  }

  const actBuf = new Float64Array(dHidden);
  const dtanhBuf = new Float64Array(dHidden);
  const snapTheta0 = new Float64Array(numParams);
  const actTable0 = new Float64Array(numSamples * dHidden);
  const dtanhTable0 = new Float64Array(numSamples * dHidden);
  const predTable0 = new Float64Array(numSamples);
  let hasCache0 = false;

  function evalFullLoss(theta: Float64Array): number {
    let lossSum = 0.0;
    for (let n = 0; n < numSamples; n = (n + 1) | 0) {
      const pred = forwardSingle(theta, Math.imul(n, dIn));
      const err = pred - yData[n]!;
      lossSum += 0.5 * err * err;
    }
    return lossSum / numSamples;
  }

  function accumulateSampleGrad(
    theta: Float64Array,
    sampleIdx: number,
    invN: number,
    outGrad: Float64Array,
    useCache0: boolean,
  ): number {
    const xOff = Math.imul(sampleIdx, dIn);
    const b1Base = dIn * dHidden;
    const w2Base = b1Base + dHidden;
    let pred = theta[numParams - 1]!;
    if (useCache0) {
      const hOff = Math.imul(sampleIdx, dHidden);
      pred = predTable0[sampleIdx]!;
      for (let h = 0; h < dHidden; h = (h + 1) | 0) {
        actBuf[h] = actTable0[(hOff + h) | 0]!;
        dtanhBuf[h] = dtanhTable0[(hOff + h) | 0]!;
      }
    } else {
      for (let h = 0; h < dHidden; h = (h + 1) | 0) {
        const w1Row = Math.imul(h, dIn);
        const z =
          theta[b1Base + h]! +
          theta[w1Row]! * xData[xOff]! +
          theta[w1Row + 1]! * xData[xOff + 1]! +
          theta[w1Row + 2]! * xData[xOff + 2]! +
          theta[w1Row + 3]! * xData[xOff + 3]! +
          theta[w1Row + 4]! * xData[xOff + 4]! +
          theta[w1Row + 5]! * xData[xOff + 5]!;
        const a = fdlibmTanh(z);
        actBuf[h] = a;
        dtanhBuf[h] = 1.0 - a * a;
        pred += theta[w2Base + h]! * a;
      }
    }
    const err = pred - yData[sampleIdx]!;
    const dOut = err * invN;
    outGrad[numParams - 1] = outGrad[numParams - 1]! + dOut;
    for (let h = 0; h < dHidden; h = (h + 1) | 0) {
      const w2h = theta[w2Base + h]!;
      outGrad[w2Base + h] = outGrad[w2Base + h]! + dOut * actBuf[h]!;
      const dz = dOut * w2h * dtanhBuf[h]!;
      outGrad[b1Base + h] = outGrad[b1Base + h]! + dz;
      const w1Row = Math.imul(h, dIn);
      outGrad[w1Row] = outGrad[w1Row]! + dz * xData[xOff]!;
      outGrad[w1Row + 1] = outGrad[w1Row + 1]! + dz * xData[xOff + 1]!;
      outGrad[w1Row + 2] = outGrad[w1Row + 2]! + dz * xData[xOff + 2]!;
      outGrad[w1Row + 3] = outGrad[w1Row + 3]! + dz * xData[xOff + 3]!;
      outGrad[w1Row + 4] = outGrad[w1Row + 4]! + dz * xData[xOff + 4]!;
      outGrad[w1Row + 5] = outGrad[w1Row + 5]! + dz * xData[xOff + 5]!;
    }
    return 0.5 * err * err * invN;
  }

  // Gradient of the network output f(x_i) (not of the loss), by backpropagation; Phase 4 builds J_i^T J_i from it.
  function computeSampleOutputGradient(theta: Float64Array, sampleIdx: number, outGrad: Float64Array): void {
    const xOff = Math.imul(sampleIdx, dIn);
    const b1Base = dIn * dHidden;
    const w2Base = b1Base + dHidden;
    outGrad.fill(0.0);
    outGrad[numParams - 1] = 1.0;
    for (let h = 0; h < dHidden; h = (h + 1) | 0) {
      const w1Row = Math.imul(h, dIn);
      let z = theta[b1Base + h]!;
      for (let j = 0; j < dIn; j = (j + 1) | 0) z += theta[w1Row + j]! * xData[xOff + j]!;
      const a = fdlibmTanh(z);
      outGrad[w2Base + h] = a;
      const dz = theta[w2Base + h]! * (1.0 - a * a);
      outGrad[b1Base + h] = dz;
      for (let j = 0; j < dIn; j = (j + 1) | 0) outGrad[w1Row + j] = dz * xData[xOff + j]!;
    }
  }

  function isCachedTheta0(theta: Float64Array): boolean {
    if (!hasCache0) return false;
    for (let i = 0; i < numParams; i = (i + 1) | 0) {
      if (theta[i] !== snapTheta0[i]) return false;
    }
    return true;
  }

  function computeFullGradient(theta: Float64Array, outGrad: Float64Array): number {
    outGrad.fill(0.0);
    const invN = 1.0 / numSamples;
    const use0 = isCachedTheta0(theta);
    let loss = 0.0;
    for (let n = 0; n < numSamples; n = (n + 1) | 0) {
      loss += accumulateSampleGrad(theta, n, invN, outGrad, use0);
    }
    return loss;
  }

  function computeMinibatchGradient(
    theta: Float64Array,
    batchIndices: Int32Array,
    batchSize: number,
    outGrad: Float64Array,
  ): number {
    outGrad.fill(0.0);
    const invB = 1.0 / batchSize;
    const use0 = theta === checkpointTheta0 && isCachedTheta0(theta);
    let loss = 0.0;
    for (let b = 0; b < batchSize; b = (b + 1) | 0) {
      loss += accumulateSampleGrad(theta, batchIndices[b]!, invB, outGrad, use0);
    }
    return loss;
  }

  // Prepare warm checkpoint Theta_0 by training 80 steps from a perturbed initialization
  const checkpointTheta0 = new Float64Array(numParams);
  for (let i = 0; i < numParams; i = (i + 1) | 0) {
    checkpointTheta0[i] = teacherTheta[i]! + prng.nextGaussian() * 0.12;
  }
  const gradTmp = new Float64Array(numParams);
  for (let step = 0; step < 60; step = (step + 1) | 0) {
    computeFullGradient(checkpointTheta0, gradTmp);
    for (let i = 0; i < numParams; i = (i + 1) | 0) {
      checkpointTheta0[i] = checkpointTheta0[i]! - 0.015 * gradTmp[i]!;
    }
  }
  const checkpointLoss0 = evalFullLoss(checkpointTheta0);
  populateMlpCache0(checkpointTheta0, xData, numSamples, dIn, dHidden, numParams, snapTheta0, actTable0, dtanhTable0, predTable0);
  hasCache0 = true;

  return {
    dIn,
    dHidden,
    numParams,
    numSamples,
    xData,
    yData,
    checkpointTheta0,
    checkpointLoss0,
    evalFullLoss,
    computeFullGradient,
    computeMinibatchGradient,
    computeSampleOutputGradient,
  };
}
