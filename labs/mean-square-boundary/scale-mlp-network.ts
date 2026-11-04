/**
 * @file labs/mean-square-boundary/scale-mlp-network.ts
 * @description Multi-class 2-layer `fdlibmTanh` neural network (`f_Theta(x) = W_2 tanh(W_1 x + b_1) + b_2`, `P` up to `12,826` parameters) with analytical backpropagation and exact forward-over-reverse Pearlmutter Hessian-vector products (`createScaleMlpDatasetAndOps`).
 * @cold
 */
import { fdlibmTanh } from '../../src/math/index.ts';
import { createXorshift128Plus } from './prng-xorshift128p.ts';

function forwardSample(
  theta: Float64Array,
  xData: Float64Array,
  xOff: number,
  dIn: number,
  dHidden: number,
  dOut: number,
  actBuf: Float64Array,
  dtanhBuf: Float64Array,
  predBuf: Float64Array,
): void {
  const b1Base = Math.imul(dHidden, dIn);
  const w2Base = (b1Base + dHidden) | 0;
  const b2Base = (w2Base + Math.imul(dOut, dHidden)) | 0;
  const dIn4 = dIn & ~3;
  for (let h = 0; h < dHidden; (h = (h + 1) | 0)) {
    const w1Row = Math.imul(h, dIn);
    let z = theta[(b1Base + h) | 0]!;
    let j = 0;
    for (; j < dIn4; (j = (j + 4) | 0)) {
      const wIdx = (w1Row + j) | 0;
      const xIdx = (xOff + j) | 0;
      z =
        z +
        theta[wIdx]! * xData[xIdx]! +
        theta[(wIdx + 1) | 0]! * xData[(xIdx + 1) | 0]! +
        theta[(wIdx + 2) | 0]! * xData[(xIdx + 2) | 0]! +
        theta[(wIdx + 3) | 0]! * xData[(xIdx + 3) | 0]!;
    }
    for (; j < dIn; (j = (j + 1) | 0)) {
      z += theta[(w1Row + j) | 0]! * xData[(xOff + j) | 0]!;
    }
    const a = fdlibmTanh(z);
    actBuf[h] = a;
    dtanhBuf[h] = 1.0 - a * a;
  }
  const dHidden4 = dHidden & ~3;
  for (let c = 0; c < dOut; (c = (c + 1) | 0)) {
    const w2Row = (w2Base + Math.imul(c, dHidden)) | 0;
    let sum = theta[(b2Base + c) | 0]!;
    let h = 0;
    for (; h < dHidden4; (h = (h + 4) | 0)) {
      const wIdx = (w2Row + h) | 0;
      sum =
        sum +
        theta[wIdx]! * actBuf[h]! +
        theta[(wIdx + 1) | 0]! * actBuf[(h + 1) | 0]! +
        theta[(wIdx + 2) | 0]! * actBuf[(h + 2) | 0]! +
        theta[(wIdx + 3) | 0]! * actBuf[(h + 3) | 0]!;
    }
    for (; h < dHidden; (h = (h + 1) | 0)) {
      sum += theta[(w2Row + h) | 0]! * actBuf[h]!;
    }
    predBuf[c] = sum;
  }
}

function loadOrForwardSample(
  theta: Float64Array,
  xData: Float64Array,
  sampleIdx: number,
  dIn: number,
  dHidden: number,
  dOut: number,
  actBuf: Float64Array,
  dtanhBuf: Float64Array,
  predBuf: Float64Array,
  cachedActTable: Float64Array | null,
  cachedDtanhTable: Float64Array | null,
  cachedPredTable: Float64Array | null,
): void {
  if (cachedActTable !== null && cachedDtanhTable !== null && cachedPredTable !== null) {
    const hOff = Math.imul(sampleIdx, dHidden);
    const yOff = Math.imul(sampleIdx, dOut);
    for (let h = 0; h < dHidden; (h = (h + 1) | 0)) {
      actBuf[h] = cachedActTable[(hOff + h) | 0]!;
      dtanhBuf[h] = cachedDtanhTable[(hOff + h) | 0]!;
    }
    for (let c = 0; c < dOut; (c = (c + 1) | 0)) {
      predBuf[c] = cachedPredTable[(yOff + c) | 0]!;
    }
    return;
  }
  forwardSample(theta, xData, Math.imul(sampleIdx, dIn), dIn, dHidden, dOut, actBuf, dtanhBuf, predBuf);
}

function accumulateSampleGrad(
  theta: Float64Array,
  xData: Float64Array,
  yData: Float64Array,
  sampleIdx: number,
  dIn: number,
  dHidden: number,
  dOut: number,
  invCount: number,
  actBuf: Float64Array,
  dtanhBuf: Float64Array,
  predBuf: Float64Array,
  outGrad: Float64Array,
  cachedActTable: Float64Array | null,
  cachedDtanhTable: Float64Array | null,
  cachedPredTable: Float64Array | null,
): number {
  const xOff = Math.imul(sampleIdx, dIn);
  const yOff = Math.imul(sampleIdx, dOut);
  const b1Base = Math.imul(dHidden, dIn);
  const w2Base = (b1Base + dHidden) | 0;
  const b2Base = (w2Base + Math.imul(dOut, dHidden)) | 0;
  loadOrForwardSample(
    theta, xData, sampleIdx, dIn, dHidden, dOut, actBuf, dtanhBuf, predBuf,
    cachedActTable, cachedDtanhTable, cachedPredTable,
  );
  let sampleLoss = 0.0;
  for (let c = 0; c < dOut; (c = (c + 1) | 0)) {
    const err = predBuf[c]! - yData[(yOff + c) | 0]!;
    sampleLoss += 0.5 * err * err;
    const dOutC = err * invCount;
    predBuf[c] = dOutC;
    outGrad[(b2Base + c) | 0] = outGrad[(b2Base + c) | 0]! + dOutC;
    const w2Row = (w2Base + Math.imul(c, dHidden)) | 0;
    for (let h = 0; h < dHidden; (h = (h + 1) | 0)) {
      outGrad[(w2Row + h) | 0] = outGrad[(w2Row + h) | 0]! + dOutC * actBuf[h]!;
    }
  }
  const dIn4 = dIn & ~3;
  for (let h = 0; h < dHidden; (h = (h + 1) | 0)) {
    let gH = 0.0;
    for (let c = 0; c < dOut; (c = (c + 1) | 0)) {
      gH += theta[(w2Base + Math.imul(c, dHidden) + h) | 0]! * predBuf[c]!;
    }
    const dz = gH * dtanhBuf[h]!;
    outGrad[(b1Base + h) | 0] = outGrad[(b1Base + h) | 0]! + dz;
    const w1Row = Math.imul(h, dIn);
    let j = 0;
    for (; j < dIn4; (j = (j + 4) | 0)) {
      const wIdx = (w1Row + j) | 0;
      const xIdx = (xOff + j) | 0;
      outGrad[wIdx] = outGrad[wIdx]! + dz * xData[xIdx]!;
      outGrad[(wIdx + 1) | 0] = outGrad[(wIdx + 1) | 0]! + dz * xData[(xIdx + 1) | 0]!;
      outGrad[(wIdx + 2) | 0] = outGrad[(wIdx + 2) | 0]! + dz * xData[(xIdx + 2) | 0]!;
      outGrad[(wIdx + 3) | 0] = outGrad[(wIdx + 3) | 0]! + dz * xData[(xIdx + 3) | 0]!;
    }
    for (; j < dIn; (j = (j + 1) | 0)) {
      outGrad[(w1Row + j) | 0] = outGrad[(w1Row + j) | 0]! + dz * xData[(xOff + j) | 0]!;
    }
  }
  return sampleLoss * invCount;
}

function accumulateSampleHvp(
  theta: Float64Array,
  vec: Float64Array,
  xData: Float64Array,
  yData: Float64Array,
  sampleIdx: number,
  dIn: number,
  dHidden: number,
  dOut: number,
  invCount: number,
  actBuf: Float64Array,
  dtanhBuf: Float64Array,
  predBuf: Float64Array,
  dzBuf: Float64Array,
  daBuf: Float64Array,
  deBuf: Float64Array,
  outHv: Float64Array,
  cachedActTable: Float64Array | null,
  cachedDtanhTable: Float64Array | null,
  cachedPredTable: Float64Array | null,
): void {
  const xOff = Math.imul(sampleIdx, dIn);
  const yOff = Math.imul(sampleIdx, dOut);
  const b1Base = Math.imul(dHidden, dIn);
  const w2Base = (b1Base + dHidden) | 0;
  const b2Base = (w2Base + Math.imul(dOut, dHidden)) | 0;
  loadOrForwardSample(
    theta, xData, sampleIdx, dIn, dHidden, dOut, actBuf, dtanhBuf, predBuf,
    cachedActTable, cachedDtanhTable, cachedPredTable,
  );
  const dIn4 = dIn & ~3;
  for (let h = 0; h < dHidden; (h = (h + 1) | 0)) {
    const w1Row = Math.imul(h, dIn);
    let dz = vec[(b1Base + h) | 0]!;
    let j = 0;
    for (; j < dIn4; (j = (j + 4) | 0)) {
      const wIdx = (w1Row + j) | 0;
      const xIdx = (xOff + j) | 0;
      dz =
        dz +
        vec[wIdx]! * xData[xIdx]! +
        vec[(wIdx + 1) | 0]! * xData[(xIdx + 1) | 0]! +
        vec[(wIdx + 2) | 0]! * xData[(xIdx + 2) | 0]! +
        vec[(wIdx + 3) | 0]! * xData[(xIdx + 3) | 0]!;
    }
    for (; j < dIn; (j = (j + 1) | 0)) {
      dz += vec[(w1Row + j) | 0]! * xData[(xOff + j) | 0]!;
    }
    dzBuf[h] = dz;
    daBuf[h] = dtanhBuf[h]! * dz;
  }
  for (let c = 0; c < dOut; (c = (c + 1) | 0)) {
    const w2Row = (w2Base + Math.imul(c, dHidden)) | 0;
    const err = (predBuf[c]! - yData[(yOff + c) | 0]!) * invCount;
    predBuf[c] = err;
    let de = vec[(b2Base + c) | 0]!;
    for (let h = 0; h < dHidden; (h = (h + 1) | 0)) {
      de += vec[(w2Row + h) | 0]! * actBuf[h]! + theta[(w2Row + h) | 0]! * daBuf[h]!;
    }
    const deScaled = de * invCount;
    deBuf[c] = deScaled;
    outHv[(b2Base + c) | 0] = outHv[(b2Base + c) | 0]! + deScaled;
    for (let h = 0; h < dHidden; (h = (h + 1) | 0)) {
      outHv[(w2Row + h) | 0] = outHv[(w2Row + h) | 0]! + deScaled * actBuf[h]! + err * daBuf[h]!;
    }
  }
  for (let h = 0; h < dHidden; (h = (h + 1) | 0)) {
    let gH = 0.0;
    let dgH = 0.0;
    for (let c = 0; c < dOut; (c = (c + 1) | 0)) {
      const idx = (w2Base + Math.imul(c, dHidden) + h) | 0;
      const pC = predBuf[c]!;
      const tIdx = theta[idx]!;
      gH += tIdx * pC;
      dgH += vec[idx]! * pC + tIdx * deBuf[c]!;
    }
    const dtanhH = dtanhBuf[h]!;
    const d2tanh = -2.0 * actBuf[h]! * dtanhH;
    const dDelta = dgH * dtanhH + gH * d2tanh * dzBuf[h]!;
    outHv[(b1Base + h) | 0] = outHv[(b1Base + h) | 0]! + dDelta;
    const w1Row = Math.imul(h, dIn);
    let j = 0;
    for (; j < dIn4; (j = (j + 4) | 0)) {
      const wIdx = (w1Row + j) | 0;
      const xIdx = (xOff + j) | 0;
      outHv[wIdx] = outHv[wIdx]! + dDelta * xData[xIdx]!;
      outHv[(wIdx + 1) | 0] = outHv[(wIdx + 1) | 0]! + dDelta * xData[(xIdx + 1) | 0]!;
      outHv[(wIdx + 2) | 0] = outHv[(wIdx + 2) | 0]! + dDelta * xData[(xIdx + 2) | 0]!;
      outHv[(wIdx + 3) | 0] = outHv[(wIdx + 3) | 0]! + dDelta * xData[(xIdx + 3) | 0]!;
    }
    for (; j < dIn; (j = (j + 1) | 0)) {
      outHv[(w1Row + j) | 0] = outHv[(w1Row + j) | 0]! + dDelta * xData[(xOff + j) | 0]!;
    }
  }
}

// Gauss-Newton part J^T J vec of the per-sample Hessian (the residual term err * Hessian of the outputs dropped), scaled
// by invCount as in accumulateSampleHvp; used by Phase 4 to separate the curvature noise that batch sharpness sees.
function accumulateSampleGaussNewtonHvp(
  theta: Float64Array,
  vec: Float64Array,
  xData: Float64Array,
  sampleIdx: number,
  dIn: number,
  dHidden: number,
  dOut: number,
  invCount: number,
  actBuf: Float64Array,
  dtanhBuf: Float64Array,
  predBuf: Float64Array,
  daBuf: Float64Array,
  deBuf: Float64Array,
  outHv: Float64Array,
  cachedActTable: Float64Array | null,
  cachedDtanhTable: Float64Array | null,
  cachedPredTable: Float64Array | null,
): void {
  const xOff = Math.imul(sampleIdx, dIn);
  const b1Base = Math.imul(dHidden, dIn);
  const w2Base = (b1Base + dHidden) | 0;
  const b2Base = (w2Base + Math.imul(dOut, dHidden)) | 0;
  loadOrForwardSample(
    theta, xData, sampleIdx, dIn, dHidden, dOut, actBuf, dtanhBuf, predBuf,
    cachedActTable, cachedDtanhTable, cachedPredTable,
  );
  const dIn4 = dIn & ~3;
  for (let h = 0; h < dHidden; (h = (h + 1) | 0)) {
    const w1Row = Math.imul(h, dIn);
    let dz = vec[(b1Base + h) | 0]!;
    let j = 0;
    for (; j < dIn4; (j = (j + 4) | 0)) {
      const wIdx = (w1Row + j) | 0;
      const xIdx = (xOff + j) | 0;
      dz =
        dz +
        vec[wIdx]! * xData[xIdx]! +
        vec[(wIdx + 1) | 0]! * xData[(xIdx + 1) | 0]! +
        vec[(wIdx + 2) | 0]! * xData[(xIdx + 2) | 0]! +
        vec[(wIdx + 3) | 0]! * xData[(xIdx + 3) | 0]!;
    }
    for (; j < dIn; (j = (j + 1) | 0)) dz += vec[(w1Row + j) | 0]! * xData[(xOff + j) | 0]!;
    daBuf[h] = dtanhBuf[h]! * dz;
  }
  for (let c = 0; c < dOut; (c = (c + 1) | 0)) {
    const w2Row = (w2Base + Math.imul(c, dHidden)) | 0;
    let de = vec[(b2Base + c) | 0]!;
    for (let h = 0; h < dHidden; (h = (h + 1) | 0)) de += vec[(w2Row + h) | 0]! * actBuf[h]! + theta[(w2Row + h) | 0]! * daBuf[h]!;
    const deScaled = de * invCount;
    deBuf[c] = deScaled;
    outHv[(b2Base + c) | 0] = outHv[(b2Base + c) | 0]! + deScaled;
    for (let h = 0; h < dHidden; (h = (h + 1) | 0)) outHv[(w2Row + h) | 0] = outHv[(w2Row + h) | 0]! + deScaled * actBuf[h]!;
  }
  for (let h = 0; h < dHidden; (h = (h + 1) | 0)) {
    let back = 0.0;
    for (let c = 0; c < dOut; (c = (c + 1) | 0)) back += theta[(w2Base + Math.imul(c, dHidden) + h) | 0]! * deBuf[c]!;
    const dDelta = back * dtanhBuf[h]!;
    outHv[(b1Base + h) | 0] = outHv[(b1Base + h) | 0]! + dDelta;
    const w1Row = Math.imul(h, dIn);
    let j = 0;
    for (; j < dIn4; (j = (j + 4) | 0)) {
      const wIdx = (w1Row + j) | 0;
      const xIdx = (xOff + j) | 0;
      outHv[wIdx] = outHv[wIdx]! + dDelta * xData[xIdx]!;
      outHv[(wIdx + 1) | 0] = outHv[(wIdx + 1) | 0]! + dDelta * xData[(xIdx + 1) | 0]!;
      outHv[(wIdx + 2) | 0] = outHv[(wIdx + 2) | 0]! + dDelta * xData[(xIdx + 2) | 0]!;
      outHv[(wIdx + 3) | 0] = outHv[(wIdx + 3) | 0]! + dDelta * xData[(xIdx + 3) | 0]!;
    }
    for (; j < dIn; (j = (j + 1) | 0)) outHv[(w1Row + j) | 0] = outHv[(w1Row + j) | 0]! + dDelta * xData[(xOff + j) | 0]!;
  }
}

function evalSampleRayWithPrecomputedHead(
  sampleIdx: number,
  x: number,
  w2x: Float64Array,
  u2: Float64Array,
  b2x: Float64Array,
  ub2: Float64Array,
  yData: Float64Array,
  z0Table: Float64Array,
  dzTable: Float64Array,
  dHidden: number,
  dOut: number,
  actBuf: Float64Array,
  daBuf: Float64Array,
): number {
  const hOff = Math.imul(sampleIdx, dHidden);
  const yOff = Math.imul(sampleIdx, dOut);
  for (let h = 0; h < dHidden; (h = (h + 1) | 0)) {
    const dz = dzTable[(hOff + h) | 0]!;
    const a = fdlibmTanh(z0Table[(hOff + h) | 0]! + x * dz);
    actBuf[h] = a;
    daBuf[h] = (1.0 - a * a) * dz;
  }
  let dirGrad = 0.0;
  for (let c = 0; c < dOut; (c = (c + 1) | 0)) {
    const cOff = Math.imul(c, dHidden);
    let pred = b2x[c]!;
    let dpred = ub2[c]!;
    for (let h = 0; h < dHidden; (h = (h + 1) | 0)) {
      const idx = (cOff + h) | 0;
      const w2ch = w2x[idx]!;
      pred += w2ch * actBuf[h]!;
      dpred += u2[idx]! * actBuf[h]! + w2ch * daBuf[h]!;
    }
    dirGrad += (pred - yData[(yOff + c) | 0]!) * dpred;
  }
  return dirGrad;
}

function accumulateSampleRayTaylorJet(
  sampleIdx: number,
  w20: Float64Array,
  u2: Float64Array,
  b20: Float64Array,
  ub2: Float64Array,
  yData: Float64Array,
  dzTable: Float64Array,
  dHidden: number,
  dOut: number,
  act0Buf: Float64Array,
  aJetBuf: Float64Array,
  pJetBuf: Float64Array,
  outTaylorAccum: Float64Array,
): void {
  const hOff = Math.imul(sampleIdx, dHidden);
  const yOff = Math.imul(sampleIdx, dOut);
  for (let h = 0; h < dHidden; (h = (h + 1) | 0)) {
    const aOff = Math.imul(h, 9);
    const a0 = act0Buf[h]!;
    const dz = dzTable[(hOff + h) | 0]!;
    aJetBuf[aOff] = a0;
    aJetBuf[(aOff + 1) | 0] = dz * (1.0 - a0 * a0);
    for (let k = 1; k < 8; (k = (k + 1) | 0)) {
      let conv = 0.0;
      for (let j = 0; j <= k; (j = (j + 1) | 0)) {
        conv += aJetBuf[(aOff + j) | 0]! * aJetBuf[(aOff + k - j) | 0]!;
      }
      aJetBuf[(aOff + k + 1) | 0] = (-dz * conv) / ((k + 1) | 0);
    }
  }
  for (let c = 0; c < dOut; (c = (c + 1) | 0)) {
    const cOff = Math.imul(c, dHidden);
    for (let k = 0; k <= 8; (k = (k + 1) | 0)) {
      let pk = k === 0 ? b20[c]! - yData[(yOff + c) | 0]! : k === 1 ? ub2[c]! : 0.0;
      for (let h = 0; h < dHidden; (h = (h + 1) | 0)) {
        const idx = (cOff + h) | 0;
        const aOff = Math.imul(h, 9);
        pk += w20[idx]! * aJetBuf[(aOff + k) | 0]!;
        if (k >= 1) {
          pk += u2[idx]! * aJetBuf[(aOff + k - 1) | 0]!;
        }
      }
      pJetBuf[k] = pk;
    }
    for (let m = 0; m < 8; (m = (m + 1) | 0)) {
      let dm = 0.0;
      for (let j = 0; j <= m; (j = (j + 1) | 0)) {
        dm += ((j + 1) | 0) * pJetBuf[(j + 1) | 0]! * pJetBuf[(m - j) | 0]!;
      }
      outTaylorAccum[m] = outTaylorAccum[m]! + dm;
    }
  }
}

function buildScaleRayEvaluator(
  theta0: Float64Array,
  u: Float64Array,
  xData: Float64Array,
  yData: Float64Array,
  numSamples: number,
  dIn: number,
  dHidden: number,
  dOut: number,
  actBuf: Float64Array,
  daBuf: Float64Array,
): {
  readonly grad0AtSamples: Float64Array;
  readonly fullGrad0DotU: number;
  readonly rayTaylorCoeffs: Float64Array;
  evalFullDirectionalGrad(x: number): number;
  evalTaylorDirectionalGrad(x: number): number;
  evalMinibatchDirectionalGrad(x: number, batchIndices: Int32Array, batchSize: number): number;
} {
  const b1Base = Math.imul(dHidden, dIn);
  const w2Base = (b1Base + dHidden) | 0;
  const w2Len = Math.imul(dOut, dHidden);
  const b2Base = (w2Base + w2Len) | 0;
  const w20 = new Float64Array(w2Len);
  const u2 = new Float64Array(w2Len);
  const w2x = new Float64Array(w2Len);
  const b20 = new Float64Array(dOut);
  const ub2 = new Float64Array(dOut);
  const b2x = new Float64Array(dOut);
  for (let i = 0; i < w2Len; (i = (i + 1) | 0)) {
    w20[i] = theta0[(w2Base + i) | 0]!;
    u2[i] = u[(w2Base + i) | 0]!;
  }
  for (let c = 0; c < dOut; (c = (c + 1) | 0)) {
    b20[c] = theta0[(b2Base + c) | 0]!;
    ub2[c] = u[(b2Base + c) | 0]!;
  }

  const z0Table = new Float64Array(Math.imul(numSamples, dHidden));
  const dzTable = new Float64Array(Math.imul(numSamples, dHidden));
  const grad0AtSamples = new Float64Array(numSamples);
  const rayTaylorCoeffs = new Float64Array(8);
  const aJetBuf = new Float64Array(Math.imul(dHidden, 9));
  const pJetBuf = new Float64Array(9);
  let fullGrad0DotU = 0.0;

  for (let n = 0; n < numSamples; (n = (n + 1) | 0)) {
    const xOff = Math.imul(n, dIn);
    const hOff = Math.imul(n, dHidden);
    for (let h = 0; h < dHidden; (h = (h + 1) | 0)) {
      const w1Row = Math.imul(h, dIn);
      let z0 = theta0[(b1Base + h) | 0]!;
      let dz = u[(b1Base + h) | 0]!;
      for (let j = 0; j < dIn; (j = (j + 1) | 0)) {
        const xnj = xData[(xOff + j) | 0]!;
        z0 += theta0[(w1Row + j) | 0]! * xnj;
        dz += u[(w1Row + j) | 0]! * xnj;
      }
      z0Table[(hOff + h) | 0] = z0;
      dzTable[(hOff + h) | 0] = dz;
    }
    const g0n = evalSampleRayWithPrecomputedHead(
      n,
      0.0,
      w20,
      u2,
      b20,
      ub2,
      yData,
      z0Table,
      dzTable,
      dHidden,
      dOut,
      actBuf,
      daBuf,
    );
    grad0AtSamples[n] = g0n;
    fullGrad0DotU += g0n;
    accumulateSampleRayTaylorJet(
      n,
      w20,
      u2,
      b20,
      ub2,
      yData,
      dzTable,
      dHidden,
      dOut,
      actBuf,
      aJetBuf,
      pJetBuf,
      rayTaylorCoeffs,
    );
  }
  fullGrad0DotU /= numSamples;
  rayTaylorCoeffs[0] = fullGrad0DotU;
  for (let m = 1; m < 8; (m = (m + 1) | 0)) {
    rayTaylorCoeffs[m] = rayTaylorCoeffs[m]! / numSamples;
  }

  function evalFullDirectionalGrad(x: number): number {
    for (let i = 0; i < w2Len; (i = (i + 1) | 0)) {
      w2x[i] = w20[i]! + x * u2[i]!;
    }
    for (let c = 0; c < dOut; (c = (c + 1) | 0)) {
      b2x[c] = b20[c]! + x * ub2[c]!;
    }
    let sum = 0.0;
    for (let n = 0; n < numSamples; (n = (n + 1) | 0)) {
      sum += evalSampleRayWithPrecomputedHead(
        n,
        x,
        w2x,
        u2,
        b2x,
        ub2,
        yData,
        z0Table,
        dzTable,
        dHidden,
        dOut,
        actBuf,
        daBuf,
      );
    }
    return sum / numSamples;
  }

  function evalTaylorDirectionalGrad(x: number): number {
    if (Math.abs(x) <= 0.025) {
      return (
        rayTaylorCoeffs[0]! +
        x *
          (rayTaylorCoeffs[1]! +
            x *
              (rayTaylorCoeffs[2]! +
                x *
                  (rayTaylorCoeffs[3]! +
                    x *
                      (rayTaylorCoeffs[4]! +
                        x *
                          (rayTaylorCoeffs[5]! +
                            x * (rayTaylorCoeffs[6]! + x * rayTaylorCoeffs[7]!))))))
      );
    }
    return evalFullDirectionalGrad(x);
  }

  function evalMinibatchDirectionalGrad(
    x: number,
    batchIndices: Int32Array,
    batchSize: number,
  ): number {
    for (let i = 0; i < w2Len; (i = (i + 1) | 0)) {
      w2x[i] = w20[i]! + x * u2[i]!;
    }
    for (let c = 0; c < dOut; (c = (c + 1) | 0)) {
      b2x[c] = b20[c]! + x * ub2[c]!;
    }
    let sum = 0.0;
    for (let b = 0; b < batchSize; (b = (b + 1) | 0)) {
      sum += evalSampleRayWithPrecomputedHead(
        batchIndices[b]!,
        x,
        w2x,
        u2,
        b2x,
        ub2,
        yData,
        z0Table,
        dzTable,
        dHidden,
        dOut,
        actBuf,
        daBuf,
      );
    }
    return sum / batchSize;
  }

  return {
    grad0AtSamples,
    fullGrad0DotU,
    rayTaylorCoeffs,
    evalFullDirectionalGrad,
    evalTaylorDirectionalGrad,
    evalMinibatchDirectionalGrad,
  };
}

function populateForwardCache(
  theta: Float64Array,
  xData: Float64Array,
  numSamples: number,
  dIn: number,
  dHidden: number,
  dOut: number,
  actBuf: Float64Array,
  dtanhBuf: Float64Array,
  predBuf: Float64Array,
  snapTheta: Float64Array,
  actTable: Float64Array,
  dtanhTable: Float64Array,
  predTable: Float64Array,
): void {
  snapTheta.set(theta);
  for (let n = 0; n < numSamples; (n = (n + 1) | 0)) {
    forwardSample(theta, xData, Math.imul(n, dIn), dIn, dHidden, dOut, actBuf, dtanhBuf, predBuf);
    const hOff = Math.imul(n, dHidden);
    const yOff = Math.imul(n, dOut);
    for (let h = 0; h < dHidden; (h = (h + 1) | 0)) {
      actTable[(hOff + h) | 0] = actBuf[h]!;
      dtanhTable[(hOff + h) | 0] = dtanhBuf[h]!;
    }
    for (let c = 0; c < dOut; (c = (c + 1) | 0)) {
      predTable[(yOff + c) | 0] = predBuf[c]!;
    }
  }
}

function sameTheta(a: Float64Array, b: Float64Array, len: number): boolean {
  for (let i = 0; i < len; (i = (i + 1) | 0)) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

function initDatasetAndTeacher(
  dIn: number,
  dHidden: number,
  dOut: number,
  numSamples: number,
  seedLo: number,
  seedHi: number,
  actBuf: Float64Array,
  dtanhBuf: Float64Array,
  predBuf: Float64Array,
): {
  readonly xData: Float64Array;
  readonly yData: Float64Array;
  readonly checkpointTheta0: Float64Array;
  readonly teacherTheta: Float64Array;
} {
  const b1Base = Math.imul(dHidden, dIn);
  const w2Base = (b1Base + dHidden) | 0;
  const b2Base = (w2Base + Math.imul(dOut, dHidden)) | 0;
  const numParams = (b2Base + dOut) | 0;
  const prng = createXorshift128Plus(seedLo, seedHi);
  const teacherTheta = new Float64Array(numParams);
  const invSqrtIn = 1.0 / Math.sqrt(dIn);
  const invSqrtHid = 1.0 / Math.sqrt(dHidden);
  for (let i = 0; i < b1Base; (i = (i + 1) | 0)) teacherTheta[i] = prng.nextGaussian() * 1.15 * invSqrtIn;
  for (let h = 0; h < dHidden; (h = (h + 1) | 0)) teacherTheta[(b1Base + h) | 0] = prng.nextGaussian() * 0.15;
  for (let c = 0; c < dOut; (c = (c + 1) | 0)) {
    const w2Row = (w2Base + Math.imul(c, dHidden)) | 0;
    const modeBoost = c === 0 ? 2.2 : c === 1 ? 1.3 : 0.8;
    for (let h = 0; h < dHidden; (h = (h + 1) | 0)) teacherTheta[(w2Row + h) | 0] = prng.nextGaussian() * modeBoost * invSqrtHid;
    teacherTheta[(b2Base + c) | 0] = prng.nextGaussian() * 0.1;
  }
  const xData = new Float64Array(Math.imul(numSamples, dIn));
  const yData = new Float64Array(Math.imul(numSamples, dOut));
  for (let n = 0; n < numSamples; (n = (n + 1) | 0)) {
    const xOff = Math.imul(n, dIn);
    const yOff = Math.imul(n, dOut);
    const latent0 = prng.nextGaussian() * 2.1;
    const latent1 = prng.nextGaussian() * 1.25;
    for (let j = 0; j < dIn; (j = (j + 1) | 0)) {
      const pat0 = (j & 1) === 0 ? 0.55 : -0.55;
      const pat1 = (j & 2) === 0 ? 0.35 : -0.35;
      xData[(xOff + j) | 0] = latent0 * pat0 + latent1 * pat1 + prng.nextGaussian() * 0.3;
    }
    forwardSample(teacherTheta, xData, xOff, dIn, dHidden, dOut, actBuf, dtanhBuf, predBuf);
    for (let c = 0; c < dOut; (c = (c + 1) | 0)) yData[(yOff + c) | 0] = predBuf[c]! + prng.nextGaussian() * 0.05;
  }
  const checkpointTheta0 = new Float64Array(numParams);
  for (let i = 0; i < numParams; (i = (i + 1) | 0)) {
    checkpointTheta0[i] = teacherTheta[i]! + prng.nextGaussian() * 0.06 * invSqrtHid;
  }
  return { xData, yData, checkpointTheta0, teacherTheta };
}

export function createScaleMlpDatasetAndOps(
  dIn = 256,
  dHidden = 48,
  dOut = 10,
  numSamples = 64,
  seedLo = 0x67400001,
  seedHi = 0x9e3779b9,
): {
  readonly dIn: number;
  readonly dHidden: number;
  readonly dOut: number;
  readonly numParams: number;
  readonly numSamples: number;
  readonly checkpointTheta0: Float64Array;
  readonly checkpointLoss0: number;
  readonly xData: Float64Array;
  readonly yData: Float64Array;
  readonly teacherTheta: Float64Array;
  evalFullLoss(theta: Float64Array): number;
  computeFullGradient(theta: Float64Array, outGrad: Float64Array): number;
  computeMinibatchGradient(
    theta: Float64Array,
    batchIndices: Int32Array,
    batchSize: number,
    outGrad: Float64Array,
  ): number;
  computeAnalyticalHvp(theta: Float64Array, vec: Float64Array, outHv: Float64Array): void;
  computeSampleHvp(theta: Float64Array, vec: Float64Array, sampleIdx: number, outHv: Float64Array): void;
  computeSampleGaussNewtonHvp(theta: Float64Array, vec: Float64Array, sampleIdx: number, outHv: Float64Array): void;
  computeSampleOutputJacobian(theta: Float64Array, sampleIdx: number, outJac: Float64Array): void;
  createRayEvaluator(u: Float64Array): ReturnType<typeof buildScaleRayEvaluator>;
} {
  const b1Base = Math.imul(dHidden, dIn);
  const w2Base = (b1Base + dHidden) | 0;
  const b2Base = (w2Base + Math.imul(dOut, dHidden)) | 0;
  const numParams = (b2Base + dOut) | 0;
  const actBuf = new Float64Array(dHidden);
  const dtanhBuf = new Float64Array(dHidden);
  const predBuf = new Float64Array(dOut);
  const dzBuf = new Float64Array(dHidden);
  const daBuf = new Float64Array(dHidden);
  const deBuf = new Float64Array(dOut);
  const { xData, yData, checkpointTheta0, teacherTheta } = initDatasetAndTeacher(
    dIn, dHidden, dOut, numSamples, seedLo, seedHi, actBuf, dtanhBuf, predBuf,
  );
  const snapTheta0 = new Float64Array(numParams);
  const actTable0 = new Float64Array(Math.imul(numSamples, dHidden));
  const dtanhTable0 = new Float64Array(Math.imul(numSamples, dHidden));
  const predTable0 = new Float64Array(Math.imul(numSamples, dOut));
  const snapTheta1 = new Float64Array(numParams);
  const actTable1 = new Float64Array(Math.imul(numSamples, dHidden));
  const dtanhTable1 = new Float64Array(Math.imul(numSamples, dHidden));
  const predTable1 = new Float64Array(Math.imul(numSamples, dOut));
  let hasCache0 = false;
  let hasCache1 = false;

  const ensureHvpCache = (theta: Float64Array): readonly [Float64Array, Float64Array, Float64Array] => {
    if (hasCache0 && (theta === checkpointTheta0 ? theta[0] === snapTheta0[0] && sameTheta(theta, snapTheta0, numParams) : sameTheta(theta, snapTheta0, numParams))) {
      return [actTable0, dtanhTable0, predTable0];
    }
    if (!hasCache1 || !sameTheta(theta, snapTheta1, numParams)) {
      populateForwardCache(theta, xData, numSamples, dIn, dHidden, dOut, actBuf, dtanhBuf, predBuf, snapTheta1, actTable1, dtanhTable1, predTable1);
      hasCache1 = true;
    }
    return [actTable1, dtanhTable1, predTable1];
  };

  function evalFullLoss(theta: Float64Array): number {
    let total = 0.0;
    for (let n = 0; n < numSamples; (n = (n + 1) | 0)) {
      const xOff = Math.imul(n, dIn);
      const yOff = Math.imul(n, dOut);
      forwardSample(theta, xData, xOff, dIn, dHidden, dOut, actBuf, dtanhBuf, predBuf);
      for (let c = 0; c < dOut; (c = (c + 1) | 0)) {
        const err = predBuf[c]! - yData[(yOff + c) | 0]!;
        total += 0.5 * err * err;
      }
    }
    return total / numSamples;
  }

  function computeFullGradient(theta: Float64Array, outGrad: Float64Array): number {
    outGrad.fill(0.0);
    const invN = 1.0 / numSamples;
    const use0 = hasCache0 && sameTheta(theta, snapTheta0, numParams);
    const aT = use0 ? actTable0 : null;
    const dT = use0 ? dtanhTable0 : null;
    const pT = use0 ? predTable0 : null;
    let loss = 0.0;
    for (let n = 0; n < numSamples; (n = (n + 1) | 0)) {
      loss += accumulateSampleGrad(theta, xData, yData, n, dIn, dHidden, dOut, invN, actBuf, dtanhBuf, predBuf, outGrad, aT, dT, pT);
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
    const use0 = hasCache0 && (theta === checkpointTheta0 ? theta[0] === snapTheta0[0] && sameTheta(theta, snapTheta0, numParams) : false);
    const aT = use0 ? actTable0 : null;
    const dT = use0 ? dtanhTable0 : null;
    const pT = use0 ? predTable0 : null;
    let loss = 0.0;
    for (let b = 0; b < batchSize; (b = (b + 1) | 0)) {
      loss += accumulateSampleGrad(
        theta, xData, yData, batchIndices[b]!, dIn, dHidden, dOut, invB, actBuf, dtanhBuf, predBuf, outGrad, aT, dT, pT,
      );
    }
    return loss;
  }

  function computeAnalyticalHvp(theta: Float64Array, vec: Float64Array, outHv: Float64Array): void {
    outHv.fill(0.0);
    const invN = 1.0 / numSamples;
    const [aT, dT, pT] = ensureHvpCache(theta);
    for (let n = 0; n < numSamples; (n = (n + 1) | 0)) {
      accumulateSampleHvp(theta, vec, xData, yData, n, dIn, dHidden, dOut, invN, actBuf, dtanhBuf, predBuf, dzBuf, daBuf, deBuf, outHv, aT, dT, pT);
    }
  }

  // Hessian of the loss of one sample (not divided by N) applied to `vec`.
  function computeSampleHvp(theta: Float64Array, vec: Float64Array, sampleIdx: number, outHv: Float64Array): void {
    outHv.fill(0.0);
    const [aT, dT, pT] = ensureHvpCache(theta);
    accumulateSampleHvp(theta, vec, xData, yData, sampleIdx, dIn, dHidden, dOut, 1.0, actBuf, dtanhBuf, predBuf, dzBuf, daBuf, deBuf, outHv, aT, dT, pT);
  }

  // Gauss-Newton part of the Hessian of one sample (not divided by N) applied to `vec`.
  function computeSampleGaussNewtonHvp(theta: Float64Array, vec: Float64Array, sampleIdx: number, outHv: Float64Array): void {
    outHv.fill(0.0);
    const [aT, dT, pT] = ensureHvpCache(theta);
    accumulateSampleGaussNewtonHvp(theta, vec, xData, sampleIdx, dIn, dHidden, dOut, 1.0, actBuf, dtanhBuf, predBuf, daBuf, deBuf, outHv, aT, dT, pT);
  }

  // Jacobian of the dOut outputs of one sample, row-major dOut x numParams (row c = gradient of f_c).
  function computeSampleOutputJacobian(theta: Float64Array, sampleIdx: number, outJac: Float64Array): void {
    const xOff = Math.imul(sampleIdx, dIn);
    forwardSample(theta, xData, xOff, dIn, dHidden, dOut, actBuf, dtanhBuf, predBuf);
    outJac.fill(0.0);
    for (let c = 0; c < dOut; (c = (c + 1) | 0)) {
      const row = Math.imul(c, numParams);
      const w2Row = (w2Base + Math.imul(c, dHidden)) | 0;
      outJac[(row + b2Base + c) | 0] = 1.0;
      for (let h = 0; h < dHidden; (h = (h + 1) | 0)) {
        outJac[(row + w2Row + h) | 0] = actBuf[h]!;
        const dz = theta[(w2Row + h) | 0]! * dtanhBuf[h]!;
        outJac[(row + b1Base + h) | 0] = dz;
        const w1Row = Math.imul(h, dIn);
        for (let j = 0; j < dIn; (j = (j + 1) | 0)) outJac[(row + w1Row + j) | 0] = dz * xData[(xOff + j) | 0]!;
      }
    }
  }

  const gradTmp = new Float64Array(numParams);
  for (let step = 0; step < 24; (step = (step + 1) | 0)) {
    computeFullGradient(checkpointTheta0, gradTmp);
    for (let i = 0; i < numParams; (i = (i + 1) | 0)) {
      checkpointTheta0[i] = checkpointTheta0[i]! - 0.01 * gradTmp[i]!;
    }
  }
  const checkpointLoss0 = evalFullLoss(checkpointTheta0);
  populateForwardCache(checkpointTheta0, xData, numSamples, dIn, dHidden, dOut, actBuf, dtanhBuf, predBuf, snapTheta0, actTable0, dtanhTable0, predTable0);
  hasCache0 = true;

  function createRayEvaluator(u: Float64Array): ReturnType<typeof buildScaleRayEvaluator> {
    return buildScaleRayEvaluator(
      checkpointTheta0,
      u,
      xData,
      yData,
      numSamples,
      dIn,
      dHidden,
      dOut,
      actBuf,
      daBuf,
    );
  }

  return {
    dIn,
    dHidden,
    dOut,
    numParams,
    numSamples,
    checkpointTheta0,
    checkpointLoss0,
    xData,
    yData,
    teacherTheta,
    evalFullLoss,
    computeFullGradient,
    computeMinibatchGradient,
    computeAnalyticalHvp,
    computeSampleHvp,
    computeSampleGaussNewtonHvp,
    computeSampleOutputJacobian,
    createRayEvaluator,
  };
}
