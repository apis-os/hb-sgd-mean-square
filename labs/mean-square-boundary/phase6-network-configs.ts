/**
 * @file labs/mean-square-boundary/phase6-network-configs.ts
 * @description The networks of the Phase 6 pilot (`PHASE6_NETWORK_CONFIGS`), keyed by parameter count. `3610` and `12826` are the
 * lab Phase 4 architectures (`64-48-10` and `256-48-10`, `N = 64`) with pilot dataset seeds; both have the same data-active dimension,
 * `48 * 64 + 10 * 49 = 3 562`, because `N = 64` input directions bound what the first layer sees. `34186` (`256-128-10`, `N = 256`)
 * raises the active dimension to `128 * 256 + 10 * 129 = 34 058`. Per entry: the data generator seeds of `createScaleMlpDatasetAndOps`,
 * the held-out seeds of `createPhase6HeldOutSet`, how often the training loss is checked for divergence (every step for the
 * lab Phase 4 networks; every 10 steps at `N = 256`, where a full loss costs more than a training step), the checkpoint interval, the
 * interval of the Gauss–Newton window block, and whether the final point also runs the Gauss–Newton Lyapunov arm (its projector
 * orthonormalises `N dOut` rows of length P, about 20 minutes at `N = 256`, so it is skipped there).
 * @cold
 */
export const PHASE6_NETWORK_CONFIGS: Readonly<Record<'3610' | '12826' | '34186', {
  readonly dIn: number;
  readonly dHidden: number;
  readonly dOut: number;
  readonly numSamples: number;
  readonly seedLo: number;
  readonly seedHi: number;
  readonly testSeedLo: number;
  readonly testSeedHi: number;
  readonly divergenceCheckEvery: number;
  readonly checkpointEvery: number;
  readonly gaussNewtonEvery: number;
  readonly gaussNewtonLyapunovArm: boolean;
}>> = {
  '3610': { dIn: 64, dHidden: 48, dOut: 10, numSamples: 64, seedLo: 0x67500001, seedHi: 0x9e3779b9, testSeedLo: 0x67510001, testSeedHi: 0x2545f491, divergenceCheckEvery: 1, checkpointEvery: 200, gaussNewtonEvery: 1000, gaussNewtonLyapunovArm: true },
  '12826': { dIn: 256, dHidden: 48, dOut: 10, numSamples: 64, seedLo: 0x67500002, seedHi: 0x85ebca6b, testSeedLo: 0x67510002, testSeedHi: 0x2545f491, divergenceCheckEvery: 1, checkpointEvery: 200, gaussNewtonEvery: 1000, gaussNewtonLyapunovArm: true },
  '34186': { dIn: 256, dHidden: 128, dOut: 10, numSamples: 256, seedLo: 0x67500003, seedHi: 0x9e3779b9, testSeedLo: 0x67510003, testSeedHi: 0x2545f491, divergenceCheckEvery: 10, checkpointEvery: 400, gaussNewtonEvery: 2000, gaussNewtonLyapunovArm: false },
};
