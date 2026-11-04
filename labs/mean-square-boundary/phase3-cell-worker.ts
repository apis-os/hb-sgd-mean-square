/**
 * @file labs/mean-square-boundary/phase3-cell-worker.ts
 * @description Worker-thread entry for Phase 3: evaluates the assigned cell with `evaluatePhase3OperatorRadiusCell` and posts the row back.
 * @cold
 */
import { parentPort, workerData } from 'node:worker_threads';
import { evaluatePhase3OperatorRadiusCell } from './evaluate-phase3-operator-radius-cell.ts';
import type { Phase3CellSpec } from './phase3-types.ts';

parentPort?.postMessage(evaluatePhase3OperatorRadiusCell(workerData as Phase3CellSpec));
