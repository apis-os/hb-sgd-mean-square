/**
 * @file tests/workers/evaluate-phase1-reduced.ts
 * @description Worker thread for `tests/phase1-secular-ceiling.test.ts`: runs the reduced `evaluatePhase1RademacherCells`
 * (1 seed, 2 streams, 120 steps, 3 bisection iterations) and posts the report. Running it off the test worker keeps the
 * Vitest worker's event loop free, so its RPC calls, whose timeout is fixed at 60 s, do not expire on a loaded machine.
 */
import { parentPort } from 'node:worker_threads';
import { evaluatePhase1RademacherCells } from '../../labs/mean-square-boundary/evaluate-phase1-rademacher-cells.ts';

if (parentPort === null) throw new Error('evaluate-phase1-reduced.ts must run as a worker thread');
parentPort.postMessage(evaluatePhase1RademacherCells(1, 2, 120, 3));
