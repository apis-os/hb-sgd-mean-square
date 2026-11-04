/**
 * @file labs/mean-square-boundary/evaluate-phase3-operator-radius-cells.ts
 * @description Phase 3 evaluator, serial (`evaluatePhase3OperatorRadiusCells`): every cell of `buildPhase3CellSpecs` through `evaluatePhase3OperatorRadiusCell`, assembled by `assemblePhase3Report`. The parallel runner `run-phase3.ts` produces the same rows in the same order.
 * @cold
 */
import { assemblePhase3Report } from './assemble-phase3-report.ts';
import { buildPhase3CellSpecs } from './build-phase3-cell-specs.ts';
import { evaluatePhase3OperatorRadiusCell } from './evaluate-phase3-operator-radius-cell.ts';
import type { Phase3CellRow, Phase3Report } from './phase3-types.ts';

export function evaluatePhase3OperatorRadiusCells(sizes: readonly number[]): Phase3Report {
  const specs = buildPhase3CellSpecs(sizes);
  const rows: Phase3CellRow[] = [];
  for (let i = 0; i < specs.length; (i = (i + 1) | 0)) rows.push(evaluatePhase3OperatorRadiusCell(specs[i]!));
  return assemblePhase3Report(rows, specs[0]!);
}
