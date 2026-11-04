/**
 * @file labs/mean-square-boundary/parse-phase4-cell-log-line.ts
 * @description Reads the Lanczos outcome of one Phase 4 cell back from a progress-log line (`parsePhase4CellLogLine`), for
 * resuming an interrupted launch: the text line `<time> <network>: beta <b> g <g>: eta* <x> (<k> Lanczos steps, converged
 * <bool>), 1D <y>, critical mode <json>` written by `evaluatePhase4NetworkRow`, or its `CELL <json>` line. The numbers were
 * printed with `String(number)`, which round-trips exactly except for the sign of a zero (`-0` prints as `0`; only a share of
 * the critical mode can be `-0`), so the parsed values are the computed ones bit for bit. Lines of
 * other kinds, and text lines without the critical mode (which carries the Lanczos residual), give `null`.
 * @cold
 */
import type { Phase4ResumedLanczosCell, Phase4ThresholdCell } from './phase4-types.ts';

const TEXT = /^\S+ (\S+): beta (\S+) g (\S+): eta\* (\S+) \((\d+) Lanczos steps, converged (true|false)(?:, resumed)?\), 1D (\S+?), critical mode (\{.*\})$/;
const JSON_LINE = /^\S+ CELL (\{.*\})$/;

export function parsePhase4CellLogLine(line: string): (Phase4ResumedLanczosCell & { readonly networkId: string }) | null {
  const j = JSON_LINE.exec(line);
  if (j !== null) {
    const c = JSON.parse(j[1]!) as Phase4ThresholdCell & { readonly networkId: string };
    return {
      networkId: c.networkId,
      beta: c.beta,
      tetherGrid: c.tetherGrid,
      etaThmG6: c.etaThmG6,
      etaOneDim: c.etaOneDim,
      lanczosIterations: c.lanczosIterations,
      lanczosConverged: c.lanczosConverged,
      lanczosRelResidual: c.lanczosRelResidual,
      ...(c.criticalMode === undefined ? {} : { criticalMode: c.criticalMode }),
    };
  }
  const m = TEXT.exec(line);
  if (m === null) return null;
  const criticalMode = JSON.parse(m[8]!) as NonNullable<Phase4ThresholdCell['criticalMode']>;
  return {
    networkId: m[1]!,
    beta: Number(m[2]),
    tetherGrid: Number(m[3]),
    etaThmG6: Number(m[4]),
    etaOneDim: Number(m[7]),
    lanczosIterations: Number.parseInt(m[5]!, 10),
    lanczosConverged: m[6] === 'true',
    lanczosRelResidual: criticalMode.lanczosRelResidual,
    criticalMode,
  };
}
