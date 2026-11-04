/**
 * @file labs/mean-square-boundary/read-float64-view.ts
 * @description The intrinsic description of a Phase 6 argument (`readFloat64View`): `{ buffer, byteOffset, length }` read through the
 * getters of `%TypedArray%.prototype` captured when this module loads, or `null` unless the argument is a genuine `Float64Array` (of
 * any realm) on non-shared, non-detached memory with at least `minLength` entries. Nothing is read through properties of the argument
 * itself, so own or inherited `buffer`, `byteOffset`, `length` or method overrides, a subclass, a `Proxy` (which has no typed-array
 * slots) and other typed-array kinds are seen for what they are. The ninth audit round passed a subclass whose `subarray` lied, an own
 * `buffer` property that hid an alias, a `Proxy` that changed its values between reads and an `Int32Array` as scratch, and each broke
 * a certificate. Built-ins replaced before this module loads are outside the contract.
 * @cold
 */
const typedArrayPrototype = Object.getPrototypeOf(Int8Array.prototype) as object;
const getter = (key: PropertyKey): ((this: unknown) => unknown) => Object.getOwnPropertyDescriptor(typedArrayPrototype, key)!.get! as (this: unknown) => unknown;
const tagOf = getter(Symbol.toStringTag);
const bufferOf = getter('buffer');
const byteOffsetOf = getter('byteOffset');
const lengthOf = getter('length');
const plainByteLength = Object.getOwnPropertyDescriptor(ArrayBuffer.prototype, 'byteLength')!.get! as (this: unknown) => number;
// `ArrayBuffer.prototype.detached` (ES2024); a detached buffer reports length 0 and does not throw (tenth audit round).
const detachedOf = Object.getOwnPropertyDescriptor(ArrayBuffer.prototype, 'detached')?.get as ((this: unknown) => boolean) | undefined;

export function readFloat64View(view: unknown, minLength: number): { readonly buffer: ArrayBuffer; readonly byteOffset: number; readonly length: number } | null {
  if (tagOf.call(view) !== 'Float64Array') return null;
  const buffer = bufferOf.call(view) as ArrayBuffer;
  try {
    plainByteLength.call(buffer);
  } catch {
    return null;
  }
  if (detachedOf !== undefined && detachedOf.call(buffer)) return null;
  const length = lengthOf.call(view) as number;
  if (!(length >= minLength)) return null;
  return { buffer, byteOffset: byteOffsetOf.call(view) as number, length };
}
