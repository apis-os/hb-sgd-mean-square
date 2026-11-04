// src/math/index.ts — the only import surface of src/math: bit-exact portable math (FDLIBM 5.3 ports)
// Used instead of Math.exp/log/sin/cos/tanh: native transcendentals differ across engines and CPUs. This repository keeps the files
// of the functions its code calls (exp, log, sin, cos, tanh) and their dependencies; those files also hold tan, sinh, cosh and the
// bit-level helpers abs, sign, min and max, which the code does not call.
export * from './float-words.ts';
export * from './abs-sign-min-max.ts';
export * from './scalbn.ts';
export * from './exp.ts';
export * from './expm1.ts';
export * from './log.ts';
export * from './sin-cos-tan.ts';
export * from './rem-pio2.ts';
export * from './sinh-cosh-tanh.ts';
