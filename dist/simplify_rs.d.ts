/* tslint:disable */
/* eslint-disable */
/**
* WASM function to simplify a path represented by a sequence of points
* to a bezier curve with a maximum error tolerance.
* The input points are expected to be in the format:
* [{x: 0, y: 0}, {x: 1, y: 1}, ...]
*
* - `tolerance`: the allowed maximum error when fitting the curves through the segment points
*
* - `auto_scale_for_precision``: if true, the polygon will be scaled to a target area
* of TARGET_AUTO_SCALE_AREA before simplification and scaled back to the original scale
* after simplification. This increases the precision of the simplification.
*
* # Returns
*
* The output is a sequence of cubic bezier curves, each represented by 4 points:
* [{x: 0, y: 0}, {x: 1, y: 1}, {x: 2, y: 2}, {x: 3, y: 3}]
* [0] = start point
* [1] = control point 1
* [2] = control point 2
* [3] = end point
* @param {any} points_js
* @param {number} tolerance
* @param {boolean} auto_scale_for_precision
* @returns {any}
*/
export function simplify_js(points_js: any, tolerance: number, auto_scale_for_precision: boolean): any;
/**
* A 2D point
*/
export class Point {
  free(): void;
/**
* @param {number} x
* @param {number} y
*/
  constructor(x: number, y: number);
/**
*/
  x: number;
/**
*/
  y: number;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_point_free: (a: number) => void;
  readonly __wbg_get_point_x: (a: number) => number;
  readonly __wbg_set_point_x: (a: number, b: number) => void;
  readonly __wbg_get_point_y: (a: number) => number;
  readonly __wbg_set_point_y: (a: number, b: number) => void;
  readonly point_new: (a: number, b: number) => number;
  readonly simplify_js: (a: number, b: number, c: number) => number;
  readonly __wbindgen_malloc: (a: number, b: number) => number;
  readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
  readonly __wbindgen_exn_store: (a: number) => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;
/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {SyncInitInput} module
*
* @returns {InitOutput}
*/
export function initSync(module: SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {InitInput | Promise<InitInput>} module_or_path
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: InitInput | Promise<InitInput>): Promise<InitOutput>;
