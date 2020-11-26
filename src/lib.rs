#![feature(test)]

use std::ops::Range;
use std::hint::unreachable_unchecked;

#[cfg(not(feature = "double_precision"))]
pub type Float = f32;
#[cfg(feature = "double_precision")]
pub type Float = f64;

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub struct Point { pub x: Float, pub y: Float }

impl Point {
    #[inline]
    pub fn add(&self, p: Point) -> Point { Point { x: self.x + p.x, y: self.y + p.y } }
    #[inline]
    pub fn subtract(&self, p: Point) -> Point { Point { x: self.x - p.x, y: self.y - p.y } }
    #[inline]
    pub fn multiply(&self, f: Float) -> Point { Point { x: self.x * f, y: self.y * f } }
    #[inline]
    pub fn negate(&self) -> Point { Point { x: -self.x, y: -self.y } }
    #[inline]
    pub fn distance(&self, p: Point) -> Float { (p.x - self.x).hypot(p.y - self.y) }
    #[inline]
    pub fn dot(&self, p: Point) -> Float { self.x * p.x + self.y * p.y }
    #[inline]
    pub fn normalize(&self, length: Float) -> Point {
        let current =  self.x.hypot(self.y);
        let scale = if current.abs() > Float::EPSILON { length / current } else { 0.0 };
        let res = Point { x: self.x * scale, y: self.y * scale };
        res
    }
}

/// Default value for the `simplify_curve(tolerance)` parameter
pub const DEFAULT_TOLERANCE: f32 = 2.5;

/// Fits a sequence of as few curves as possible through the path's anchor
/// points, ignoring the path items's curve-handles, with an allowed maximum
/// error. When called on {@link CompoundPath} items, each of the nested
/// paths is simplified. On {@link Path} items, the {@link Path#segments}
/// array is processed and replaced by the resulting sequence of fitted
/// curves.
///
/// This method can be used to process and simplify the point data received
/// from a mouse or touch device.
///
/// @param tolerance - the allowed maximum error when fitting
///     the curves through the segment points
///
/// @return true if the method was capable of fitting curves
///     through the path's segment points
///
pub fn simplify_curve(points: &[Point], tolerance: Float) -> Vec<Point> {

    // filter points for duplicates
    let mut cur_points = points.windows(2).filter_map(|w| {
        let first = w.get(0)?;
        let next = w.get(1)?;
        if first == next { None } else { Some(*first) }
    }).collect::<Vec<Point>>();

    // windows() is fast, but excludes the last point
    if let (Some(last_minus_one), Some(last)) = (points.get(points.len() - 2), points.last()) {
        if last_minus_one != last { cur_points.push(*last); }
    }

    // sanity check
    if cur_points.len() < 2 {
        return cur_points;
    } else if cur_points.len() == 2 {
        return vec![cur_points[0], cur_points[1]];
    }

    // cur_points.len() is assured to be greater than 2
    let closed = cur_points.first() == cur_points.last();

    // We need to duplicate the first and last point when
    // simplifying a closed path
    if closed {
        let last = match points.last().copied() {
            Some(s) => s,
            None => unsafe { unreachable_unchecked() },
        };
        let first = match points.first().copied() {
            Some(s) => s,
            None => unsafe { unreachable_unchecked() },
        };
        let mut new_cur_points = vec![last];
        new_cur_points.extend(cur_points.drain(..));
        new_cur_points.push(first);
        cur_points = new_cur_points;
    }

    fit(&cur_points[..], tolerance)
}

#[derive(Clone)]
struct Split {
    global_range: Range<usize>,
    tan1: Point,
    tan2: Point,
}

#[inline]
fn fit(points: &[Point], tolerance: Float) -> Vec<Point> {

    // To support reducing paths with multiple points in the same place
    // to one segment:
    let mut segments = Vec::new();
    let distances = chord_length_parametrize(points);

    if distances.len() != points.len() {
        return segments; // never happens, necessary for compiler
    }

    // elide bounds checks
    if points.len() < 2 {
        return segments;
    } else if points.len() == 2 {
        return vec![points[0], points[1]];
    }

    let mut splits_to_eval = vec![Split {
        global_range: 0..points.len(),
        tan1: points[1].subtract(points[0]),
        tan2: points[points.len() - 2].subtract(points[points.len() - 1]),
    }];

    while let Some(split) = splits_to_eval.pop() {

        // elide bounds checks
        if split.global_range.end > points.len() {
            continue;
        }

        let result = fit_cubic(FitCubicParams {
            points: &points[split.global_range.clone()],
            chord_lengths: &distances[split.global_range.clone()],
            segments: &mut segments,
            error: tolerance,
            tan1: split.tan1,
            tan2: split.tan2,
        });

        if let Some(r) = result {
            // Fitting failed -- split at max error point and fit recursively
            let tan_center = points[split.global_range.start + r - 1].subtract(points[split.global_range.start + r + 1]);
            splits_to_eval.extend_from_slice(&[
                Split {
                    global_range: split.global_range.start..(split.global_range.start + r + 1),
                    tan1: split.tan1,
                    tan2: tan_center,
                },
                Split {
                    global_range: (split.global_range.start + r)..split.global_range.end,
                    tan1: tan_center.negate(),
                    tan2: split.tan2,
                },
            ]);
        }
    }

    segments
}

struct FitCubicParams<'a> {
    segments: &'a mut Vec<Point>,
    points: &'a [Point],
    chord_lengths: &'a [Float],
    error: Float,
    tan1: Point,
    tan2: Point,
}

fn fit_cubic(params: FitCubicParams) -> Option<usize> {

    let FitCubicParams { segments, points, chord_lengths, error, tan1, tan2 } = params;

    // Use heuristic if region only has two points in it
    if points.len() < 3 {
        return None;
    } else if points.len() == 3 {
        let pt1 = points[0];
        let pt2 = points[2];
        let dist = pt1.distance(pt2) / 3.0;
        add_curve(segments, &[
            pt1,
            pt1.add(tan1.normalize(dist)),
            pt2.add(tan2.normalize(dist)),
            pt2
        ]);
        return None;
    }

    // points.len() at least 4

    // Parameterize points, and attempt to fit curve
    // (Slightly) faster version of chord lengths, re-uses the results from original count
    let mut u_prime = chord_lengths.to_owned();
    let u_prime_first = match u_prime.first().copied() {
        Some(s) => s,
        None => unsafe { unreachable_unchecked() },
    };
    let u_prime_last = match u_prime.last().copied() {
        Some(s) => s,
        None => unsafe { unreachable_unchecked() },
    };
    let u_prime_last = u_prime_last - u_prime_first;
    u_prime.iter_mut().for_each(|p| { *p = (*p - u_prime_first) / u_prime_last; });

    let mut max_error = error.max(error.powi(2));
    let mut parameters_in_order = true;
    let mut split = 2;

    // Try 4 iterations
    for _ in 0..4 {

        let curve = generate_bezier(points, &u_prime, tan1, tan2);

        //  Find max deviation of points to fitted curve
        let max = find_max_error(points, &curve, &u_prime);

        if max.error < error && parameters_in_order {
            // solution found
            add_curve(segments, &curve);
            return None;
        }

        split = max.index;

        // If error not too large, try reparameterization and iteration
        if max.error >= max_error {
            break;
        }
        parameters_in_order = reparameterize(points, &mut u_prime, &curve);
        max_error = max.error;
    }

    Some(split)
}

#[inline]
fn add_curve(segments: &mut Vec<Point>, curve: &[Point;4]) {
    segments.extend_from_slice(curve);
}

#[inline]
#[allow(non_snake_case)]
fn generate_bezier(points: &[Point], u_prime: &[Float], tan1: Point, tan2: Point) -> [Point;4] {

    debug_assert!(u_prime.len() > 3);
    debug_assert!(points.len() > 3);
    debug_assert!(u_prime.len() == points.len());

    let pt1 = &points[0];
    let pt2 = &points[points.len() - 1];

    // Create the C and X matrices
    let mut C = [
        [0.0, 0.0],
        [0.0, 0.0]
    ];
    let mut X = [0.0, 0.0];

    for (p, u) in points.iter().zip(u_prime.iter()) {

        let t = 1.0 - u;
        let b = 3.0 * u * t;
        let b0 = t * t * t;
        let b1 = b * t;
        let b2 = b * u;
        let b3 = u * u * u;
        let a1 = tan1.normalize(b1);
        let a2 = tan2.normalize(b2);
        let pt1_multiplied = pt1.multiply(b0 + b1);
        let pt2_multiplied = pt2.multiply(b2 + b3);
        let tmp = p.subtract(pt1_multiplied).subtract(pt2_multiplied);

        C[0][0] += a1.dot(a1);
        C[0][1] += a1.dot(a2);
        C[1][0] = C[0][1];
        C[1][1] += a2.dot(a2);

        X[0] += a1.dot(tmp);
        X[1] += a2.dot(tmp);
    }

    // Compute the determinants of C and X
    let det_c0_c1 = C[0][0] * C[1][1] - C[1][0] * C[0][1];

    let mut alpha1;
    let mut alpha2;

    if det_c0_c1.abs() > Float::EPSILON {
        // Kramer's rule
        let det_c0_x = C[0][0] * X[1]    - C[1][0] * X[0];
        let det_x_c1 = X[0]    * C[1][1] - X[1]    * C[0][1];
        // Derive alpha values
        alpha1 = det_x_c1 / det_c0_c1;
        alpha2 = det_c0_x / det_c0_c1;
    } else {
        // Matrix is under-determined, try assuming alpha1 == alpha2
        let c0 = C[0][0] + C[0][1];
        let c1 = C[1][0] + C[1][1];
        alpha1 = if c0.abs() > Float::EPSILON {
            X[0] / c0
        } else if c1.abs() > Float::EPSILON {
            X[1] / c1
        } else {
            0.0
        };
        alpha2 = alpha1;
    }

    // If alpha negative, use the Wu/Barsky heuristic (see text)
    // (if alpha is 0, you get coincident control points that lead to
    // divide by zero in any subsequent NewtonRaphsonRootFind() call.
    let seg_length = pt2.distance(*pt1);
    let eps = Float::EPSILON * seg_length;
    let mut handle1_2 = None;

    if alpha1 < eps || alpha2 < eps {
        // fall back on standard (probably inaccurate) formula,
        // and subdivide further if needed.
        alpha1 = seg_length / 3.0;
        alpha2 = alpha1;
    } else {
        // Check if the found control points are in the right order when
        // projected onto the line through pt1 and pt2.
        let line = pt2.subtract(*pt1);

        // Control points 1 and 2 are positioned an alpha distance out
        // on the tangent vectors, left and right, respectively
        let tmp_handle_1 = tan1.normalize(alpha1);
        let tmp_handle_2 = tan2.normalize(alpha2);

        let seg_length_squared = seg_length * seg_length;

        if tmp_handle_1.dot(line) - tmp_handle_2.dot(line) > seg_length_squared {
            // Fall back to the Wu/Barsky heuristic above.
            alpha1 = seg_length / 3.0;
            alpha2 = alpha1;
            // Force recalculation
            handle1_2 = None;
        } else {
            handle1_2 = Some((tmp_handle_1, tmp_handle_2));
        }
    }

    // First and last control points of the Bezier curve are
    // positioned exactly at the first and last data points
    if let Some((h1, h2)) = handle1_2 {
        [*pt1, pt1.add(h1), pt2.add(h2), *pt2]
    } else {
        [*pt1, pt1.add(tan1.normalize(alpha1)), pt2.add(tan2.normalize(alpha2)), *pt2]
    }
}

/// Given set of points and their parameterization, try to find
/// a better parameterization.
#[inline]
fn reparameterize(points: &[Point], u: &mut [Float], curve: &[Point;4]) -> bool {

    points.iter().zip(u.iter_mut()).for_each(|(p, u)| { *u = find_root(curve, p, *u); });

    // Detect if the new parameterization has reordered the points.
    // In that case, we would fit the points of the path in the wrong order.
    !u.windows(2).any(|w| w[1] <= w[0])
}

#[inline]
fn find_root(curve: &[Point;4], point: &Point, u: Float) -> Float {

    let mut curve1 = [Point { x: 0.0, y: 0.0 };3];
    let mut curve2 = [Point { x: 0.0, y: 0.0 };2];

    // Generate control vertices for Q'
    for i in 0..curve1.len() {
        curve1[i] = curve[i + 1].subtract(curve[i]).multiply(3.0);
    }

    // Generate control vertices for Q''
    for i in 0..curve2.len() {
        curve2[i] = curve1[i + 1].subtract(curve1[i]).multiply(2.0);
    }

    // Compute Q(u), Q'(u) and Q''(u)
    let pt = evaluate_4(&curve, u);
    let pt1 = evaluate_3(&curve1, u);
    let pt2 = evaluate_2(&curve2, u);
    let diff = pt.subtract(*point);
    let df = pt1.dot(pt1) + diff.dot(pt2);

    // Newton: u = u - f(u) / f'(u)
    if df.abs() < Float::EPSILON {
        u
    } else {
        u - diff.dot(pt1) / df
    }
}

macro_rules! evaluate {
    ($curve:expr, $t:expr) => {{
        // Copy curve
        let mut tmp = *$curve;

        // Triangle computation
        for i in 1..$curve.len() {
            for j in 0..($curve.len() - i) {
                tmp[j] = tmp[j].multiply(1.0 - $t).add(tmp[j + 1].multiply($t));
            }
        }

        tmp[0]
    }};
}

// evaluate the bezier curve at point t
#[inline]
fn evaluate_4(curve: &[Point;4], t: Float) -> Point { let ret = evaluate!(curve, t); ret }
#[inline]
fn evaluate_3(curve: &[Point;3], t: Float) -> Point { let ret = evaluate!(curve, t); ret }
#[inline]
fn evaluate_2(curve: &[Point;2], t: Float) -> Point { let ret = evaluate!(curve, t); ret }

// chord length parametrize the curve points[first..last]
#[inline]
fn chord_length_parametrize(points: &[Point]) -> Vec<Float> {

    let mut u = vec![0.0;points.len()];
    let mut last_dist = 0.0;

    for (prev, (next_id, next)) in points.iter().zip(points.iter().enumerate().skip(1)) {
        let new_dist = last_dist + prev.distance(*next);
        u[next_id] = new_dist;
        last_dist = new_dist;
    }

    u
}

struct FindMaxErrorReturn {
    error: Float,
    index: usize
}

// find maximum squared distance error between real points and curve
#[inline]
fn find_max_error(points: &[Point], curve: &[Point;4], u: &[Float]) -> FindMaxErrorReturn {

    let mut index = points.len() / 2.0 as usize;
    let mut max_dist = 0.0;

    for (i, (p, u)) in points.iter().zip(u.iter()).enumerate() {
        let point_on_curve = evaluate_4(curve, *u);
        let dist = point_on_curve.subtract(*p);
        let dist_squared = dist.x.mul_add(dist.x, dist.y.powi(2)); // compute squared distance

        if dist_squared >= max_dist {
            max_dist = dist_squared;
            index = i;
        }
    }

    FindMaxErrorReturn {
        error: max_dist,
        index: index
    }
}

#[cfg(test)]
mod bench {
    extern crate test;
    use test::{Bencher, black_box};

    #[bench]
    fn bench(b: &mut Bencher) {
        use crate::Point;
        let points = vec![
            Point { x:256.0,                    y:318.0},
            Point { x:258.6666666666667,        y:315.3333333333333},
            Point { x:266.6666666666667,        y:308.6666666666667},
            Point { x:314.0,                    y:274.6666666666667},
            Point { x:389.3333333333333,        y:218.0},
            Point { x:448.6666666666667,        y:176.0},
            Point { x:472.0,                    y:160.66666666666666},
            Point { x:503.3333333333333,        y:145.33333333333334},
            Point { x:516.0,                    y:144.66666666666666},
            Point { x:520.0,                    y:156.66666666666666},
            Point { x:479.3333333333333,        y:220.66666666666666},
            Point { x:392.6666666666667,        y:304.0},
            Point { x:314.0,                    y:376.6666666666667},
            Point { x:253.33333333333334,       y:436.6666666666667},
            Point { x:238.0,                    y:454.6666666666667},
            Point { x:228.66666666666666,       y:468.0},
            Point { x:236.0,                    y:467.3333333333333},
            Point { x:293.3333333333333,        y:428.0},
            Point { x:428.0,                    y:337.3333333333333},
            Point { x:516.6666666666666,        y:283.3333333333333},
            Point { x:551.3333333333334,        y:262.0},
            Point { x:566.6666666666666,        y:253.33333333333334},
            Point { x:579.3333333333334,        y:246.0},
            Point { x:590.0,                    y:241.33333333333334},
            Point { x:566.6666666666666,        y:260.0},
            Point { x:532.0,                    y:290.6666666666667},
            Point { x:516.6666666666666,        y:306.0},
            Point { x:510.6666666666667,        y:313.3333333333333},
            Point { x:503.3333333333333,        y:324.6666666666667},
            Point { x:527.3333333333334,        y:324.6666666666667},
            Point { x:570.6666666666666,        y:313.3333333333333},
            Point { x:614.0,                    y:302.6666666666667},
            Point { x:631.3333333333334,        y:301.3333333333333},
            Point { x:650.0,                    y:300.0},
            Point { x:658.6666666666666,        y:304.0},
            Point { x:617.3333333333334,        y:333.3333333333333},
            Point { x:546.0,                    y:381.3333333333333},
            Point { x:518.6666666666666,        y:400.6666666666667},
            Point { x:505.3333333333333,        y:412.6666666666667},
            Point { x:488.0,                    y:430.6666666666667},
            Point { x:489.3333333333333,        y:435.3333333333333},
            Point { x:570.6666666666666,        y:402.0},
            Point { x:700.0,                    y:328.6666666666667},
            Point { x:799.3333333333334,        y:266.0},
            Point { x:838.0,                    y:240.0},
            Point { x:854.0,                    y:228.66666666666666},
            Point { x:868.0,                    y:218.66666666666666},
            Point { x:879.3333333333334,        y:210.66666666666666},
            Point { x:872.6666666666666,        y:216.0},
            Point { x:860.0,                    y:223.33333333333334},
        ];

        // let points = points.repeat(1);       // 50 points: f64 - 50µs, f32 - 50µs
        // let points = points.repeat(10);      // 500 points: f64 - 830µs, f32 - 781µs
        // let points = points.repeat(100);     // 5.000 points: f64 - 12.7ms, f32 - 1.27ms
        // let points = points.repeat(1000);    // 50.000 points: f64 - 167ms, f32 - 65ms

        b.iter(|| { black_box(crate::simplify_curve(&points, 800.0)); });
    }
}