#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub struct Point { pub x: f64, pub y: f64 }

impl Point {
    #[inline]
    pub fn add(&self, p: Point) -> Point { Point { x: self.x + p.x, y: self.y + p.y } }
    #[inline]
    pub fn subtract(&self, p: Point) -> Point { Point { x: self.x - p.x, y: self.y - p.y } }
    #[inline]
    pub fn multiply(&self, f: f64) -> Point { Point { x: self.x * f, y: self.y * f } }
    #[inline]
    pub fn negate(&self) -> Point { Point { x: -self.x, y: -self.y } }
    #[inline]
    pub fn distance(&self, p: Point) -> f64 { (p.x - self.x).hypot(p.y - self.y) }
    #[inline]
    pub fn dot(&self, p: Point) -> f64 { self.x * p.x + self.y * p.y }
    #[inline]
    pub fn normalize(&self, length: f64) -> Point {
        let current =  self.x.hypot(self.y);
        let scale = if current.abs() > f64::EPSILON { length / current } else { 0.0 };
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
pub fn simplify_curve(points: &[Point], tolerance: f64) -> Vec<Point> {

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
        let mut new_cur_points = vec![points.last().copied().unwrap()];
        new_cur_points.extend(cur_points.drain(..));
        new_cur_points.push(points.first().copied().unwrap());
        cur_points = new_cur_points;
    }

    fit(&cur_points[..], tolerance)
}

#[derive(Copy, Clone)]
struct Split {
    global_range_start: usize,
    global_range_end: usize,
    tan1: Point,
    tan2: Point,
}

#[inline]
fn fit(points: &[Point], tolerance: f64) -> Vec<Point> {

    // To support reducing paths with multiple points in the same place
    // to one segment:
    let mut segments = Vec::new();
    let distances = chord_length_parametrize(points);

    let mut splits_to_eval = vec![Split {
        global_range_start: 0,
        global_range_end: points.len(),
        tan1: points[1].subtract(points[0]),
        tan2: points[points.len() - 2].subtract(points[points.len() - 1]),
    }];

    while let Some(split) = splits_to_eval.pop() {
        let result = fit_cubic(FitCubicParams {
            points: &points[split.global_range_start..split.global_range_end],
            chord_lengths: &distances[split.global_range_start..split.global_range_end],
            segments: &mut segments,
            error: tolerance,
            tan1: split.tan1,
            tan2: split.tan2,
        });

        if let Some(r) = result {
            // Fitting failed -- split at max error point and fit recursively
            let tan_center = points[split.global_range_start + r - 1].subtract(points[split.global_range_start + r + 1]);
            splits_to_eval.extend_from_slice(&[
                Split {
                    global_range_start: split.global_range_start,
                    global_range_end: split.global_range_start + r + 1,
                    tan1: split.tan1,
                    tan2: tan_center,
                },
                Split {
                    global_range_start: split.global_range_start + r,
                    global_range_end: split.global_range_end,
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
    chord_lengths: &'a [f64],
    error: f64,
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
    let u_prime_first = u_prime.first().copied().unwrap();
    let u_prime_last = u_prime.last().copied().unwrap() - u_prime_first;
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
fn generate_bezier(points: &[Point], u_prime: &[f64], tan1: Point, tan2: Point) -> [Point;4] {

    assert!(u_prime.len() > 3);
    assert!(points.len() > 3);
    assert!(u_prime.len() == points.len());

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

    if det_c0_c1.abs() > f64::EPSILON {
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
        alpha1 = if c0.abs() > f64::EPSILON {
            X[0] / c0
        } else if c1.abs() > f64::EPSILON {
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
    let eps = f64::EPSILON * seg_length;
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
fn reparameterize(points: &[Point], u: &mut [f64], curve: &[Point;4]) -> bool {

    points.iter().zip(u.iter_mut()).for_each(|(p, u)| { *u = find_root(curve, p, *u); });

    // Detect if the new parameterization has reordered the points.
    // In that case, we would fit the points of the path in the wrong order.
    !u.windows(2).any(|w| w[1] <= w[0])
}

#[inline]
fn find_root(curve: &[Point;4], point: &Point, u: f64) -> f64 {

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
    if df.abs() < f64::EPSILON {
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
fn evaluate_4(curve: &[Point;4], t: f64) -> Point { let ret = evaluate!(curve, t); ret }
fn evaluate_3(curve: &[Point;3], t: f64) -> Point { let ret = evaluate!(curve, t); ret }
fn evaluate_2(curve: &[Point;2], t: f64) -> Point { let ret = evaluate!(curve, t); ret }

// chord length parametrize the curve points[first..last]
#[inline]
fn chord_length_parametrize(points: &[Point]) -> Vec<f64> {

    let mut u = vec![0.0;points.len()];
    let mut last_dist = 0.0;

    for (prev, (next_id, next)) in points.iter().zip(points.iter().enumerate().skip(1)) {
        let new_dist = last_dist + prev.distance(*next);
        u[next_id] = new_dist;
        last_dist = new_dist;
    }

    u
}

#[derive(Debug, Copy, Clone)]
struct FindMaxErrorReturn {
    error: f64,
    index: usize
}

// find maximum squared distance error between real points and curve
#[inline]
fn find_max_error(points: &[Point], curve: &[Point;4], u: &[f64]) -> FindMaxErrorReturn {

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