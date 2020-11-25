pub use beziercurve_wkt::*;

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
pub fn simplify_curve(path: BezierCurve, tolerance: f32) -> Option<BezierCurve> {
    PathFitter::new(path).fit(tolerance)
}

struct PathFitter {
    curve: BezierCurve,
}

impl PathFitter {
    const fn new(curve: BezierCurve) -> Self {
        Self { curve }
    }

    fn fit(&self, tolerance: f32) -> Option<BezierCurve> {


            /*
        var segments = new PathFitter(this).fit(tolerance || 2.5);
        if (segments)
            this.setSegments(segments);
        return !!segments;
            */
            None
    }
}