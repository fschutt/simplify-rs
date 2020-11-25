use simplify_rs::{BezierCurve, Point, BezierCurveItem};

fn main() {
    let points = vec![
        Point { x:256,                    y:318},
        Point { x:258.6666666666667,      y:315.3333333333333},
        Point { x:266.6666666666667,      y:308.6666666666667},
        Point { x:314,                    y:274.6666666666667},
        Point { x:389.3333333333333,      y:218},
        Point { x:448.6666666666667,      y:176},
        Point { x:472,                    y:160.66666666666666},
        Point { x:503.3333333333333,      y:145.33333333333334},
        Point { x:516,                    y:144.66666666666666},
        Point { x:520,                    y:156.66666666666666},
        Point { x:479.3333333333333,      y:220.66666666666666},
        Point { x:392.6666666666667,      y:304},
        Point { x:314,                    y:376.6666666666667},
        Point { x:253.33333333333334,     y:436.6666666666667},
        Point { x:238,                    y:454.6666666666667},
        Point { x:228.66666666666666,     y:468},
        Point { x:236,                    y:467.3333333333333},
        Point { x:293.3333333333333,      y:428},
        Point { x:428,                    y:337.3333333333333},
        Point { x:516.6666666666666,      y:283.3333333333333},
        Point { x:551.3333333333334,      y:262},
        Point { x:566.6666666666666,      y:253.33333333333334},
        Point { x:579.3333333333334,      y:246},
        Point { x:590,                    y:241.33333333333334},
        Point { x:566.6666666666666,      y:260},
        Point { x:532,                    y:290.6666666666667},
        Point { x:516.6666666666666,      y:306},
        Point { x:510.6666666666667,      y:313.3333333333333},
        Point { x:503.3333333333333,      y:324.6666666666667},
        Point { x:527.3333333333334,      y:324.6666666666667},
        Point { x:570.6666666666666,      y:313.3333333333333},
        Point { x:614,                    y:302.6666666666667},
        Point { x:631.3333333333334,      y:301.3333333333333},
        Point { x:650,                    y:300},
        Point { x:658.6666666666666,      y:304},
        Point { x:617.3333333333334,      y:333.3333333333333},
        Point { x:546,                    y:381.3333333333333},
        Point { x:518.6666666666666,      y:400.6666666666667},
        Point { x:505.3333333333333,      y:412.6666666666667},
        Point { x:488,                    y:430.6666666666667},
        Point { x:489.3333333333333,      y:435.3333333333333},
        Point { x:570.6666666666666,      y:402},
        Point { x:700,                    y:328.6666666666667},
        Point { x:799.3333333333334,      y:266},
        Point { x:838,                    y:240},
        Point { x:854,                    y:228.66666666666666},
        Point { x:868,                    y:218.66666666666666},
        Point { x:879.3333333333334,      y:210.66666666666666},
        Point { x:872.6666666666666,      y:216},
        Point { x:860,                    y:223.33333333333334},
    ];

    let lines = points.iter().zip(points.iter().skip(1)).map(|(prev, next)| {
        BezierCurveItem::Line((*prev, *next))
    }).collect::<Vec<_>>();

    let curve = BezierCurve { items: lines };

    println!("original: {:#?}", curve);

    let result = simplify_rs::simplify_curve(curve, 800.0);

    // example solution: https://bit.ly/2UYdxLw
    // note: y is inverted,
    // note: the second and third points denote the "handle in" / "handle out", i.e they have to be added to the first point X / Y
    println!("expected: {:#?}", BezierCurve {
        items: vec![
            BezierCurveItem::QuadraticCurve((Point { x: 256.0, y: 256.0 },Point { x: 0.0, y: 0.0 },Point { x: 35.41715, y: 35.41715 })),
            BezierCurveItem::QuadraticCurve((Point { x: 389.33333, y: 389.33333 },Point { x: -38.37766, y: -38.37766 },Point { x: 27.05067, y: 27.05067 })),
            BezierCurveItem::QuadraticCurve((Point { x: 472.0, y: 472.0 },Point { x: -29.24663, y: -29.24663 },Point { x: 196.25843, y: 196.25843 })),
            BezierCurveItem::QuadraticCurve((Point { x: 23.08, y: 238.0 },Point { x: 3.59779, y: 3.59779 },Point { x: -12.66477, y: -12.66477 })),
            BezierCurveItem::QuadraticCurve((Point { x: 293.33333, y: 293.33333 },Point { x: -16.95396, y: -16.95396 },Point { x: 89.70388, y: 89.70388 })),
            BezierCurveItem::QuadraticCurve((Point { x: 566.66667, y: 566.66667 },Point { x: -94.06971, y: -94.06971 },Point { x: 74.39514, y: 74.39514 })),
            BezierCurveItem::QuadraticCurve((Point { x: 527.33333, y: 527.33333 },Point { x: -76.47158, y: -76.47158 },Point { x: 35.36912, y: 35.36912 })),
            BezierCurveItem::QuadraticCurve((Point { x: 631.33333, y: 631.33333 },Point { x: -36.05963, y: -36.05963 },Point { x: 9.12936, y: 9.12936 })),
            BezierCurveItem::QuadraticCurve((Point { x: 658.66667, y: 658.66667 },Point { x: 6.4074, y: 6.4074 },Point { x: -20.00968, y: -20.00968 })),
            BezierCurveItem::QuadraticCurve((Point { x: 570.66667, y: 570.66667 },Point { x: -206.55945, y: -206.55945 },Point { x: 47.90984, y: 47.90984 })),
            BezierCurveItem::QuadraticCurve((Point { x: 860.0, y: 860.0 },Point { x: 18.83642, y: 18.83642 },Point { x: 0.0, y: 0.0 })),
        ]
    });

    println!("simplified: {:#?}", result);
}