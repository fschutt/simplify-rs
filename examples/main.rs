use simplify_rs::Point;
use std::time::Instant;

fn main() {
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

    // unoptimized: 96.1µs
    // noalloc: 86.496µs
    // noloop:
    for _ in 0..100 {
        let now = Instant::now();
        let result = simplify_rs::simplify_curve(&points, 800.0);
        println!("time: {:?}", Instant::now() - now);
    }
/*
    // example solution: https://bit.ly/2UYdxLw
    // note: y is inverted,
    // note: the second and third points denote the "handle in" / "handle out", i.e they have to be added to the first point X / Y
    if result.is_empty() {
        println!("no solution!");
    } else if result.len() == 1 {
        println!("solution:\r\n\tpoint {:?}", result[0]);
    } else if result.len() == 2 {
        println!("solution:\r\n\tline {:?} - {:?}", result[0], result[1]);
    } else {
        println!("solution:");
        for cubic_curve in result.chunks_exact(4) {
            println!("\tcubic curve: [");
            println!("\t\t{:?}", cubic_curve[0]);
            println!("\t\t{:?}", cubic_curve[1]);
            println!("\t\t{:?}", cubic_curve[2]);
            println!("\t\t{:?}", cubic_curve[3]);
            println!("\t]");
        }
    }
*/

}