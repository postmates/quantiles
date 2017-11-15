#![feature(test)]

extern crate quantiles;
extern crate test;

use self::test::Bencher;
use quantiles::ckms::CKMS;

macro_rules! generate_tests {
    ($t:ty, $fn:ident, $s:expr) => {
        #[bench]
        fn $fn(b: &mut Bencher) {
            b.iter(|| {
                let mut ckms = CKMS::<$t>::new(0.001);
                for i in 0..$s {
                    ckms.insert(i);
                }
            });
        }
    }
}

mod u16 {
    use super::*;

    generate_tests!(u16, bench_insert_100, 100);
    generate_tests!(u16, bench_insert_1000, 1000);
    generate_tests!(u16, bench_insert_10000, 10_000);
    //    generate_tests!(u16, bench_insert_65535, 65_535);
}

// mod u32 {
//     use super::*;

//     generate_tests!(u32, bench_insert_100, 100);
//     generate_tests!(u32, bench_insert_1000, 1000);
//     generate_tests!(u32, bench_insert_10000, 10_000);
//     generate_tests!(u32, bench_insert_100000, 100_000);
//     generate_tests!(u32, bench_insert_1000000, 1_000_000);
//     generate_tests!(u32, bench_insert_10000000, 10_000_000);
//     generate_tests!(u32, bench_insert_100000000, 100_000_000);
// }
