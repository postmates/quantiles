#![feature(test)]

extern crate quantiles;
extern crate test;

use self::test::Bencher;
use quantiles::ckms::CKMS;

fn xorshift(seed: u64) -> (u64, u32) {
    // implementation inspired by
    // https://github.com/astocko/xorshift/blob/master/src/splitmix64.rs
    use std::num::Wrapping as w;
    
    let mut z = w(seed) + w(0x9E3779B97F4A7C15_u64);
    let nxt_seed = z.0;
    z = (z ^ (z >> 30)) * w(0xBF58476D1CE4E5B9_u64);
    z = (z ^ (z >> 27)) * w(0x94D049BB133111EB_u64);
    (nxt_seed, (z ^ (z >> 31)).0 as u32)
}

macro_rules! generate_tests {
    ($t:ty, $fn:ident, $s:expr) => {
        #[bench]
        fn $fn(b: &mut Bencher) {
            let (mut seed, mut val) = xorshift(1972);
            b.iter(|| {
                let mut ckms = CKMS::<$t>::new(0.001);
                for _ in 0..$s {
                    ckms.insert(val);
                    let res = xorshift(seed);
                    seed = res.0;
                    val = res.1;
                }
            });
        }
    }
}

// mod u16 {
//     use super::*;

//     generate_tests!(u16, bench_insert_100, 100);
//     generate_tests!(u16, bench_insert_1000, 1000);
//     generate_tests!(u16, bench_insert_10000, 10_000);
//     //    generate_tests!(u16, bench_insert_65535, 65_535);
// }

mod u32 {
    use super::*;

    generate_tests!(u32, bench_insert_100, 100);
    generate_tests!(u32, bench_insert_1000, 1000);
    generate_tests!(u32, bench_insert_10000, 10_000);
    generate_tests!(u32, bench_insert_100000, 100_000);
//     generate_tests!(u32, bench_insert_1000000, 1_000_000);
//     generate_tests!(u32, bench_insert_10000000, 10_000_000);
//     generate_tests!(u32, bench_insert_100000000, 100_000_000);
}
