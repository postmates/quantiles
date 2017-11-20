#![feature(test)]

extern crate quantiles;
extern crate test;

mod ckms {
    #[derive(Debug, Clone, Copy)]
    pub struct Xorshift {
        seed: u64,
    }

    impl Xorshift {
        pub fn new(seed: u64) -> Xorshift {
            Xorshift { seed: seed }
        }

        pub fn next_val(&mut self) -> u32 {
            // implementation inspired by
            // https://github.com/astocko/xorshift/blob/master/src/splitmix64.rs
            use std::num::Wrapping as w;

            let mut z = w(self.seed) + w(0x9E37_79B9_7F4A_7C15_u64);
            let nxt_seed = z.0;
            z = (z ^ (z >> 30)) * w(0xBF58_476D_1CE4_E5B9_u64);
            z = (z ^ (z >> 27)) * w(0x94D0_49BB_1331_11EB_u64);
            self.seed = nxt_seed;
            u32::from((z ^ (z >> 31)).0 as u16)
        }
    }

    use quantiles::ckms::CKMS;
    use test::Bencher;

    macro_rules! generate_tests {
        ($t:ty, $fn:ident, $s:expr) => {
            #[bench]
            fn $fn(b: &mut Bencher) {
                let mut xshft = Xorshift::new(1972);
                b.iter(|| {
                    let mut ckms = CKMS::<$t>::new(0.001);
                    for _ in 0..$s {
                        let val = xshft.next_val();
                        ckms.insert(val as $t);
                    }
                });
            }
        }
    }

    macro_rules! generate_primed_tests {
        ($t:ty, $fn:ident, $s:expr) => {
            #[bench]
            fn $fn(b: &mut Bencher) {
                let mut xshft = Xorshift::new(1972);
                let mut ckms = CKMS::<$t>::new(0.001);
                for _ in 0..1_000_000 {
                    let elem = xshft.next_val() as $t;
                    ckms.insert(elem);
                }
                
                b.iter(|| {
                    let elem = xshft.next_val() as $t;
                    ckms.insert(elem);
                });
            }
        }
    }

    mod u16 {
        use super::*;

        generate_tests!(u16, bench_insert_100, 100);
        generate_tests!(u16, bench_insert_1000, 1000);
        generate_tests!(u16, bench_insert_10000, 10_000);
        generate_tests!(u16, bench_insert_65535, 65_535);

        generate_primed_tests!(u16, bench_primed_100, 100);
        generate_primed_tests!(u16, bench_primed_1000, 1000);
        generate_primed_tests!(u16, bench_primed_10000, 10_000);
        generate_primed_tests!(u16, bench_primed_65535, 65_535);
    }

    mod u32 {
        use super::*;

        generate_tests!(u32, bench_insert_100, 100);
        generate_tests!(u32, bench_insert_1000, 1000);
        generate_tests!(u32, bench_insert_10000, 10_000);
        generate_tests!(u32, bench_insert_100000, 100_000);

        generate_primed_tests!(u32, bench_primed_100, 100);
        generate_primed_tests!(u32, bench_primed_1000, 1000);
        generate_primed_tests!(u32, bench_primed_10000, 10_000);
        generate_primed_tests!(u32, bench_primed_65535, 65_535);
    }

}
