#![feature(test)]

extern crate quantiles;
extern crate test;

mod ckms {
    use quantiles::ckms::CKMS;
    use quantiles::util::Xorshift;
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

    mod u16 {
        use super::*;

        generate_tests!(u16, bench_insert_100, 100);
        generate_tests!(u16, bench_insert_1000, 1000);
        generate_tests!(u16, bench_insert_10000, 10_000);
        generate_tests!(u16, bench_insert_65535, 65_535);
    }

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

}
