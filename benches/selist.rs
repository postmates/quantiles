#![feature(test)]

extern crate quantiles;
extern crate test;

mod selist {
    use quantiles::util::Xorshift;
    use quantiles::util::selist::SEList;
    use test::Bencher;

    macro_rules! generate_tests {
        ($t:ty, $fn:ident, $s:expr) => {
            #[bench]
            fn $fn(b: &mut Bencher) {
                let mut xshft = Xorshift::new(1972);
                b.iter(|| {
                    let mut selist = SEList::<$t>::new(8192);
                    for _ in 0..$s {
                        let elem = xshft.next_val() as $t;
                        let idx = match selist.search(&elem) {
                            Ok(i) | Err(i) => i,
                        };
                        let _ = selist.insert(idx, elem);
                    }
                });
            }
        }
    }

    macro_rules! generate_baseline_tests {
        ($t:ty, $fn:ident, $s:expr) => {
            #[bench]
            fn $fn(b: &mut Bencher) {
                let mut xshft = Xorshift::new(1972);
                b.iter(|| {
                    let mut v = Vec::with_capacity(1024);
                    for _ in 0..$s {
                        let elem = xshft.next_val() as $t;
                        let idx = match v.binary_search(&elem) {
                            Ok(i) | Err(i) => i,
                        };
                        v.insert(idx, elem)
                    }
                });
            }
        }
    }

    mod u16 {
        use super::*;

        generate_tests!(u16, insert_100, 100);
        generate_tests!(u16, insert_1000, 1000);
        generate_tests!(u16, insert_10000, 10_000);
        generate_tests!(u16, insert_65535, 65_535);

        #[bench]
        fn primed_insertion(b: &mut Bencher) {
            let mut xshft = Xorshift::new(1972);
            let mut selist = SEList::<u16>::new(65_536);
            for _ in 0..1_000_000 {
                let elem = xshft.next_val() as u16;
                let idx = match selist.search(&elem) {
                    Ok(i) | Err(i) => i,
                };
                let _ = selist.insert(idx, elem);
            }

            b.iter(|| {
                let elem = xshft.next_val() as u16;
                let idx = match selist.search(&elem) {
                    Ok(i) | Err(i) => i,
                };
                let _ = selist.insert(idx, elem);
            });
        }

        mod baseline {
            use super::*;

            generate_baseline_tests!(u16, insert_100, 100);
            generate_baseline_tests!(u16, insert_1000, 1000);
            generate_baseline_tests!(u16, insert_10000, 10_000);
            generate_baseline_tests!(u16, insert_65535, 65_535);

            #[bench]
            fn primed_insertion(b: &mut Bencher) {
                let mut xshft = Xorshift::new(1972);
                let mut v = Vec::with_capacity(1024);
                for _ in 0..1_000_000 {
                    let elem = xshft.next_val() as u16;
                    let idx = match v.binary_search(&elem) {
                        Ok(i) | Err(i) => i,
                    };
                    v.insert(idx, elem)
                }

                b.iter(|| {
                    let elem = xshft.next_val() as u16;
                    let idx = match v.binary_search(&elem) {
                        Ok(i) | Err(i) => i,
                    };
                    v.insert(idx, elem)
                });
            }
        }
    }

    mod u32 {
        use super::*;

        generate_tests!(u32, insert_100, 100);
        generate_tests!(u32, insert_1000, 1000);
        generate_tests!(u32, insert_10000, 10_000);
        generate_tests!(u32, insert_100000, 100_000);
        mod baseline {
            use super::*;

            generate_baseline_tests!(u32, insert_100, 100);
            generate_baseline_tests!(u32, insert_1000, 1000);
            generate_baseline_tests!(u32, insert_10000, 10_000);
            generate_baseline_tests!(u32, insert_100000, 100_000);
        }
    }
}
