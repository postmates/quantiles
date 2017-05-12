#![feature(test)]

extern crate test;
extern crate quantiles;

use self::test::Bencher;
use quantiles::ckms::CKMS;

#[bench]
fn bench_sequential_insert(b: &mut Bencher) {
    b.iter(|| {
               let mut ckms = CKMS::<i32>::new(0.001);
               for i in 0..10000 {
                   ckms.insert(i);
               }
               ckms.query(0.999);
           });
}

#[bench]
fn bench_inverted_insert(b: &mut Bencher) {
    b.iter(|| {
               let seq = (0..10000).rev();
               let mut ckms = CKMS::<i32>::new(0.001);
               for i in seq {
                   ckms.insert(i);
               }
               ckms.query(0.999);
           });
}
