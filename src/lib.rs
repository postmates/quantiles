//! This crate provides approximate quantiles over data streams in a moderate
//! amount of memory.
//!
//! Order statistics is a rough business. Exact solutions are expensive in terms
//! of memory and computation. Recent literature has advanced approximations but
//! each have fundamental tradeoffs. This crate is intended to be a collection
//! of approxiate algorithms that provide guarantees around space consumption.

#![deny(missing_docs)]

#[cfg(test)]
extern crate quickcheck;

use std::fmt::Debug;

#[derive(Debug,Clone)]
struct Entry<T: Copy> {
    v: T,
    g: usize,
    delta: usize,
}

enum Cmp {
    LT,
    GT,
    EQ,
}

fn cmp<T: PartialOrd + Debug>(l: T, r: T) -> Cmp {
    if l == r {
        Cmp::EQ
    } else if l < r {
        Cmp::LT
    } else {
        Cmp::GT
    }
}

/// This is an implementation of the algorithm presented in Cormode, Korn,
/// Muthukrishnan, Srivastava's paper "Effective Computation of Biased Quantiles
/// over Data Streams". The ambition here is to approximate quantiles on a
/// stream of data without having a boatload of information kept in memory.
///
/// As of this writing you _must_ use the presentation in the IEEE version of
/// the paper. The authors' self-published copy of the paper is incorrect and
/// this implementation will _not_ make sense if you follow along using that
/// version. Only the 'full biased' invariant is used. The 'targeted quantiles'
/// variant of this algorithm is fundamentally flawed, an issue which the
/// authors correct in their "Space- and Time-Efficient Deterministic Algorithms
/// for Biased Quantiles over Data Streams"
pub struct CKMS<T: Copy> {
    n: usize,

    // We follow the 'batch' method of the above paper. In this method,
    // incoming items are buffered in a priority queue, called 'buffer' here,
    // and once insert_threshold items are stored in the buffer it is drained
    // into the 'samples' collection. Insertion will cause some extranious
    // points to be held that can be merged. Once compress_threshold threshold
    // items are buffered the COMPRESS operation merges these extranious points.
    insert_threshold: usize,
    inserts: usize,

    // We aim for the full biased quantiles method. The paper this
    // implementation is based on includes a 'targeted' method but the authors
    // have granted that it is flawed in private communication. As such, all
    // queries for all quantiles will have the same error factor.
    error: f64,

    // We store incoming points in a priority queue and every insert_threshold
    // items drain buffer into samples. This is the 'cursor' method of the
    // paper.
    buffer: Vec<T>,

    // This is the S(n) of the above paper. Entries are stored here and
    // occasionally merged. The outlined implementation uses a linked list but
    // we prefer a Vec for reasons of cache locality at the cost of worse
    // computational complexity.
    samples: Vec<Entry<T>>,
}

impl<T: Copy + PartialOrd + Debug> CKMS<T>
{
    /// Create a new CKMS
    ///
    /// A CKMS is meant to answer quantile queries with a known error bound. If
    /// the error passed here is ε and there have been `n` items inserted into
    /// CKMS then for any quantile query Φ the deviance from the true quantile
    /// will be +/- εΦn.
    ///
    /// For an error ε this structure will require T*(floor(1/(2*ε)) + O(1/ε log
    /// εn)) + f64 + usize + usize words of storage.
    ///
    /// # Examples
    /// ```
    /// use quantiles::CKMS;
    ///
    /// let mut ckms = CKMS::<u16>::new(0.001);
    /// for i in 1..1000 {
    ///     ckms.insert(i as u16);
    /// }
    /// assert_eq!(ckms.query(0.999), Some((998, 998)));
    /// ```
    pub fn new(error: f64) -> CKMS<T> {
        let insert_threshold = 1.0 / (2.0 * error);
        CKMS {
            n: 0,

            error: 0.001,

            insert_threshold: insert_threshold as usize,
            inserts: 0,

            buffer: Vec::<T>::new(),
            samples: Vec::<Entry<T>>::new(),
        }
    }

    /// Insert a T into the CKMS
    ///
    /// Insertion will gradulally shift the approximate quantiles. This
    /// implementation is biased toward fast writes and slower queries. Storage
    /// may grow gradually, as defined in the module-level documentatio, but
    /// will remain bounded.
    pub fn insert(&mut self, v: T) {
        // NOTE This is O(n) but n is relatively small unless ε is very, very
        // small. A clear optimization here is to make insertion sub-linear.
        let mut idx = 0;
        for i in 0..self.buffer.len() {
            if v < self.buffer[i] {
                break;
            } else {
                idx += 1;
            }
        }
        self.buffer.insert(idx, v);
        self.inserts = (self.inserts + 1) % self.insert_threshold;
        if self.inserts == 0 {
            self.flush()
        }
    }

    /// Query CKMS for a ε-approximate quantile
    ///
    /// This function returns an approximation to the true quantile-- +/- εΦn
    /// --for the points inserted. Argument q is valid 0. <= q <= 1.0. The
    /// minimum and maximum quantile, corresponding to 0.0 and 1.0 respectively,
    /// are always known precisely.
    ///
    /// Return
    ///
    /// # Examples
    /// ```
    /// use quantiles::CKMS;
    ///
    /// let mut ckms = CKMS::<u16>::new(0.001);
    /// for i in 0..1000 {
    ///     ckms.insert(i as u16);
    /// }
    ///
    /// assert_eq!(ckms.query(0.0), Some((1, 0)));
    /// assert_eq!(ckms.query(0.998), Some((998, 997)));
    /// assert_eq!(ckms.query(1.0), Some((1000, 999)));
    /// ```
    pub fn query(&mut self, q: f64) -> Option<(usize, T)> {
        self.flush();

        let s = self.samples.len();

        if s == 0 {
            return None;
        }

        let mut r = 0;
        let nphi = q * (self.n as f64);
        for i in 1..s {
            let prev = &self.samples[i - 1];
            let cur = &self.samples[i];

            r += prev.g;

            let lhs = (r + cur.g + cur.delta) as f64;
            let rhs = nphi + ((self.invariant(nphi) as f64) / 2.0);

            if lhs > rhs {
                return Some((r, prev.v));
            }
        }

        let v = self.samples[s - 1].v;
        Some((s, v))
    }

    #[inline]
    fn invariant(&self, r: f64) -> usize {
        let i = (2.0 * self.error * r).floor() as usize;
        if 1 > i {
            1
        } else {
            i
        }
    }

    fn compress(&mut self) {
        if self.samples.len() < 3 {
            return;
        }

        let mut s_mx = self.samples.len() - 1;
        let mut i = 0;
        let mut r = 1;

        loop {
            let cur_g = self.samples[i].g;
            let nxt_v = self.samples[i + 1].v;
            let nxt_g = self.samples[i + 1].g;
            let nxt_delta = self.samples[i + 1].delta;

            if cur_g + nxt_g + nxt_delta <= self.invariant(r as f64) {
                let ent = Entry {
                    v: nxt_v,
                    g: nxt_g + cur_g,
                    delta: nxt_delta,
                };
                self.samples[i] = ent;
                self.samples.remove(i + 1);
                s_mx -= 1;
            } else {
                i += 1;
            }
            r += 1;

            if i == s_mx {
                break;
            }
        }
    }

    fn insert_batch(&mut self) {
        let mut s = self.samples.len();
        while let Some(v) = self.buffer.pop() {
            let mut r = 0;
            if s == 0 {
                self.samples.insert(0,
                                    Entry {
                                        v: v,
                                        g: 1,
                                        delta: 0,
                                    });
                self.n += 1;
                continue;
            }

            let mut idx = 0;
            for i in 0..s {
                let smpl = &self.samples[i];
                match cmp(smpl.v, v) {
                    Cmp::LT => idx += 1,
                    Cmp::EQ | Cmp::GT => break,
                }
                r += smpl.g;
            }
            let delta = if idx == 0 || idx == s {
                0
            } else {
                self.invariant(r as f64) - 1
            };
            self.samples.insert(idx,
                                Entry {
                                    v: v,
                                    g: 1,
                                    delta: delta,
                                });
            self.n += 1;
            s += 1;
        }
    }

    fn flush(&mut self) {
        self.insert_batch();
        self.compress();
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use quickcheck::{QuickCheck, TestResult};
    use std::f64::consts::E;

    // prop: forany phi. (phi*n - f(phi*n, n)/2) =< r_i =< (phi*n + f(phi*n, n)/2)
    #[test]
    fn query_invariant_test() {
        fn query_invariant(f: f64, fs: Vec<i64>) -> TestResult {
            if fs.len() < 1 {
                return TestResult::discard();
            }

            let phi = (1.0 / (1.0 + E.powf(f.abs()))) * 2.0;

            let mut ckms = CKMS::<i64>::new(0.001);
            for f in fs {
                ckms.insert(f);
            }
            ckms.flush();

            match ckms.query(phi) {
                None => TestResult::passed(), // invariant to check here? n*phi + f > 1?
                Some((rank, _)) => {
                    let nphi = phi * (ckms.n as f64);
                    let fdiv2 = (ckms.invariant(nphi) as f64) / 2.0;
                    TestResult::from_bool(((nphi - fdiv2) <= (rank as f64)) ||
                                          ((rank as f64) <= (nphi + fdiv2)))
                }
            }
        }
        QuickCheck::new()
            .tests(10000)
            .max_tests(100000)
            .quickcheck(query_invariant as fn(f64, Vec<i64>) -> TestResult);
    }

    #[test]
    fn insert_test() {
        let mut ckms = CKMS::<f64>::new(0.001);
        for i in 0..2 {
            ckms.insert(i as f64);
        }
        ckms.flush();

        assert_eq!(0.0, ckms.samples[0].v);
        assert_eq!(1.0, ckms.samples[1].v);
    }


    // prop: v_i-1 < v_i =< v_i+1
    #[test]
    fn asc_samples_test() {
        fn asc_samples(fs: Vec<i64>) -> TestResult {
            let mut ckms = CKMS::<i64>::new(0.001);
            let fsc = fs.clone();
            for f in fs {
                ckms.insert(f);
            }
            ckms.flush();

            if ckms.samples.len() == 0 && fsc.len() == 0 {
                return TestResult::passed();
            }
            let mut cur = ckms.samples[0].v;
            for ent in ckms.samples {
                let s = ent.v;
                if s < cur {
                    return TestResult::failed();
                }
                cur = s;
            }
            TestResult::passed()
        }
        QuickCheck::new()
            .tests(10000)
            .max_tests(100000)
            .quickcheck(asc_samples as fn(Vec<i64>) -> TestResult);
    }

    // prop: forall i. g_i + delta_i =< f(r_i, n)
    #[test]
    fn f_invariant_test() {
        fn f_invariant(fs: Vec<i64>) -> TestResult {
            let mut ckms = CKMS::<i64>::new(0.001);
            for f in fs {
                ckms.insert(f);
            }
            ckms.flush();

            let s = ckms.samples.len();
            let mut r = 0;
            for i in 1..s {
                let ref prev = ckms.samples[i - 1];
                let ref cur = ckms.samples[i];

                r += prev.g;

                let res = (cur.g + cur.delta) <= ckms.invariant(r as f64);
                if !res {
                    println!("{:?} <= {:?}", cur.g + cur.delta, ckms.invariant(r as f64));
                    println!("samples: {:?}", ckms.samples);
                    return TestResult::failed();
                }
            }
            TestResult::passed()
        }
        QuickCheck::new()
            .tests(10000)
            .max_tests(100000)
            .quickcheck(f_invariant as fn(Vec<i64>) -> TestResult);
    }

    #[test]
    fn compression_test() {
        let mut ckms = CKMS::<i64>::new(0.1);
        for i in 1..10000 {
            ckms.insert(i);
        }
        ckms.flush();

        let l = ckms.samples.len();
        let n = ckms.n;
        assert_eq!(9999, n);
        assert_eq!(3763, l);
    }

    // prop: post-compression, samples is bounded above by O(1/e log^2 en)
    #[test]
    fn compression_bound_test() {
        fn compression_bound(fs: Vec<i64>) -> TestResult {
            if fs.len() < 15 {
                return TestResult::discard();
            }

            let mut ckms = CKMS::<i64>::new(0.001);
            for f in fs {
                ckms.insert(f);
            }
            ckms.flush();

            let s = ckms.samples.len() as f64;
            let bound = (1.0 / ckms.error) * (ckms.error * (ckms.n as f64)).log10().powi(2);

            if !(s <= bound) {
                println!("error: {:?} n: {:?} log10: {:?}",
                         ckms.error,
                         ckms.n as f64,
                         (ckms.error * (ckms.n as f64)).log10().powi(2));
                println!("{:?} <= {:?}", s, bound);
                return TestResult::failed();
            }
            TestResult::passed()
        }
        QuickCheck::new()
            .tests(10000)
            .max_tests(100000)
            .quickcheck(compression_bound as fn(Vec<i64>) -> TestResult);
    }

    #[test]
    fn test_basics() {
        let mut ckms = CKMS::<i32>::new(0.001);
        for i in 1..1001 {
            ckms.insert(i as i32);
        }

        assert_eq!(ckms.query(0.00), Some((1, 1)));
        assert_eq!(ckms.query(0.05), Some((50, 50)));
        assert_eq!(ckms.query(0.10), Some((100, 100)));
        assert_eq!(ckms.query(0.15), Some((150, 150)));
        assert_eq!(ckms.query(0.20), Some((200, 200)));
        assert_eq!(ckms.query(0.25), Some((250, 250)));
        assert_eq!(ckms.query(0.30), Some((300, 300)));
        assert_eq!(ckms.query(0.35), Some((350, 350)));
        assert_eq!(ckms.query(0.40), Some((400, 400)));
        assert_eq!(ckms.query(0.45), Some((450, 450)));
        assert_eq!(ckms.query(0.50), Some((500, 500)));
        assert_eq!(ckms.query(0.55), Some((550, 550)));
        assert_eq!(ckms.query(0.60), Some((600, 600)));
        assert_eq!(ckms.query(0.65), Some((650, 650)));
        assert_eq!(ckms.query(0.70), Some((700, 700)));
        assert_eq!(ckms.query(0.75), Some((750, 750)));
        assert_eq!(ckms.query(0.80), Some((800, 800)));
        assert_eq!(ckms.query(0.85), Some((850, 850)));
        assert_eq!(ckms.query(0.90), Some((900, 900)));
        assert_eq!(ckms.query(0.95), Some((950, 950)));
        assert_eq!(ckms.query(0.99), Some((990, 990)));
        assert_eq!(ckms.query(1.00), Some((1000, 1000)));
    }

    #[test]
    fn test_basics_float() {
        let mut ckms = CKMS::<f64>::new(0.001);
        for i in 1..1001 {
            ckms.insert(i as f64);
        }

        assert_eq!(ckms.query(0.00), Some((1, 1.0)));
        assert_eq!(ckms.query(1.00), Some((1000, 1000.0)));
    }
}
