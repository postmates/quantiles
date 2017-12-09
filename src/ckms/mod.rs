//! This is an implementation of the algorithm presented in Cormode, Korn,
//! Muthukrishnan, Srivastava's paper "Effective Computation of Biased Quantiles
//! over Data Streams". The ambition here is to approximate quantiles on a
//! stream of data without having a boatload of information kept in memory.
//!
//! As of this writing you _must_ use the presentation in the IEEE version of
//! the paper. The authors' self-published copy of the paper is incorrect and
//! this implementation will _not_ make sense if you follow along using that
//! version. Only the 'full biased' invariant is used. The 'targeted quantiles'
//! variant of this algorithm is fundamentally flawed, an issue which the
//! authors correct in their "Space- and Time-Efficient Deterministic Algorithms
//! for Biased Quantiles over Data Streams"
use std;
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, Sub};

mod entry;
mod store;

use self::store::Store;

/// A structure to provide approximate quantiles queries in bounded memory and
/// with bounded error.
#[derive(Clone, PartialEq, Debug)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct CKMS<T>
where
    T: Copy + PartialEq,
{
    n: usize,

    // We follow the 'batch' method of the above paper. In this method,
    // incoming items are buffered in a priority queue, called 'buffer' here,
    // and once insert_threshold items are stored in the buffer it is drained
    // into the 'samples' collection. Insertion will cause some extranious
    // points to be held that can be merged. Once compress_threshold threshold
    // items are buffered the COMPRESS operation merges these extranious points.
    insert_threshold: usize,
    inserts: usize,

    // This is the S(n) of the above paper. Entries are stored here and
    // occasionally merged. The outlined implementation uses a linked list but
    // we prefer a Vec for reasons of cache locality at the cost of worse
    // computational complexity.
    samples: Store<T>,

    sum: Option<T>,
    cma: Option<f64>,
    last_in: Option<T>,
}

impl<T> AddAssign for CKMS<T>
where
    T: Copy
        + Add<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + PartialOrd
        + Debug
        + std::convert::Into<f64>,
{
    fn add_assign(&mut self, rhs: CKMS<T>) {
        self.last_in = rhs.last_in;
        self.sum = match (self.sum, rhs.sum) {
            (None, None) => None,
            (None, Some(y)) => Some(y),
            (Some(x), None) => Some(x),
            (Some(x), Some(y)) => Some(x.add(y)),
        };
        self.cma = match (self.cma, rhs.cma) {
            (None, None) => None,
            (None, Some(y)) => Some(y),
            (Some(x), None) => Some(x),
            (Some(x), Some(y)) => {
                let x_n: f64 = self.n as f64;
                let y_n: f64 = rhs.n as f64;
                Some(((x_n * x) + (y_n * y)) / (x_n + y_n))
            }
        };
        self.n += rhs.n;
        for inner in rhs.samples.data {
            for v in inner.data.iter().map(|x| x.v) {
                self.samples.insert(v);
            }
        }
        self.compress();
    }
}

impl<
    T: Copy
        + PartialOrd
        + Debug
        + Add<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + std::convert::Into<f64>,
> CKMS<T> {
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
    /// use quantiles::ckms::CKMS;
    ///
    /// let mut ckms = CKMS::<u32>::new(0.001);
    /// for i in 1..1001 {
    ///     ckms.insert(i as u32);
    /// }
    /// assert_eq!(ckms.query(0.0), Some((1, 1)));
    /// assert_eq!(ckms.query(0.998), Some((998, 998)));
    /// assert_eq!(ckms.query(0.999), Some((999, 999)));
    /// assert_eq!(ckms.query(1.0), Some((1000, 1000)));
    /// ```
    ///
    /// `error` must but a value between 0 and 1, exclusive of both extremes. If
    /// you input an error <= 0.000_000_000_1 CKMS will assign an error of
    /// 0.000_000_000_1. Likewise, if your error is >= 1.0 CKMS will assign an
    /// error of 0.99.
    pub fn new(error: f64) -> CKMS<T> {
        let error = if error <= 0.000_000_000_1 {
            0.000_000_000_1
        } else if error >= 1.0 {
            0.99
        } else {
            error
        };
        let insert_threshold = 1.0 / (2.0 * error);
        let insert_threshold: usize = if insert_threshold < 1.0 {
            1
        } else {
            insert_threshold as usize
        };
        CKMS {
            n: 0,

            insert_threshold: insert_threshold,
            inserts: 0,

            samples: Store::new(2048, error),

            last_in: None,
            sum: None,
            cma: None,
        }
    }

    /// Return the last element added to the CKMS
    ///
    /// # Example
    /// ```
    /// use quantiles::ckms::CKMS;
    ///
    /// let mut ckms = CKMS::new(0.1);
    /// ckms.insert(1.0);
    /// ckms.insert(2.0);
    /// ckms.insert(3.0);
    /// assert_eq!(Some(3.0), ckms.last());
    /// ```
    pub fn last(&self) -> Option<T> {
        self.last_in
    }

    /// Return the sum of the elements added to the CKMS
    ///
    /// # Example
    /// ```
    /// use quantiles::ckms::CKMS;
    ///
    /// let mut ckms = CKMS::new(0.1);
    /// ckms.insert(1.0);
    /// ckms.insert(2.0);
    /// ckms.insert(3.0);
    /// assert_eq!(Some(6.0), ckms.sum());
    /// ```
    pub fn sum(&self) -> Option<T> {
        self.sum
    }

    /// Return the cummulative moving average of the elements added to the CKMS
    ///
    /// # Example
    /// ```
    /// use quantiles::ckms::CKMS;
    ///
    /// let mut ckms = CKMS::new(0.1);
    /// ckms.insert(0.0);
    /// ckms.insert(100.0);
    ///
    /// assert_eq!(Some(50.0), ckms.cma());
    /// ```
    pub fn cma(&self) -> Option<f64> {
        self.cma
    }

    /// Return the guaranteed error bound of this CKMS
    ///
    /// # Example
    /// ```
    /// use quantiles::ckms::CKMS;
    ///
    /// let mut ckms: CKMS<f64> = CKMS::new(0.1);
    /// assert_eq!(0.1, ckms.error_bound());
    /// ```
    pub fn error_bound(&self) -> f64 {
        self.samples.error
    }

    /// Insert a T into the CKMS
    ///
    /// Insertion will gradulally shift the approximate quantiles. This
    /// implementation is biased toward fast writes and slower queries. Storage
    /// may grow gradually, as defined in the module-level documentation, but
    /// will remain bounded.
    pub fn insert(&mut self, v: T) {
        self.sum = self.sum.map_or(Some(v), |s| Some(s.add(v)));
        self.last_in = Some(v);
        self.n += 1;
        let v_f64: f64 = v.into();
        self.cma = self.cma
            .map_or(Some(v_f64), |s| Some(s + ((v_f64 - s) / (self.n as f64))));
        self.samples.insert(v);
        self.inserts = (self.inserts + 1) % self.insert_threshold;
        if self.inserts == 0 {
            self.compress()
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
    /// use quantiles::ckms::CKMS;
    ///
    /// let mut ckms = CKMS::<u32>::new(0.001);
    /// for i in 0..1000 {
    ///     ckms.insert(i as u32);
    /// }
    ///
    /// assert_eq!(ckms.query(0.0), Some((1, 0)));
    /// assert_eq!(ckms.query(0.998), Some((998, 997)));
    /// assert_eq!(ckms.query(1.0), Some((1000, 999)));
    /// ```
    pub fn query(&self, q: f64) -> Option<(usize, T)> {
        self.samples.query(q)
    }

    /// Query CKMS for the count of its points
    ///
    /// This function returns the total number of points seen over the lifetime
    /// of the datastructure, _not_ the number of points currently stored in the
    /// structure.
    ///
    /// # Examples
    /// ```
    /// use quantiles::ckms::CKMS;
    ///
    /// let mut ckms = CKMS::<u32>::new(0.001);
    /// for i in 0..1000 {
    ///     ckms.insert(i as u32);
    /// }
    ///
    /// assert_eq!(ckms.count(), 1000);
    /// ```
    pub fn count(&self) -> usize {
        self.n
    }

    /// Retrieve a representative vector of points
    ///
    /// This function returns a represenative sample of points from the
    /// CKMS. Doing so consumes the CKMS.
    ///
    /// # Examples
    /// ```
    /// use quantiles::ckms::CKMS;
    ///
    /// let mut ckms = CKMS::<u32>::new(0.1);
    /// for i in 0..10 {
    ///     ckms.insert(i as u32);
    /// }
    ///
    /// assert_eq!(ckms.into_vec(), vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    /// ```
    pub fn into_vec(self) -> Vec<T> {
        let mut res = vec![];
        for inner in self.samples.data {
            for v in inner.data.iter().map(|x| x.v) {
                res.push(v);
            }
        }
        res
    }

    fn compress(&mut self) {
        self.samples.compress();
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use quickcheck::{QuickCheck, TestResult};
    use std::f64::consts::E;
    use ckms::store::invariant;

    fn percentile(data: &Vec<f64>, prcnt: f64) -> f64 {
        let idx = (prcnt * (data.len() as f64)) as usize;
        return data[idx];
    }

    #[test]
    fn test_cma() {
        fn inner(data: Vec<f64>, err: f64) -> TestResult {
            if data.is_empty() {
                return TestResult::discard();
            } else if !(err >= 0.0) || !(err <= 1.0) {
                return TestResult::discard();
            }

            let mut ckms = CKMS::<f64>::new(err);
            for d in &data {
                ckms.insert(*d);
            }

            let sum: f64 = data.iter().sum();
            let expected_mean: f64 = sum / (data.len() as f64);
            let mean = ckms.cma();
            assert!(mean.is_some());

            assert!((expected_mean - mean.unwrap()).abs() < err);
            return TestResult::passed();
        }
        QuickCheck::new().quickcheck(inner as fn(Vec<f64>, f64) -> TestResult);
    }

    #[test]
    fn test_cma_add_assign() {
        fn inner(l_data: Vec<f64>, r_data: Vec<f64>, err: f64) -> TestResult {
            if !(err >= 0.0) || !(err <= 1.0) {
                return TestResult::discard();
            }

            let mut l_ckms = CKMS::<f64>::new(err);
            for d in &l_data {
                l_ckms.insert(*d);
            }
            let mut r_ckms = CKMS::<f64>::new(err);
            for d in &r_data {
                r_ckms.insert(*d);
            }

            let sum: f64 = l_data.iter().chain(r_data.iter()).sum();
            let expected_mean: f64 = sum / ((l_data.len() + r_data.len()) as f64);
            l_ckms += r_ckms;
            let mean = l_ckms.cma();
            if mean.is_some() {
                assert!((expected_mean - mean.unwrap()).abs() < err);
            }
            return TestResult::passed();
        }
        QuickCheck::new().quickcheck(inner as fn(Vec<f64>, Vec<f64>, f64) -> TestResult);
    }

    #[test]
    fn error_nominal_test() {
        fn inner(mut data: Vec<f64>, prcnt: f64) -> TestResult {
            data.sort_by(|a, b| a.partial_cmp(b).unwrap());
            if !(prcnt >= 0.0) || !(prcnt <= 1.0) {
                return TestResult::discard();
            } else if data.len() < 1 {
                return TestResult::discard();
            }
            let err = 0.001;

            let mut ckms = CKMS::<f64>::new(err);
            for d in &data {
                ckms.insert(*d);
            }

            if let Some((_, v)) = ckms.query(prcnt) {
                debug_assert!(
                    (v - percentile(&data, prcnt)) < err,
                    "v: {} | percentile: {} | prcnt: {} | data: {:?}",
                    v,
                    percentile(&data, prcnt),
                    prcnt,
                    data
                );
                TestResult::passed()
            } else {
                TestResult::failed()
            }
        }
        QuickCheck::new().quickcheck(inner as fn(Vec<f64>, f64) -> TestResult);
    }

    #[test]
    fn error_nominal_with_merge_test() {
        fn inner(lhs: Vec<f64>, rhs: Vec<f64>, prcnt: f64, err: f64) -> TestResult {
            if !(prcnt >= 0.0) || !(prcnt <= 1.0) {
                return TestResult::discard();
            } else if !(err >= 0.0) || !(err <= 1.0) {
                return TestResult::discard();
            } else if (lhs.len() + rhs.len()) < 1 {
                return TestResult::discard();
            }
            if lhs.is_empty() || rhs.is_empty() {
                return TestResult::discard();
            }
            let mut data = lhs.clone();
            data.append(&mut rhs.clone());
            data.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let err = 0.001;

            let mut ckms = CKMS::<f64>::new(err);
            for d in &lhs {
                ckms.insert(*d);
            }
            let mut ckms_rhs = CKMS::<f64>::new(err);
            for d in &rhs {
                ckms_rhs.insert(*d);
            }
            ckms += ckms_rhs;

            if let Some((_, v)) = ckms.query(prcnt) {
                debug_assert!(
                    (v - percentile(&data, prcnt)) < err,
                    "v: {} | percentile: {} | prcnt: {} | data: {:?}",
                    v,
                    percentile(&data, prcnt),
                    prcnt,
                    data
                );
                TestResult::passed()
            } else {
                TestResult::failed()
            }
        }
        QuickCheck::new().quickcheck(inner as fn(Vec<f64>, Vec<f64>, f64, f64) -> TestResult);
    }

    #[test]
    fn n_invariant_test() {
        fn n_invariant(fs: Vec<i32>) -> bool {
            let l = fs.len();

            let mut ckms = CKMS::<i32>::new(0.001);
            for f in fs {
                ckms.insert(f);
            }

            ckms.count() == l
        }
        QuickCheck::new().quickcheck(n_invariant as fn(Vec<i32>) -> bool);
    }

    #[test]
    fn count_sum_test() {
        fn inner(lhs: Vec<i32>, rhs: Vec<i32>) -> TestResult {
            let mut lhs_ckms = CKMS::<i32>::new(0.001);
            for f in lhs {
                lhs_ckms.insert(f);
            }

            let mut rhs_ckms = CKMS::<i32>::new(0.001);
            for f in rhs {
                rhs_ckms.insert(f);
            }

            let expected_count = lhs_ckms.count() + rhs_ckms.count();
            lhs_ckms += rhs_ckms;

            assert_eq!(lhs_ckms.count(), expected_count);
            TestResult::passed()
        }
        QuickCheck::new().quickcheck(inner as fn(Vec<i32>, Vec<i32>) -> TestResult);
    }

    #[test]
    fn add_assign_test() {
        fn inner(pair: (i32, i32)) -> bool {
            let mut lhs = CKMS::<i32>::new(0.001);
            lhs.insert(pair.0);
            let mut rhs = CKMS::<i32>::new(0.001);
            rhs.insert(pair.1);

            let expected: i32 = pair.0 + pair.1;
            lhs += rhs;

            if let Some(x) = lhs.sum() {
                if x == expected {
                    if let Some(y) = lhs.last() {
                        y == pair.1
                    } else {
                        false
                    }
                } else {
                    false
                }
            } else {
                false
            }
        }
        QuickCheck::new().quickcheck(inner as fn((i32, i32)) -> bool);
    }

    // prop: forany phi. (phi*n - f(phi*n, n)/2) =< r_i =< (phi*n + f(phi*n, n)/2)
    #[test]
    fn query_invariant_test() {
        fn query_invariant(f: f64, fs: Vec<i32>) -> TestResult {
            let error = 0.001;
            if fs.len() < 1 {
                return TestResult::discard();
            }

            let phi = (1.0 / (1.0 + E.powf(f.abs()))) * 2.0;

            let mut ckms = CKMS::<i32>::new(error);
            for f in fs {
                ckms.insert(f);
            }

            match ckms.query(phi) {
                None => TestResult::passed(), // invariant to check here? n*phi + f > 1?
                Some((rank, _)) => {
                    let nphi = phi * (ckms.n as f64);
                    let fdiv2 = (invariant(nphi, error) as f64) / 2.0;
                    TestResult::from_bool(
                        ((nphi - fdiv2) <= (rank as f64)) || ((rank as f64) <= (nphi + fdiv2)),
                    )
                }
            }
        }
        QuickCheck::new().quickcheck(query_invariant as fn(f64, Vec<i32>) -> TestResult);
    }

    #[test]
    fn insert_test() {
        let mut ckms = CKMS::<f64>::new(0.001);
        for i in 0..2 {
            ckms.insert(i as f64);
        }

        assert_eq!(0.0, ckms.samples[0].v);
        assert_eq!(1.0, ckms.samples[1].v);
    }


    // prop: v_i-1 < v_i =< v_i+1
    #[test]
    fn asc_samples_test() {
        fn asc_samples(fs: Vec<i32>) -> TestResult {
            let mut ckms = CKMS::<i32>::new(0.001);
            let fsc = fs.clone();
            for f in fs {
                ckms.insert(f);
            }

            if ckms.samples.len() == 0 && fsc.len() == 0 {
                return TestResult::passed();
            }
            let mut cur = ckms.samples[0].v;
            for ent in ckms.samples.iter() {
                let s = ent.v;
                if s < cur {
                    return TestResult::failed();
                }
                cur = s;
            }
            TestResult::passed()
        }
        QuickCheck::new().quickcheck(asc_samples as fn(Vec<i32>) -> TestResult);
    }

    // prop: forall i. g_i + delta_i =< f(r_i, n)
    #[test]
    fn f_invariant_test() {
        fn f_invariant(fs: Vec<i32>) -> TestResult {
            let error = 0.001;
            let mut ckms = CKMS::<i32>::new(error);
            for f in fs {
                ckms.insert(f);
            }

            let s = ckms.samples.len();
            let mut r = 0;
            for i in 1..s {
                let ref prev = ckms.samples[i - 1];
                let ref cur = ckms.samples[i];

                r += prev.g;

                let res = (cur.g + cur.delta) <= invariant(r as f64, error);
                if !res {
                    println!(
                        "{:?} <= {:?}",
                        cur.g + cur.delta,
                        invariant(r as f64, error)
                    );
                    println!("samples: {:?}", ckms.samples);
                    return TestResult::failed();
                }
            }
            TestResult::passed()
        }
        QuickCheck::new().quickcheck(f_invariant as fn(Vec<i32>) -> TestResult);
    }

    #[test]
    fn compression_test() {
        let mut ckms = CKMS::<i32>::new(0.1);
        for i in 1..10000 {
            ckms.insert(i);
        }

        let l = ckms.samples.len();
        let n = ckms.count();
        assert_eq!(9999, n);
        assert_eq!(320, l);
    }

    // prop: post-compression, samples is bounded above by O(1/e log^2 en)
    #[test]
    fn compression_bound_test() {
        fn compression_bound(fs: Vec<i32>) -> TestResult {
            if fs.len() < 15 {
                return TestResult::discard();
            }

            let mut ckms = CKMS::<i32>::new(0.001);
            for f in fs {
                ckms.insert(f);
            }
            ckms.compress();

            let s = ckms.samples.len() as i64;
            let bound = ((1.0 / ckms.error_bound()) * (ckms.error_bound() * (ckms.count() as f64)).log10().powi(2)).ceil() as i64;

            // We have to choose an arbitrary, lowish constant for bound
            // invalidation buffer. This is because I don't have a precise
            // boundary. 1024 samples worth of slop isn't bad, I guess.
            if !(s <= bound) && !((s - bound).abs() < 1_024) { 
                println!(
                    "error: {:?} n: {:?} log10: {:?}",
                    ckms.error_bound(),
                    ckms.count() as f64,
                    (ckms.error_bound() * (ckms.count() as f64)).log10().powi(2)
                );
                println!("{:?} <= {:?}", s, bound);
                return TestResult::failed();
            }
            TestResult::passed()
        }
        QuickCheck::new().quickcheck(compression_bound as fn(Vec<i32>) -> TestResult);
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
}
