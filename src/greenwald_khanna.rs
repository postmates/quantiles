//! Greenwald Khanna calculates epsilon-approximate quantiles.
//! If the desired quantile is phi, the epsilon-approximate
//! quantile is any element in the range of elements that rank
//! between `lbound((phi-epsilon) x N)` and `lbound((phi+epsilon) x N)`
//!
//! terminology from the paper:
//!
//!   * S: set of observations
//!   * n: number of observations in S
//!   * v[i]: observation i in S
//!   * r: rank of observation in S from 1 to n.
//!   * `r_min(v[i])`: lower bound on rank r of v[i]
//!   * `r_max(v[i])`: upper bound on rank r of v[i]
//!   * `g[i] = r_min(v[i]) - r_min(v[i - 1])`
//!   * `delta[i] = r_max(v[i]) - r_min(v[i])`
//!   * `t[i] = tuple(v[i], g[i], delta[i])`
//!   * phi: quantile as a real number in the range [0,1]
//!   * r: ubound(phi * n)
//!
//! identities:
//!
//! * `r_min(v[i]) = forall j<=i sum of g[j]`
//! * `r_max(v[i]) = ( forall j<=i sum of g[j] ) + delta[i]`
//! * g[i] + delta[i] - 1 is an upper bound on the total number of observations
//! * between v[i] and v[i-1]
//! * sum of g[i] = n
//!
//! results:
//!
//! * `max_i(g[i] + delta[i]) <= 2 * epsilon * n`
//! * a tuple is full if g[i] + delta[i] = floor(2 * epsilon * n)
//!
//! `@inproceedings{Greenwald:2001:SOC:375663.375670,
//!       author = {Greenwald, Michael and Khanna, Sanjeev},
//!       title = {Space-efficient Online Computation of Quantile Summaries},
//! booktitle = {Proceedings of the 2001 ACM SIGMOD International
//! Conference
//!                    on Management of Data},
//!       series = {SIGMOD '01},
//!       year = {2001},
//!       isbn = {1-58113-332-4},
//!       location = {Santa Barbara, California, USA},
//!       pages = {58--66},
//!       numpages = {9},
//!       url = {http://doi.acm.org/10.1145/375663.375670},
//!       doi = {10.1145/375663.375670},
//!       acmid = {375670},
//!       publisher = {ACM},
//!       address = {New York, NY, USA},
//!     }`
//!
//! # Examples
//!
//! ```
//! use quantiles::greenwald_khanna::*;
//!
//! let epsilon = 0.01;
//!
//! let mut stream = Stream::new(epsilon);
//!
//! let n = 1001;
//! for i in 1..n {
//!     stream.insert(i);
//! }
//! let in_range = |phi: f64, value: u32| {
//!   let lower = ((phi - epsilon) * (n as f64)) as u32;
//!   let upper = ((phi + epsilon) * (n as f64)) as u32;
//!   (epsilon > phi || lower <= value) && value <= upper
//! };
//! assert!(in_range(0f64, *stream.quantile(0f64)));
//! assert!(in_range(0.1f64, *stream.quantile(0.1f64)));
//! assert!(in_range(0.2f64, *stream.quantile(0.2f64)));
//! assert!(in_range(0.3f64, *stream.quantile(0.3f64)));
//! assert!(in_range(0.4f64, *stream.quantile(0.4f64)));
//! assert!(in_range(1f64, *stream.quantile(1f64)));
//! ```

use std::cmp;
use std::ops::AddAssign;

/// Locates the proper position of v in a vector vs
/// such that when v is inserted at position i,
/// it is less then the element at i+1 if any,
/// and greater than or equal to the element at i-1 if any.
pub fn find_insert_pos<T>(vs: &[T], v: &T) -> usize
where
    T: Ord,
{
    if vs.len() <= 10 {
        return find_insert_pos_linear(vs, v);
    }

    let middle = vs.len() / 2;
    let pivot = &vs[middle];

    if v < pivot {
        find_insert_pos(&vs[0..middle], v)
    } else {
        middle + find_insert_pos(&vs[middle..], v)
    }
}

/// Locates the proper position of v in a vector vs
/// such that when v is inserted at position i,
/// it is less then the element at i+1 if any,
/// and greater than or equal to the element at i-1 if any.
/// Works by scanning the slice from start to end.
pub fn find_insert_pos_linear<T>(vs: &[T], v: &T) -> usize
where
    T: Ord,
{
    for (i, vi) in vs.iter().enumerate() {
        if v < vi {
            return i;
        }
    }

    vs.len()
}

/// 3-tuple of a value v[i], g[i] and delta[i].
#[derive(Eq, Ord, Debug)]
pub struct Tuple<T>
where
    T: Ord,
{
    /// v[i], an observation in the set of observations
    pub v: T,

    /// the difference between the rank lowerbounds of t[i] and t[i-1]
    /// g = r_min(v[i]) - r_min(v[i - 1])
    pub g: usize,

    /// the difference betweeh the rank upper and lower bounds for this tuple
    pub delta: usize,
}

impl<T> Tuple<T>
where
    T: Ord,
{
    /// Creates a new instance of a Tuple
    pub fn new(v: T, g: usize, delta: usize) -> Tuple<T> {
        Tuple { v, g, delta }
    }
}

impl<T> PartialEq for Tuple<T>
where
    T: Ord,
{
    fn eq(&self, other: &Self) -> bool {
        self.v == other.v
    }
}

impl<T> PartialOrd for Tuple<T>
where
    T: Ord,
{
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        self.v.partial_cmp(&other.v)
    }
}

/// The summary S of the observations seen so far.
#[derive(Debug)]
pub struct Stream<T>
where
    T: Ord,
{
    /// An ordered sequence of the selected observations
    summary: Vec<Tuple<T>>,

    /// The error factor
    epsilon: f64,

    /// The number of observations
    n: usize,
}

impl<T> Stream<T>
where
    T: Ord,
{
    /// Creates a new instance of a Stream
    pub fn new(epsilon: f64) -> Stream<T> {
        Stream {
            summary: vec![],
            epsilon,
            n: 0,
        }
    }

    /// Locates the correct position in the summary data set
    /// for the observation v, and inserts a new tuple (v,1,floor(2en))
    /// If v is the new minimum or maximum, then instead insert
    /// tuple (v,1,0).
    pub fn insert(&mut self, v: T) {
        let mut t = Tuple::new(v, 1, 0);

        let pos = find_insert_pos(&self.summary, &t);

        if pos != 0 && pos != self.summary.len() {
            t.delta = (2f64 * self.epsilon * (self.n as f64).floor()) as usize;
        }

        self.summary.insert(pos, t);

        self.n += 1;

        if self.should_compress() {
            self.compress();
        }
    }

    /// Compute the epsilon-approximate phi-quantile
    /// from the summary data structure.
    pub fn quantile(&self, phi: f64) -> &T {
        assert!(!self.summary.is_empty());
        assert!(phi >= 0f64 && phi <= 1f64);

        let r = (phi * self.n as f64).floor() as usize;
        let en = (self.epsilon * self.n as f64) as usize;

        let first = &self.summary[0];

        let mut prev = &first.v;
        let mut prev_rmin = first.g;

        for t in self.summary.iter().skip(1) {
            let rmax = prev_rmin + t.g + t.delta;

            if rmax > r + en {
                return prev;
            }

            prev_rmin += t.g;
            prev = &t.v;
        }

        prev
    }

    fn should_compress(&self) -> bool {
        let period = (1f64 / (2f64 * self.epsilon)).floor() as usize;

        self.n % period == 0
    }

    fn compress(&mut self) {
        let s = self.s();
        for i in (1..(s - 1)).rev() {
            if self.can_delete(i) {
                self.delete(i);
            }
        }
    }

    fn can_delete(&self, i: usize) -> bool {
        assert!(self.summary.len() >= 2);
        assert!(i < self.summary.len() - 1);

        let t = &self.summary[i];
        let tnext = &self.summary[i + 1];
        let p = self.p();

        let safety_property = t.g + tnext.g + tnext.delta < p;

        let optimal = Self::band(t.delta, p) <= Self::band(tnext.delta, p);

        safety_property && optimal
    }

    /// Remove the ith tuple from the summary.
    /// Panics if i is not in the range [0,summary.len() - 1)
    /// Only permitted if g[i] + g[i+1] + delta[i+1] < 2 * epsilon * n
    fn delete(&mut self, i: usize) {
        assert!(self.summary.len() >= 2);
        assert!(i < self.summary.len() - 1);

        let t = self.summary.remove(i);
        let tnext = &mut self.summary[i];

        tnext.g += t.g;
    }

    /// Compute which band a delta lies in.
    fn band(delta: usize, p: usize) -> usize {
        assert!(p >= delta);

        let diff = p - delta + 1;

        (diff as f64).log(2f64).floor() as usize
    }

    /// Calculate p = 2epsilon * n
    pub fn p(&self) -> usize {
        (2f64 * self.epsilon * (self.n as f64)).floor() as usize
    }

    /// The number of observations inserted into the stream.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Indication of the space usage of the summary data structure
    /// Returns the number of tuples in the summary
    /// data structure.
    pub fn s(&self) -> usize {
        self.summary.len()
    }
}

impl<T: Ord> AddAssign for Stream<T> {
    fn add_assign(&mut self, rhs: Self) {
        // The GK algorithm is a bit unclear about it, but we need to adjust the statistics during the
        // merging. The main idea is that samples that come from one side will suffer from the lack of
        // precision of the other.
        // As a concrete example, take two QuantileSummaries whose samples (value, g, delta) are:
        // `a = [(0, 1, 0), (20, 99, 0)]` and `b = [(10, 1, 0), (30, 49, 0)]`
        // This means `a` has 100 values, whose minimum is 0 and maximum is 20,
        // while `b` has 50 values, between 10 and 30.
        // The resulting samples of the merge will be:
        // a+b = [(0, 1, 0), (10, 1, ??), (20, 99, ??), (30, 49, 0)]
        // The values of `g` do not change, as they represent the minimum number of values between two
        // consecutive samples. The values of `delta` should be adjusted, however.
        // Take the case of the sample `10` from `b`. In the original stream, it could have appeared
        // right after `0` (as expressed by `g=1`) or right before `20`, so `delta=99+0-1=98`.
        // In the GK algorithm's style of working in terms of maximum bounds, one can observe that the
        // maximum additional uncertainty over samples comming from `b` is `max(g_a + delta_a) =
        // floor(2 * eps_a * n_a)`. Likewise, additional uncertainty over samples from `a` is
        // `floor(2 * eps_b * n_b)`.
        // Only samples that interleave the other side are affected. That means that samples from
        // one side that are lesser (or greater) than all samples from the other side are just copied
        // unmodifed.
        // If the merging instances have different `relativeError`, the resulting instance will cary
        // the largest one: `eps_ab = max(eps_a, eps_b)`.
        // The main invariant of the GK algorithm is kept:
        // `max(g_ab + delta_ab) <= floor(2 * eps_ab * (n_a + n_b))` since
        // `max(g_ab + delta_ab) <= floor(2 * eps_a * n_a) + floor(2 * eps_b * n_b)`
        // Finally, one can see how the `insert(x)` operation can be expressed as `merge([(x, 1, 0])`

        let mut merged_summary = Vec::with_capacity(self.summary.len() + rhs.summary.len());
        let merged_epsilon = self.epsilon.max(rhs.epsilon);
        let merged_n = self.n + rhs.n;
        let additional_self_delta = (2. * rhs.epsilon * rhs.n as f64).floor() as usize;
        let additional_rhs_delta = (2. * self.epsilon * self.n as f64).floor() as usize;

        // Do a merge of two sorted lists until one of the lists is fully consumed
        let mut self_samples = std::mem::replace(&mut self.summary, Vec::new())
            .into_iter()
            .peekable();
        let mut rhs_samples = rhs.summary.into_iter().peekable();
        let mut started_self = false;
        let mut started_rhs = false;
        while let (Some(self_sample), Some(rhs_sample)) = (self_samples.peek(), rhs_samples.peek())
        {
            // Detect next sample
            let (next_sample, additional_delta) = if self_sample.v < rhs_sample.v {
                started_self = true;
                (
                    self_samples.next().unwrap(),
                    if started_rhs {
                        additional_self_delta
                    } else {
                        0
                    },
                )
            } else {
                started_rhs = true;
                (
                    rhs_samples.next().unwrap(),
                    if started_self {
                        additional_rhs_delta
                    } else {
                        0
                    },
                )
            };

            // Insert it
            let next_sample = Tuple {
                v: next_sample.v,
                g: next_sample.g,
                delta: next_sample.delta + additional_delta,
            };
            merged_summary.push(next_sample);
        }

        // Copy the remaining samples from the rhs list
        // (by construction, at most one `while` loop will run)
        merged_summary.extend(self_samples);
        merged_summary.extend(rhs_samples);

        self.summary = merged_summary;
        self.epsilon = merged_epsilon;
        self.n = merged_n;
        self.compress();
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::ops::Range;

    #[test]
    fn test_find_insert_pos() {
        let mut vs = vec![];
        for v in 0..10 {
            vs.push(v);
        }

        for v in 0..10 {
            assert_eq!(find_insert_pos_linear(&vs, &v), v + 1);
        }
    }

    fn get_quantile_for_range(r: &Range<u32>, phi: f64) -> u32 {
        (phi * ((r.end - 1) - r.start) as f64).floor() as u32 + r.start
    }

    fn get_quantile_bounds_for_range(r: Range<u32>, phi: f64, epsilon: f64) -> (u32, u32) {
        let lower = get_quantile_for_range(&r, (phi - epsilon).max(0f64));
        let upper = get_quantile_for_range(&r, phi + epsilon);

        (lower, upper)
    }

    fn quantile_in_bounds(r: Range<u32>, s: &Stream<u32>, phi: f64, epsilon: f64) -> bool {
        let approx_quantile = *s.quantile(phi);
        let (lower, upper) = get_quantile_bounds_for_range(r, phi, epsilon);

        // println!("approx_quantile={} lower={} upper={} phi={} epsilon={}",
        // approx_quantile, lower, upper, phi, epsilon);

        approx_quantile >= lower && approx_quantile <= upper
    }

    #[test]
    fn test_basics() {
        let epsilon = 0.01;

        let mut stream = Stream::new(epsilon);

        for i in 1..1001 {
            stream.insert(i);
        }

        for phi in 0..100 {
            assert!(quantile_in_bounds(
                1..1001,
                &stream,
                (phi as f64) / 100f64,
                epsilon
            ));
        }
    }

    #[test]
    fn test_add_assign() {
        let epsilon = 0.01;

        let mut stream = Stream::new(epsilon);
        let mut stream2 = Stream::new(epsilon);

        for i in 0..1000 {
            stream.insert(2 * i);
            stream2.insert(2 * i + 1);
        }

        for phi in 0..100 {
            assert!(quantile_in_bounds(
                0..2000,
                &stream,
                (phi as f64) / 100f64,
                epsilon
            ));
        }
    }

    quickcheck! {
        fn find_insert_pos_log_equals_find_insert_pos_linear(vs: Vec<i32>) -> bool {
            let mut vs = vs;
            vs.sort();

            for v in -100..100 {
                if find_insert_pos(&vs, &v) != find_insert_pos_linear(&vs, &v) {
                    return false;
                }
            }

            true
        }

        fn test_gk(vs: Vec<u32>) -> bool {
            let mut s = Stream::new(0.25);

            for v in vs {
                s.insert(v);
            }

            true
        }
    }
}
