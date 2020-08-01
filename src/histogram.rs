//! 'histogram' approximates a distribution calculation by counting the number
//! of times samples fall into pre-configured bins. This implementation does not
//! require bins to be equally sized. The user must specify upper bounds on bins
//! via `Bounds`. The implementation includes a +Inf bound automatically.
//!
//! Storage cost is proportional to the number of bins. The implementation is
//! biased in favor of writes.

use std::cmp;
use std::fmt;
use std::ops;
use std::slice;

#[derive(Debug, Copy, Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
/// The upper bound for each `Histogram` bins. The user is responsible for
/// determining effective bins for their use-case.
pub enum Bound<T>
where
    T: Copy,
{
    /// A finite upper bound.
    Finite(T),
    /// A positively infinite upper bound. We cheat when doing ordering and say
    /// that PosInf == PosInf. This is not strictly true but it's true enough
    /// for us.
    PosInf,
}

impl<T> PartialOrd for Bound<T>
where
    T: Copy + cmp::PartialOrd,
{
    fn partial_cmp(&self, other: &Bound<T>) -> Option<cmp::Ordering> {
        match *self {
            Bound::Finite(ref x) => match *other {
                Bound::Finite(y) => x.partial_cmp(&y),
                Bound::PosInf => Some(cmp::Ordering::Less),
            },
            Bound::PosInf => match *other {
                Bound::Finite(_) => Some(cmp::Ordering::Greater),
                Bound::PosInf => Some(cmp::Ordering::Equal),
            },
        }
    }
}

impl<T> PartialEq for Bound<T>
where
    T: Copy + cmp::PartialEq,
{
    fn eq(&self, other: &Bound<T>) -> bool {
        match *self {
            Bound::Finite(ref x) => match *other {
                Bound::Finite(y) => y.eq(x),
                Bound::PosInf => false,
            },
            Bound::PosInf => match *other {
                Bound::Finite(_) => false,
                Bound::PosInf => true,
            },
        }
    }
}

impl<T> ops::AddAssign for Histogram<T>
where
    T: Copy + cmp::PartialOrd + fmt::Debug + ops::Add<Output = T>,
{
    fn add_assign(&mut self, rhs: Histogram<T>) {
        let lhs_sum = self.sum;
        let rhs_sum = rhs.sum;
        let sum = match (lhs_sum, rhs_sum) {
            (None, None) => None,
            (None, Some(y)) => Some(y),
            (Some(x), None) => Some(x),
            (Some(x), Some(y)) => Some(x + y),
        };
        self.sum = sum;
        self.count += rhs.count;
        for (i, bnd) in rhs.iter().enumerate() {
            assert_eq!(self.bins[i].0, bnd.0);
            self.bins[i].1 += bnd.1;
        }
    }
}

/// A binning histogram of unequal, pre-defined bins
///
/// This implementation performs summation over `T`. It's possible that this
/// summation will overflow, a crash condition in Rust. Unfortunately there's no
/// generic saturating / checked add over a generic. Please take care when
/// inserting into Histogram for small `T`s.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct Histogram<T>
where
    T: Copy,
{
    count: usize,
    sum: Option<T>,
    bins: Vec<(Bound<T>, usize)>,
}

/// Struct to implement Iterator over Histogram
#[derive(Debug)]
pub struct Iter<'a, T>
where
    T: 'a + Copy,
{
    rx: slice::Iter<'a, (Bound<T>, usize)>,
}

impl<'a, T> Iterator for Iter<'a, T>
where
    T: Copy,
{
    type Item = &'a (Bound<T>, usize);

    fn next(&mut self) -> Option<Self::Item> {
        self.rx.next()
    }
}

#[derive(Debug, Copy, Clone)]
/// Construction errors
///
/// `Histogram` is a little finicky when you construct it. We signal errors out
/// to the user with this enumeration.
pub enum Error {
    /// The bounds given to Histogram are empty. We need bounds.
    BoundsEmpty,
    /// The bounds given to Histogram are not sorted. They must be.
    BoundsNotSorted,
}

fn is_sorted<T>(bounds: &[T]) -> bool
where
    T: cmp::PartialOrd + fmt::Debug,
{
    let mut prev = None;
    for i in bounds {
        if prev.is_none() {
            prev = Some(i);
            continue;
        }
        let p: &T = prev.unwrap();
        match i.partial_cmp(p) {
            Some(cmp::Ordering::Less) => {
                return false;
            }
            _ => {
                prev = Some(i);
            }
        }
    }
    true
}

impl<T> Histogram<T>
where
    T: Copy + cmp::PartialOrd + fmt::Debug,
{
    /// Create a new Histogram
    ///
    /// This Histogram is a binning histogram of unequal bins. The user is
    /// responsible for defining the upper bounds of bins. Users are able to
    /// query bin counts without exact bins but should be aware that the results
    /// will only be approximate unless the explicit bin is used. See `total_*`
    /// functions for details.
    ///
    /// # Examples
    /// ```
    /// use quantiles::histogram::{Bound, Histogram};
    ///
    /// let mut histo = Histogram::<u64>::new(vec![10, 256, 1987,
    /// 1990]).unwrap();
    /// for i in 0..2048 {
    ///     histo.insert(i as u64);
    /// }
    ///
    /// assert_eq!(histo.total_above(Bound::Finite(0)), 2048);
    /// assert_eq!(histo.total_above(Bound::Finite(11)), 2037);
    /// assert_eq!(histo.total_above(Bound::Finite(10)), 2037);
    /// assert_eq!(histo.total_between(Bound::Finite(1987),
    /// Bound::Finite(1990)), 3);
    /// assert_eq!(histo.total_below(Bound::PosInf), 2048);
    /// ```
    pub fn new(bounds: Vec<T>) -> Result<Histogram<T>, Error> {
        if bounds.is_empty() {
            return Err(Error::BoundsEmpty);
        }
        if !is_sorted(&bounds) {
            return Err(Error::BoundsNotSorted);
        }

        let mut bins: Vec<(Bound<T>, usize)> = bounds
            .into_iter()
            .map(|x| (Bound::Finite(x), usize::min_value()))
            .collect();
        let cap: (Bound<T>, usize) = (Bound::PosInf, 0);
        bins.push(cap);

        Ok(Histogram {
            count: 0,
            sum: None,
            bins,
        })
    }

    /// Insert a T into the Histogram
    ///
    /// Insertion will search for the appropriate bin and increase the counter
    /// found there. If two bins `a` and `b` form a bin with `a < b` then `X`
    /// will be placed into that bin if `a < X <= b`.
    ///
    /// # Examples
    /// ```
    /// use quantiles::histogram::{Bound, Histogram};
    ///
    /// let mut histo = Histogram::<u64>::new(vec![10, 100]).unwrap();
    /// histo.insert(99 as u64);
    /// histo.insert(100 as u64);
    ///
    /// assert_eq!(histo.total_between(Bound::Finite(10), Bound::Finite(100)),
    /// 2);
    /// ```
    pub fn insert(&mut self, value: T)
    where
        T: ops::Add<Output = T>,
    {
        self.sum = match self.sum {
            None => Some(value),
            Some(x) => Some(x + value),
        };

        let mut idx = 0;
        let val_bound = Bound::Finite(value);
        for &(ref bound, _) in &self.bins {
            match bound.partial_cmp(&val_bound) {
                Some(cmp::Ordering::Greater) | Some(cmp::Ordering::Equal) => {
                    break;
                }
                Some(cmp::Ordering::Less) | None => idx += 1,
            }
        }
        self.bins[idx].1 += 1;
        self.count += 1;
    }

    /// Returns the total number of items 'stored' in the histogram
    ///
    /// # Examples
    /// ```
    /// use quantiles::histogram::{Bound, Histogram};
    ///
    /// let mut histo = Histogram::<u64>::new(vec![10, 256, 1987,
    /// 1990]).unwrap();
    /// for i in 0..2048 {
    ///     histo.insert(i as u64);
    /// }
    ///
    /// assert_eq!(histo.count(), 2048);
    /// ```
    pub fn count(&self) -> usize {
        self.count
    }

    /// Returns the sum of the items 'stored' in the histogram
    ///
    /// # Examples
    /// ```
    /// use quantiles::histogram::Histogram;
    ///
    /// let mut histo = Histogram::<u64>::new(vec![10, 256, 1987,
    /// 1990]).unwrap();
    ///
    /// assert_eq!(histo.sum(), None);
    ///
    /// for i in 0..2048 {
    ///     histo.insert(i as u64);
    /// }
    ///
    /// assert_eq!(histo.sum(), Some(2096128));
    /// ```
    pub fn sum(&self) -> Option<T> {
        self.sum
    }

    /// Total number of items below supplied upper_bound
    ///
    /// # Examples
    /// ```
    /// use quantiles::histogram::{Bound, Histogram};
    ///
    /// let mut histo = Histogram::<u64>::new(vec![10, 256, 1987,
    /// 1990]).unwrap();
    /// for i in 0..2048 {
    ///     histo.insert(i as u64);
    /// }
    ///
    /// assert_eq!(histo.total_below(Bound::PosInf), 2048);
    /// ```
    pub fn total_below(&self, upper: Bound<T>) -> usize {
        let mut count = 0;
        for &(ref bound, cnt) in &self.bins {
            if bound > &upper {
                break;
            } else {
                count += cnt;
            }
        }
        count
    }

    /// Total number of items above supplied lower_bound
    ///
    /// # Examples
    /// ```
    /// use quantiles::histogram::{Bound, Histogram};
    ///
    /// let mut histo = Histogram::<u64>::new(vec![10, 256, 1987,
    /// 1990]).unwrap();
    /// for i in 0..2048 {
    ///     histo.insert(i as u64);
    /// }
    ///
    /// assert_eq!(histo.total_above(Bound::Finite(0)), 2048);
    /// assert_eq!(histo.total_above(Bound::Finite(11)), 2037);
    /// assert_eq!(histo.total_above(Bound::Finite(10)), 2037);
    /// ```
    pub fn total_above(&self, lower: Bound<T>) -> usize {
        let mut count = 0;
        for &(ref bound, cnt) in &self.bins {
            if bound <= &lower {
                continue;
            }
            count += cnt;
        }
        count
    }

    /// Total number of items between [lower_bound, upper_bound)
    ///
    /// # Examples
    /// ```
    /// use quantiles::histogram::{Bound, Histogram};
    ///
    /// let mut histo = Histogram::<u64>::new(vec![10, 256, 1987,
    /// 1990]).unwrap();
    /// for i in 0..2048 {
    ///     histo.insert(i as u64);
    /// }
    ///
    /// assert_eq!(histo.total_between(Bound::Finite(1987),
    /// Bound::Finite(1990)), 3);
    /// ```
    pub fn total_between(&self, lower: Bound<T>, upper: Bound<T>) -> usize {
        if lower >= upper {
            return 0;
        }
        let mut count = 0;
        for &(ref bound, cnt) in &self.bins {
            if bound > &lower && bound <= &upper {
                count += cnt;
            }
        }
        count
    }

    /// Iterate over the bounds and counts of bounds
    /// # Examples
    /// ```
    /// use quantiles::histogram::{Bound, Histogram};
    ///
    /// let mut histo = Histogram::<u64>::new(vec![10, 256, 1987,
    /// 1990]).unwrap();
    /// for i in 0..2048 {
    ///     histo.insert(i as u64);
    /// }
    ///
    /// let expected: Vec<(Bound<u64>, usize)> = vec![(Bound::Finite(10), 11),
    /// (Bound::Finite(256), 246), (Bound::Finite(1987), 1731),
    /// (Bound::Finite(1990), 3), (Bound::PosInf, 57)];
    /// let actual: Vec<(Bound<u64>, usize)> = histo.iter().map(|x|
    /// *x).collect();
    /// assert_eq!(expected[0], actual[0]);
    /// assert_eq!(expected[1], actual[1]);
    /// assert_eq!(expected[2], actual[2]);
    /// assert_eq!(expected[3], actual[3]);
    /// assert_eq!(expected[4], actual[4]);
    /// ```
    pub fn iter(&self) -> Iter<T> {
        Iter {
            rx: self.bins.iter(),
        }
    }

    /// Convert a Histogram into an array of tuples
    ///
    /// # Examples
    /// ```
    /// use quantiles::histogram::{Bound, Histogram};
    ///
    /// let mut histo = Histogram::<u64>::new(vec![10, 256, 1987,
    /// 1990]).unwrap();
    /// for i in 0..2048 {
    ///     histo.insert(i as u64);
    /// }
    ///
    /// let expected: Vec<(Bound<u64>, usize)> = vec![(Bound::Finite(10), 11),
    /// (Bound::Finite(256), 246), (Bound::Finite(1987), 1731),
    /// (Bound::Finite(1990), 3), (Bound::PosInf, 57)];
    /// let actual: Vec<(Bound<u64>, usize)> = histo.into_vec();
    /// assert_eq!(expected[0], actual[0]);
    /// assert_eq!(expected[1], actual[1]);
    /// assert_eq!(expected[2], actual[2]);
    /// assert_eq!(expected[3], actual[3]);
    /// assert_eq!(expected[4], actual[4]);
    /// ```
    pub fn into_vec(self) -> Vec<(Bound<T>, usize)> {
        self.iter().cloned().collect()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use quickcheck::{QuickCheck, TestResult};

    #[test]
    fn test_addassign() {
        fn inner(mut bounds: Vec<f64>, lpyld: Vec<f64>, rpyld: Vec<f64>) -> TestResult {
            if bounds.is_empty() {
                return TestResult::discard();
            }
            bounds.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let mut x = Histogram::new(bounds.clone()).unwrap();
            for i in lpyld {
                x.insert(i);
            }
            let mut y = Histogram::new(bounds).unwrap();
            for i in rpyld {
                y.insert(i);
            }

            let mut res = x.clone();
            res += y.clone();

            assert_eq!(res.count(), x.count() + y.count());
            if res.sum().is_some() {
                match (x.sum().is_some(), y.sum().is_some()) {
                    (true, true) => {
                        assert!(
                            (res.sum().unwrap() - (x.sum().unwrap() + y.sum().unwrap())).abs()
                                < f64::EPSILON
                        );
                    }
                    (false, true) => {
                        assert!((res.sum().unwrap() - y.sum().unwrap()).abs() < f64::EPSILON);
                    }
                    (true, false) => {
                        assert!((res.sum().unwrap() - x.sum().unwrap()).abs() < f64::EPSILON);
                    }
                    (false, false) => unreachable!(),
                }
            } else {
                assert!(x.sum().is_none());
                assert!(y.sum().is_none());
            }

            let mut x_iter = x.iter();
            let mut y_iter = y.iter();
            for &(bound, count) in res.iter() {
                let next_x = x_iter.next().unwrap();
                let next_y = y_iter.next().unwrap();

                assert_eq!(bound, next_x.0);
                assert_eq!(bound, next_y.0);

                assert_eq!(count, next_x.1 + next_y.1)
            }
            TestResult::passed()
        }
        QuickCheck::new().quickcheck(inner as fn(Vec<f64>, Vec<f64>, Vec<f64>) -> TestResult);
    }

    macro_rules! generate_tests {
        ($m:ident, $t:ty) => {
            mod $m {
                use super::*;

                #[test]
                fn test_is_sorted() {
                    fn inner(mut pyld: Vec<$t>) -> TestResult {
                        pyld.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        assert!(is_sorted(&pyld));
                        TestResult::passed()
                    }
                    QuickCheck::new().quickcheck(inner as fn(Vec<$t>) -> TestResult);
                }

                #[test]
                fn test_insertion_count() {
                    fn inner(mut bounds: Vec<$t>, pyld: Vec<$t>) -> TestResult {
                        if bounds.is_empty() {
                            return TestResult::discard();
                        }
                        bounds.sort_by(|a, b| a.partial_cmp(b).unwrap());

                        let mut histo = Histogram::new(bounds).unwrap();
                        let total = pyld.len();
                        for i in pyld.clone() {
                            histo.insert(i);
                        }

                        // confirm that the histogram holds the correct number of items
                        assert_eq!(total, histo.count());

                        TestResult::passed()
                    }
                    QuickCheck::new().quickcheck(inner as fn(Vec<$t>, Vec<$t>) -> TestResult);
                }

                #[test]
                fn test_insertion_sum() {
                    fn inner(mut bounds: Vec<$t>, pyld: Vec<$t>) -> TestResult {
                        if bounds.is_empty() {
                            return TestResult::discard();
                        }
                        bounds.sort_by(|a, b| a.partial_cmp(b).unwrap());

                        let mut histo = Histogram::new(bounds).unwrap();
                        let mut sum: $t = 0 as $t;
                        for i in pyld.clone() {
                            sum += i;
                            histo.insert(i);
                        }

                        // confirm that the histogram holds the correct sum of items
                        if pyld.is_empty() {
                            assert_eq!(None, histo.sum());
                        } else {
                            assert_eq!(Some(sum), histo.sum());
                        }

                        TestResult::passed()
                    }
                    QuickCheck::new().quickcheck(inner as fn(Vec<$t>, Vec<$t>) -> TestResult);
                }

                #[test]
                fn test_insertion_below_count() {
                    fn inner(mut bounds: Vec<$t>, mut pyld: Vec<$t>) -> TestResult {
                        if bounds.is_empty() {
                            return TestResult::discard();
                        }
                        bounds.sort_by(|a, b| a.partial_cmp(b).unwrap());

                        let mut histo = Histogram::new(bounds.clone()).unwrap();
                        for i in pyld.clone() {
                            histo.insert(i);
                        }

                        let mut bounds: Vec<Bound<$t>> =
                            bounds.into_iter().map(Bound::Finite).collect();
                        bounds.push(Bound::PosInf);

                        // confirm that the histogram has correctly binned by
                        // asserting that for every bound the correct number of
                        // payload items are below that upper bound
                        pyld.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        for b in bounds.iter() {
                            let mut below_count = 0;
                            for v in pyld.iter() {
                                match b {
                                    Bound::Finite(ref bnd) => {
                                        if v <= bnd {
                                            below_count += 1;
                                        } else {
                                            break;
                                        }
                                    }
                                    Bound::PosInf => {
                                        below_count += 1;
                                    }
                                }
                            }
                            assert_eq!(below_count, histo.total_below(*b))
                        }

                        TestResult::passed()
                    }
                    QuickCheck::new().quickcheck(inner as fn(Vec<$t>, Vec<$t>) -> TestResult);
                }

                #[test]
                fn test_insertion_above_count() {
                    fn inner(mut bounds: Vec<$t>, mut pyld: Vec<$t>) -> TestResult {
                        if bounds.is_empty() {
                            return TestResult::discard();
                        }
                        bounds.sort_by(|a, b| a.partial_cmp(b).unwrap());

                        let mut histo = Histogram::new(bounds.clone()).unwrap();
                        for i in pyld.clone() {
                            histo.insert(i);
                        }

                        let mut bounds: Vec<Bound<$t>> =
                            bounds.into_iter().map(Bound::Finite).collect();
                        bounds.push(Bound::PosInf);

                        // confirm that the histogram has correctly binned by
                        // asserting that for every bound the correct number of
                        // payload items are above that upper bound
                        pyld.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        for b in bounds.iter() {
                            let mut above_count = 0;
                            for v in pyld.iter() {
                                match b {
                                    Bound::Finite(ref bnd) => {
                                        if v > bnd {
                                            above_count += 1;
                                        }
                                    }
                                    Bound::PosInf => {}
                                }
                            }
                            assert_eq!(above_count, histo.total_above(*b))
                        }

                        TestResult::passed()
                    }
                    QuickCheck::new().quickcheck(inner as fn(Vec<$t>, Vec<$t>) -> TestResult);
                }

                #[test]
                fn test_insertion_between_count() {
                    fn inner(mut bounds: Vec<$t>, mut pyld: Vec<$t>) -> TestResult {
                        if bounds.is_empty() {
                            return TestResult::discard();
                        }
                        bounds.sort_by(|a, b| a.partial_cmp(b).unwrap());

                        let mut histo = Histogram::new(bounds.clone()).unwrap();
                        for i in pyld.clone() {
                            histo.insert(i);
                        }

                        let mut bounds: Vec<Bound<$t>> =
                            bounds.into_iter().map(Bound::Finite).collect();
                        bounds.push(Bound::PosInf);

                        // confirm that the histogram has correctly binned by
                        // asserting that for every (lower, upper] bound the
                        // correct number of payload items are recorded between
                        // that bound
                        pyld.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        for lower_b in bounds.iter() {
                            for upper_b in bounds.iter() {
                                let mut between_count = 0;
                                if lower_b < upper_b {
                                    for v in pyld.iter() {
                                        match (lower_b, upper_b) {
                                            (
                                                &Bound::Finite(ref lw_b),
                                                &Bound::Finite(ref up_b),
                                            ) => {
                                                if v > lw_b && v <= up_b {
                                                    between_count += 1;
                                                }
                                            }
                                            (&Bound::Finite(ref lw_b), &Bound::PosInf) => {
                                                if v > lw_b {
                                                    between_count += 1;
                                                }
                                            }
                                            _ => {}
                                        }
                                    }
                                }
                                assert_eq!(between_count, histo.total_between(*lower_b, *upper_b))
                            }
                        }

                        TestResult::passed()
                    }
                    QuickCheck::new().quickcheck(inner as fn(Vec<$t>, Vec<$t>) -> TestResult);
                }
            }
        };
    }
    // Why no generation for u8? Please see note on Histogram.
    generate_tests!(u16, u16);
    generate_tests!(u32, u32);
    generate_tests!(i16, i16);
    generate_tests!(i32, i32);
    generate_tests!(f32, f32);
    generate_tests!(f64, f64);
    generate_tests!(u64, u64);
    generate_tests!(i64, i64);
    generate_tests!(usize, usize);
    generate_tests!(isize, isize);
}
