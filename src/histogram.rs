//! 'histogram' approximates a distribution calculation by counting the number
//! of times samples fall into pre-configured bins. This implementation does not
//! require bins to be equally sized. The user must specify upper bounds on bins
//! via `Bounds`. The implementation includes a +Inf bound automatically.
//!
//! Storage cost is proportional to the number of bins.
//!
//! # Examples

use std::cmp;
use std::fmt;

#[derive(Debug, Copy, Clone)]
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
            Bound::Finite(ref x) => {
                match *other {
                    Bound::Finite(y) => x.partial_cmp(&y),
                    Bound::PosInf => Some(cmp::Ordering::Less),
                }
            }
            Bound::PosInf => {
                match *other {
                    Bound::Finite(_) => Some(cmp::Ordering::Greater),
                    Bound::PosInf => Some(cmp::Ordering::Equal),
                }
            }
        }
    }
}

impl<T> PartialEq for Bound<T>
where
    T: Copy + cmp::PartialEq,
{
    fn eq(&self, other: &Bound<T>) -> bool {
        match *self {
            Bound::Finite(ref x) => {
                match *other {
                    Bound::Finite(y) => y.eq(x),
                    Bound::PosInf => false,
                }
            }
            Bound::PosInf => false,
        }
    }
}

/// The Histogram type.
///
/// TODO more docs to come 
#[derive(Debug)]
pub struct Histogram<T>
where
    T: Copy,
{
    count: usize,
    sum: Option<T>,
    bins: Vec<(Bound<T>, usize)>,
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
    /// TODO documentation
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
            sum: None, // TODO make summations, tests for same 
            bins: bins,
        })
    }

    /// TODO documentation
    pub fn insert(&mut self, value: T) -> () {
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

    /// Returns the total number of items stored in the histogram 
    /// 
    /// TODO documentation
    pub fn count(&self) -> usize {
        self.count
    }

    /// Total number of items below supplied upper_bound
    ///
    /// TODO documentation
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
    /// TODO documentation
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
    /// TODO documentation
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
}

#[cfg(test)]
mod test {
    use super::*;
    use quickcheck::{QuickCheck, TestResult};

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

                        let mut histo = Histogram::new(bounds.clone()).unwrap();
                        let total = pyld.len();
                        for i in pyld.clone() {
                            histo.insert(i);
                        }

                        let mut bounds: Vec<Bound<$t>> =
                            bounds.into_iter().map(|x| Bound::Finite(x)).collect();
                        bounds.push(Bound::PosInf);

                        // confirm that the histogram holds the correct number of items
                        assert_eq!(total, histo.count());

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
                            bounds.into_iter().map(|x| Bound::Finite(x)).collect();
                        bounds.push(Bound::PosInf);

                        // confirm that the histogram has correctly binned by asserting that
                        // for every bound the correct number of payload items are below
                        // that upper bound
                        pyld.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        for b in bounds.iter() {
                            let mut below_count = 0;
                            for v in pyld.iter() {
                                match b {
                                    &Bound::Finite(ref bnd) => {
                                        if v <= bnd {
                                            below_count += 1;
                                        } else {
                                            break;
                                        }
                                    }
                                    &Bound::PosInf => {
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
                            bounds.into_iter().map(|x| Bound::Finite(x)).collect();
                        bounds.push(Bound::PosInf);

                        // confirm that the histogram has correctly binned by asserting that
                        // for every bound the correct number of payload items are above
                        // that upper bound
                        pyld.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        for b in bounds.iter() {
                            let mut above_count = 0;
                            for v in pyld.iter() {
                                match b {
                                    &Bound::Finite(ref bnd) => {
                                        if v > bnd {
                                            above_count += 1;
                                        }
                                    }
                                    &Bound::PosInf => {}
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
                           bounds.into_iter().map(|x| Bound::Finite(x)).collect();
                       bounds.push(Bound::PosInf);

                       // confirm that the histogram has correctly binned by asserting that
                       // for every (lower, upper] bound the correct number of payload
                       // items are recorded between that bound
                       pyld.sort_by(|a, b| a.partial_cmp(b).unwrap());
                       for lower_b in bounds.iter() {
                           for upper_b in bounds.iter() {
                               let mut between_count = 0;
                               if lower_b < upper_b {
                                   for v in pyld.iter() {
                                       match (lower_b, upper_b) {
                                           (&Bound::Finite(ref lw_b), &Bound::Finite(ref up_b)) => {
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
        }
    }
    generate_tests!(u16, u16);
    generate_tests!(u32, u32);
    generate_tests!(f32, f32);
    generate_tests!(f64, f64);
    generate_tests!(u64, u64);
    generate_tests!(usize, usize);
}
