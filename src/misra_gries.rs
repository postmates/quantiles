//! Misra-Gries calculates an ε-approximate frequency count for a stream of N
//! elements. The output is the k most frequent elements.
//!
//! 1. the approximate count f'[e] is smaller than the true frequency f[e] of e,
//!    but by at most εN, i.e., (f[e] - εN) ≤ f'[e] ≤ f[e]
//! 2. any element e with a frequency f[e] ≥ εN appears in the result set
//!
//! The error bound ε = 1/(k+1) where k is the number of counters used in the
//! algorithm.
//! When k = 1 i.e. a single counter, the algorithm is equivalent to the
//! Boyer-Moore Majority algorithm.
//!
//! If you want to check for elements that appear at least εN times, you will
//! want to perform a second pass to calculate the exact frequencies of the
//! values in the result set which can be done in constant space.
//!
//! `@article{MISRA1982143,
//! title = "Finding repeated elements",
//! journal = "Science of Computer Programming",
//! volume = "2",
//! number = "2",
//! pages = "143 - 152",
//! year = "1982",
//! issn = "0167-6423",
//! doi = "http://dx.doi.org/10.1016/0167-6423(82)90012-0",
//! url = "http://www.sciencedirect.com/science/article/pii/0167642382900120",
//! author = "J. Misra and David Gries",
//! }`
//!
//! # Examples
//!
//! ```
//! use quantiles::misra_gries::*;
//!
//! let k: usize = 3;
//! let numbers: Vec<u32> = vec![1,3,2,1,3,4,3,1,2,1];
//! let counts = misra_gries(numbers.iter(), k);
//! let bound = numbers.len() / (k+1);
//! let in_range = |f_expected: usize, f_approx: usize| {
//!   f_approx <= f_expected &&
//!   (bound >= f_expected || f_approx >= (f_expected - bound))
//! };
//! assert!(in_range(4usize, *counts.get(&1).unwrap()));
//! assert!(in_range(2usize, *counts.get(&2).unwrap()));
//! assert!(in_range(3usize, *counts.get(&3).unwrap()));
//! ```

use std::collections::btree_map::Entry;
use std::collections::BTreeMap;

/// Calculates the `k` most frequent elements in the iterable
/// stream of elements `stream` using an ε-approximate frequency count where ε
/// = 1/(k+1)
pub fn misra_gries<I, V>(stream: I, k: usize) -> BTreeMap<V, usize>
where
    I: IntoIterator<Item = V>,
    V: Ord + Clone,
{
    let mut counters = BTreeMap::new();
    for i in stream {
        let counters_len = counters.len();
        let mut counted = false;

        match counters.entry(i.clone()) {
            Entry::Occupied(mut item) => {
                *item.get_mut() += 1;
                counted = true;
            }
            Entry::Vacant(slot) => {
                if counters_len < k {
                    slot.insert(1);
                    counted = true;
                }
            }
        }

        if !counted {
            for c in counters.values_mut() {
                *c -= 1;
            }

            counters = counters.into_iter().filter(|&(_, v)| v != 0).collect();
        }
    }

    counters
}

#[cfg(test)]
mod test {
    use super::*;
    use std::collections::BTreeMap;

    /// Calculate exact element frequencies using O(n) space.
    pub fn exact_frequencies<I, V>(stream: I) -> BTreeMap<V, usize>
    where
        I: IntoIterator<Item = V>,
        V: Ord + Clone,
    {
        let mut counts = BTreeMap::new();
        for i in stream {
            *counts.entry(i.clone()).or_insert(0) += 1;
        }
        counts
    }

    #[test]
    fn test_exact_frequencies() {
        let numbers = vec![1, 2, 1, 3, 3, 1, 2, 4];
        let counts = exact_frequencies(numbers.iter());
        assert_eq!(*counts.get(&1).unwrap() as u32, 3);
        assert_eq!(*counts.get(&2).unwrap() as u32, 2);
        assert_eq!(*counts.get(&3).unwrap() as u32, 2);
        assert_eq!(*counts.get(&4).unwrap() as u32, 1);
    }

    quickcheck! {
        fn is_exact(xs: Vec<u32>) -> bool {
            exact_frequencies(xs.iter()) == misra_gries(xs.iter(), xs.len())
        }

        fn is_approximate(xs: Vec<u32>) -> bool {
            //(f[e] − εN) ≤ f'[e] ≤ f[e]

            let exacts = exact_frequencies(xs.iter());
            let n = xs.len();

            for k in 1..n {
                let epsilon_n = n / (k+1);
                let approxes = misra_gries(xs.iter(), k);

                for (i, c) in approxes {
                    let exact = *exacts.get(i).unwrap();

                    if c > exact {
                        return false;
                    }

                    if epsilon_n < exact && c < (exact - epsilon_n) {
                        return false;
                    }
                }
            }

            true
        }
    }
}
