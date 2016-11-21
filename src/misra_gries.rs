//! J. Misra and D. Gries.
//! @article{MISRA1982143,
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
//! }
//!
//! Misra-Gries calculates an ε-approximate frequency count for a stream of N elements.
//! The output is the k most frequent elements.
//! 1) the approximate count f'[e] is smaller than the true frequency f[e] of e, 
//! but by at most εN, i.e., (f[e] − εN) ≤ f'[e] ≤ f[e]
//! 2) any element e with a frequency f[e] ≥ εN appears in the result set
//! The error bound ε = 1/k where k is the number of counters used in the algorithm.

use std::collections::BTreeMap;
use std::collections::btree_map::Entry;

/// Calculates the `k` most frequent elements in the iterable
/// stream of elements `stream` using an ε-approximate frequency count where ε = 1/k
pub fn misra_gries<I,V>(stream: I, k: usize) -> BTreeMap<V,usize> where I: IntoIterator<Item=V>, V: Ord + Clone {
    let mut counters = BTreeMap::new();
    for i in stream {
        let counters_len = counters.len();
        let mut counted = false;

        match counters.entry(i.clone()) {
            Entry::Occupied(mut item) => {
                *item.get_mut() += 1;
                counted = true;       
            },
            Entry::Vacant(slot) => {
                if counters_len < k {
                    slot.insert(1);
                    counted = true;
                }
            }
        }

        if !counted {
            for (_i, c) in counters.iter_mut() {
                *c -= 1;
            }

            counters = counters.into_iter()
                .filter(|&(_, v)| v != 0)
                .collect();
        }
    }

    counters
}

#[cfg(test)]
mod test {
    use super::*;
    use std::collections::BTreeMap;

    /// Calculate exact element frequencies using O(n) space.
    pub fn exact_frequencies<I,V>(stream: I) -> BTreeMap<V,usize> where I: IntoIterator<Item=V>, V: Ord + Clone {
        let mut counts = BTreeMap::new();
        for i in stream {
            *counts.entry(i.clone()).or_insert(0) += 1;
        }
        counts
    }

    #[test]
    fn test_exact_frequencies() {
        let numbers = vec![1,2,1,3,3,1,2,4];
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
                let approxes = misra_gries(xs.iter(), k);

                for (i, c) in approxes {
                    let exact = *exacts.get(i).unwrap();

                    if c > exact {
                        return false;
                    }

                    if (n/k) < exact && c < (exact - n / k) {
                        return false;
                    }
                }
            }

            true
        }
    }
}