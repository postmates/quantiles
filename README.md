# Now Archived and Forked
_quantiles_ will not be maintained in this repository going forward. Please use, create issues on, and make PRs to the fork of _quantiles_ located [here](https://github.com/blt/quantiles).

# quantiles

[![Build Status](https://travis-ci.org/postmates/quantiles.svg?branch=master)](https://travis-ci.org/postmates/quantiles) [![Codecov](https://img.shields.io/codecov/c/github/postmates/quantiles.svg)](https://codecov.io/gh/postmates/quantiles) [![Crates.io](https://img.shields.io/crates/v/quantiles.svg)](https://crates.io/crates/quantiles)

This crate is intended to be a collection of approxiate quantile algorithms that
provide guarantees around space and computation. Recent literature has advanced
approximation techniques but none are generally applicable and have fundamental
tradeoffs.

Initial work was done to support internal Postmates projects but the hope is
that the crate can be generally useful.

## The Algorithms

### CKMS - Effective Computation of Biased Quantiles over Data Streams

This is an implementation of the algorithm presented in Cormode, Korn,
Muthukrishnan, Srivastava's paper "Effective Computation of Biased Quantiles
over Data Streams". The ambition here is to approximate quantiles on a stream of
data without having a boatload of information kept in memory. This
implementation follows the
[IEEE version](http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=1410103&url=http%3A%2F%2Fieeexplore.ieee.org%2Fxpls%2Fabs_all.jsp%3Farnumber%3D1410103)
of the paper. The authors' self-published copy of the paper is incorrect and
this implementation will _not_ make sense if you follow along using that
version. Only the 'full biased' invariant is used. The 'targeted quantiles'
variant of this algorithm is fundamentally flawed, an issue which the authors
correct in their "Space- and Time-Efficient Deterministic Algorithms for Biased
Quantiles over Data Streams"

```rust
use quantiles::CKMS;

let mut ckms = CKMS::<u16>::new(0.001);
for i in 1..1001 {
    ckms.insert(i as u16);
}

assert_eq!(ckms.query(0.0), Some((1, 1)));
assert_eq!(ckms.query(0.998), Some((998, 998)));
assert_eq!(ckms.query(0.999), Some((999, 999)));
assert_eq!(ckms.query(1.0), Some((1000, 1000)));
```

Queries provide an approximation to the true quantile, +/- εΦn. In the above, ε
is set to 0.001, n is 1000. Minimum and maximum quantiles--0.0 and 1.0--are
already precise. The error for the middle query is then +/- 0.998. (This so
happens to be the exact quantile, but that doesn't always hold.)

For an error ε this structure will require `T*(floor(1/(2*ε)) + O(1/ε log εn)) +
f64 + usize + usize` words of storage, where T is the specialized type.

In local testing, insertion per point takes approximately 4 microseconds with a
variance of 7%. This comes to 250k points per second.

### Misra Gries - ε-approximate frequency counts

Misra-Gries calculates an ε-approximate frequency count for a stream of N elements.
The output is the k most frequent elements.

1. the approximate count f'[e] is smaller than the true frequency f[e] of e,
   but by at most εN, i.e., (f[e] - εN) ≤ f'[e] ≤ f[e]
2. any element e with a frequency f[e] ≥ εN appears in the result set

The error bound ε = 1/(k+1) where k is the number of counters used in the algorithm.
When k = 1 i.e. a single counter, the algorithm is equivalent to the
Boyer-Moore Majority algorithm.

If you want to check for elements that appear at least εN times, you will want
to perform a second pass to calculate the exact frequencies of the values in the
result set which can be done in constant space.

```rust
use quantiles::misra_gries::*;

let k: usize = 3;
let numbers: Vec<u32> = vec![1,3,2,1,3,4,3,1,2,1];
let counts = misra_gries(numbers.iter(), k);
let bound = numbers.len() / (k+1);
let in_range = |f_expected: usize, f_approx: usize| {
  f_approx <= f_expected &&
  (bound >= f_expected || f_approx >= (f_expected - bound))
};
assert!(in_range(4usize, *counts.get(&1).unwrap()));
assert!(in_range(2usize, *counts.get(&2).unwrap()));
assert!(in_range(3usize, *counts.get(&3).unwrap()));
```

### Greenwald Khanna - ε-approximate quantiles

Greenwald Khanna calculates ε-approximate quantiles.
If the desired quantile is `φ`, the ε-approximate
quantile is any element in the range of elements that rank
between `⌊(φ-ε)N⌋` and `⌊(φ+ε)N⌋`

The stream summary datastructure can cope with up to max[usize]
observations.

The beginning and end quantiles are clamped at the Minimum
and maximum observed elements respectively.

This page explains the theory:
[http://www.mathcs.emory.edu/~cheung/Courses/584-StreamDB/Syllabus/08-Quantile/Greenwald.html](http://www.mathcs.emory.edu/~cheung/Courses/584-StreamDB/Syllabus/08-Quantile/Greenwald.html)

```rust
use quantiles::greenwald_khanna::*;

let epsilon = 0.01;

let mut stream = Stream::new(epsilon);

let n = 1001;
for i in 1..n {
    stream.insert(i);
}
let in_range = |phi: f64, value: u32| {
  let lower = ((phi - epsilon) * (n as f64)) as u32;
  let upper = ((phi + epsilon) * (n as f64)) as u32;
  (epsilon > phi || lower <= value) && value <= upper
};
assert!(in_range(0f64, *stream.quantile(0f64)));
assert!(in_range(0.1f64, *stream.quantile(0.1f64)));
assert!(in_range(0.2f64, *stream.quantile(0.2f64)));
assert!(in_range(0.3f64, *stream.quantile(0.3f64)));
assert!(in_range(0.4f64, *stream.quantile(0.4f64)));
assert!(in_range(1f64, *stream.quantile(1f64)));
```

## Upgrading

### 0.2 -> 0.3

This release introduces two new algorithms, "Greenwald Khanna" and "Misra
Gries". The existing CKMS has been moved from root to its own submodule. You'll
need to update your imports from

```rust
use quantiles::CMKS;
```

to

```rust
use quantiles::ckms::CKMS;
```
