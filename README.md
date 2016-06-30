# quantiles

This crate is intended to be a collection of approxiate quantlie algorithms that
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

```
use quantiles::CKMS;

let mut ckms = CKMS::<u16>::new(0.001); // ε = 0.001
for i in 0..1000 {
    ckms.insert(i as u16);
}

assert_eq!(ckms.query(0.0), Some((1, 0)));       // Φ = 0.0
assert_eq!(ckms.query(0.998), Some((998, 997))); // Φ = 0.998
assert_eq!(ckms.query(1.0), Some((1000, 999)));  // Φ = 1.0
```

Queries provide an approximation to the true quantile, +/- εΦn. In the above, ε
is set to 0.001, n is 1000. Minimum and maximum quantiles--0.0 and 1.0--are
already precise. The error for the middle query is then +/- 0.998. (This so
happens to be the exact quantile, but that doesn't always hold.)

For an error ε this structure will require `T*(floor(1/(2*ε)) + O(1/ε log εn)) +
f64 + usize + usize` words of storage, where T is the specialized type.

In local testing, insertion per point takes approximately 4 microseconds with a
variance of 7%. This comes to 250k points per second.
