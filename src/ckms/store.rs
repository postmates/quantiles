use std::fmt;
use std::ops::{Index, IndexMut};

use ckms::entry::Entry;

#[derive(Clone, PartialEq, Debug)]
pub struct Inner<T>
where
    T: PartialEq,
{
    data: Vec<Entry<T>>,
    g_sum: u32,
}

pub fn invariant(r: f64, error: f64) -> u32 {
    let i = (2.0 * error * r).floor() as u32;
    if i == 0 {
        1
    } else {
        i
    }
}

impl<T> Inner<T>
where
    T: PartialEq + PartialOrd + Copy,
{
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn split_off(&mut self, index: usize) -> Self {
        assert!(index < self.data.len());
        let nxt = self.data.split_off(index);
        let nxt_g_sum = nxt.iter().fold(0, |acc, ref x| acc + x.g);
        self.g_sum -= nxt_g_sum;
        Inner {
            data: nxt,
            g_sum: nxt_g_sum,
        }
    }
}

#[derive(Clone, PartialEq, Debug)]
pub struct Store<T>
where
    T: PartialEq,
{
    // We aim for the full biased quantiles method. The paper this
    // implementation is based on includes a 'targeted' method but the authors
    // have granted that it is flawed in private communication. As such, all
    // queries for all quantiles will have the same error factor.
    pub error: f64,

    data: Vec<Inner<T>>,
    inner_cap: usize,
    len: usize, // samples currently stored
    n: usize,   // total samples ever stored
}

impl<T> Store<T>
where
    T: PartialEq + PartialOrd + Copy,
{
    pub fn new(inner_cap: usize, error: f64) -> Store<T> {
        assert!(inner_cap != 0);
        let data = Inner {
            data: Vec::with_capacity(inner_cap),
            g_sum: 0,
        };
        Store {
            error: error,
            data: vec![data],
            inner_cap: inner_cap,
            len: 0,
            n: 0,
        }
    }

    pub fn insert(&mut self, element: T) -> ()
    where
        T: fmt::Debug,
    {
        // insert at the front
        if self.data[0].data.is_empty() || (self.data[0].data[0].v >= element) {
            self.data[0].data.insert(
                0,
                Entry {
                    v: element,
                    g: 1,
                    delta: 0,
                },
            );
            self.data[0].g_sum += 1;
            self.n += 1;
            self.len += 1;

            if self.data[0].len() > self.inner_cap {
                let nxt = self.data[0].split_off(self.inner_cap);
                if self.data.len() > 1 {
                    self.data.insert(1, nxt);
                } else {
                    self.data.push(nxt);
                }
            }
            return;
        }

        let mut outer_idx = self.data.len() - 1;
        let mut inner_idx = self.data[outer_idx].len() - 1;

        // insert at the back
        if self.data[outer_idx].data[inner_idx].v < element {
            self.data[outer_idx].data.push(Entry {
                v: element,
                g: 1,
                delta: 0,
            });
            self.data[outer_idx].g_sum += 1;
            self.n += 1;
            self.len += 1;
            if self.data[outer_idx].len() > self.inner_cap {
                let nxt = self.data[outer_idx].split_off(self.inner_cap);
                self.data.push(nxt);
            }
            return;
        }

        // insert in the middle
        outer_idx = 0;
        inner_idx = 0;
        let mut r = 0;

        // Seek the outer_idx forward to the right cache line
        while outer_idx < self.data.len() {
            // The element for insertion is larger than the largest in the
            // present inner cache. In that case, we kick the outer_idx up and
            // capture the g_sum into our r.
            let mx = self.data[outer_idx].data.len();
            if element > self.data[outer_idx].data[mx - 1].v {
                outer_idx += 1;
                r += self.data[outer_idx].g_sum;
            } else {
                break;
            }
        }

        // Seek the inner_idx forward to the right location
        while inner_idx < self.data[outer_idx].data.len() {
            // The inner cache for insertion is here at outer_cache. We now seek
            // inner_idx forward while the current inner_idx is < than the
            // element for insertion.
            if self.data[outer_idx].data[inner_idx].v < element {
                inner_idx += 1;
                r += 1;
            } else {
                break;
            }
        }

        self.data[outer_idx].data.insert(
            inner_idx,
            Entry {
                v: element,
                g: 1,
                delta: invariant(r as f64, self.error) - 1,
            },
        );
        self.data[outer_idx].g_sum += 1;

        if self.data[outer_idx].len() > self.inner_cap {
            let nxt = self.data[outer_idx].split_off(self.inner_cap);
            self.data.insert(outer_idx + 1, nxt);
        }

        self.n += 1;
        self.len += 1;
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Total stored samples
    ///
    /// This value will fluctuate as compression happens.
    pub fn len(&self) -> usize {
        self.len
    }

    #[cfg(test)]
    /// Total samples, ever
    ///
    /// This value will never decrease and may or may not be equivalent to
    /// `Self::len`
    pub fn count(&self) -> usize {
        self.n
    }

    pub fn compress(&mut self) {
        if self.len() < 3 {
            return;
        }

        let mut cur_outer_idx = 0;
        let mut cur_inner_idx = 0;
        let mut nxt_outer_idx = 0;
        let mut nxt_inner_idx = 1;

        let mut r: u32 = 1;

        while cur_outer_idx < self.data.len() {
            let cur_g = self.data[cur_outer_idx][cur_inner_idx].g;

            // If the nxt_inner_idx has gone off the rails then it's time for us
            // to move up to the next inner cache for the next point.
            if nxt_inner_idx >= self.data[nxt_outer_idx].len() {
                nxt_inner_idx = 0;
                nxt_outer_idx += 1;
                // When nxt_outer_idx goes off the end we've run out of samples
                // to compress.
                if nxt_outer_idx >= self.data.len() {
                    break;
                }
            }

            let nxt_v = self.data[nxt_outer_idx][nxt_inner_idx].v;
            let nxt_g = self.data[nxt_outer_idx][nxt_inner_idx].g;
            let nxt_delta = self.data[nxt_outer_idx][nxt_inner_idx].delta;

            if cur_g + nxt_g + nxt_delta <= invariant(r as f64, self.error) {
                self.data[cur_outer_idx][cur_inner_idx].v = nxt_v;
                self.data[cur_outer_idx][cur_inner_idx].g += nxt_g;
                self.data[cur_outer_idx][cur_inner_idx].delta = nxt_delta;
                // If the two outer indexes don't match then we've 'moved' a g
                // from one inner cache to another. So, we scoot them.
                if cur_outer_idx != nxt_outer_idx {
                    self.data[nxt_outer_idx].g_sum -= nxt_g;
                    self.data[cur_outer_idx].g_sum += nxt_g;
                }
                self.data[nxt_outer_idx].data.remove(nxt_inner_idx);
                // Now that we've collapsed a point it's possible that we can
                // collapse the next next point into the current one as well. We
                // leave the indexes well enough alone as we've just removed an
                // item from the present inner cache.
                self.len -= 1;
            } else {
                // If we haven't collapsed any points we move the current
                // indexes to the next indexes. We also scoot up the next INNER
                // index, taking care to not adjust the outer index. We avoid
                // adjusting the outer index because it's possible we don't need
                // to move to a new inner cache yet.
                cur_outer_idx = nxt_outer_idx;
                cur_inner_idx = nxt_inner_idx;
                nxt_inner_idx += 1;
            }
            r += 1;
        }

        // TODO combine inners that will fit inside the cap
    }

    pub fn query(&self, q: f64) -> Option<(usize, T)> {
        if self.is_empty() {
            return None;
        }

        let mut r: u32 = 0;
        let s = self.len();
        let nphi = q * (self.n as f64);
        for i in 1..s {
            // TODO indexing is no longer constant, make sure we don't do two
            // seeking indexes
            let prev = &self[i - 1];
            let cur = &self[i];

            r += prev.g;

            let lhs = (r + cur.g + cur.delta) as f64;

            let inv = invariant(nphi, self.error);
            let rhs = nphi + ((inv as f64) / 2.0);

            if lhs > rhs {
                return Some((r as usize, prev.v));
            }
        }

        let v = self[s - 1].v;
        Some((s, v))
    }

    #[cfg(test)]
    pub fn iter(&self) -> StoreIter<T> {
        StoreIter {
            store: &self.data,
            outer_idx: 0,
            inner_idx: 0,
        }
    }
}

impl<T> IndexMut<usize> for Inner<T>
where
    T: PartialEq,
{
    fn index_mut<'a>(&'a mut self, index: usize) -> &'a mut Entry<T> {
        &mut self.data[index]
    }
}

impl<T> Index<usize> for Inner<T>
where
    T: PartialEq,
{
    type Output = Entry<T>;

    fn index<'a>(&'a self, index: usize) -> &'a Self::Output {
        &self.data[index]
    }
}

impl<T> IndexMut<usize> for Store<T>
where
    T: PartialEq + PartialOrd + Copy,
{
    fn index_mut<'a>(&'a mut self, index: usize) -> &'a mut Entry<T> {
        let mut outer_idx = 0;
        let mut idx = index;
        while idx >= self.data[outer_idx].len() {
            idx -= self.data[outer_idx].len();
            outer_idx += 1;
        }
        &mut self.data[outer_idx][idx]
    }
}

impl<T> Index<usize> for Store<T>
where
    T: PartialEq + PartialOrd + Copy,
{
    type Output = Entry<T>;

    fn index(&self, index: usize) -> &Self::Output {
        let mut outer_idx = 0;
        let mut idx = index;
        while idx >= self.data[outer_idx].len() {
            idx -= self.data[outer_idx].len();
            outer_idx += 1;
        }
        &self.data[outer_idx][idx]
    }
}

#[cfg(test)]
pub struct StoreIter<'a, T>
where
    T: 'a + PartialEq,
{
    store: &'a Vec<Inner<T>>,
    outer_idx: usize,
    inner_idx: usize,
}

#[cfg(test)]
impl<'a, T> Iterator for StoreIter<'a, T>
where
    T: PartialEq + Copy + PartialOrd + fmt::Debug,
{
    type Item = &'a Entry<T>;

    fn next(&mut self) -> Option<Self::Item> {
        while self.outer_idx < self.store.len() {
            if self.inner_idx < self.store[self.outer_idx].len() {
                let ret = &self.store[self.outer_idx][self.inner_idx];
                self.inner_idx += 1;
                return Some(ret);
            }
            self.inner_idx = 0;
            self.outer_idx += 1;
        }
        None
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use quickcheck::{QuickCheck, TestResult};

    #[test]
    fn inner_caches_test() {
        let mut store = Store::<i32>::new(10, 0.99);
        for i in 0..100 {
            store.insert(i);
        }

        assert_eq!(10, store.data.len());
    }

    #[test]
    fn compression_test() {
        let mut store = Store::<i32>::new(100, 0.1);
        for i in 0..10_000 {
            store.insert(i);
        }
        store.compress();

        assert_eq!(10_000, store.count());
        assert_eq!(42, store.len());
    }

    #[test]
    fn obey_inner_cap() {
        fn inner(data: Vec<f64>, inner_cap: usize, err: f64) -> TestResult {
            if data.is_empty() {
                return TestResult::discard();
            } else if inner_cap == 0 {
                return TestResult::discard();
            } else if !(err >= 0.0) || !(err <= 1.0) {
                return TestResult::discard();
            }

            let mut store = Store::<f64>::new(inner_cap, err);
            for d in &data {
                store.insert(*d);
            }

            for inner in store.data {
                assert!(inner.len() <= store.inner_cap);
            }

            return TestResult::passed();
        }
        QuickCheck::new().quickcheck(inner as fn(Vec<f64>, usize, f64) -> TestResult);
    }
}
