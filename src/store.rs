use std::{fmt, mem};
use std::ops::{Index, IndexMut};

#[derive(Clone, PartialEq, Debug)]
pub struct Store<T>
where
    T: PartialEq,
{
    data: Vec<Vec<T>>,
    inner_cap: usize,
    len: usize,
}

impl<T> Store<T>
where
    T: PartialEq,
{
    pub fn new() -> Store<T> {
        let inner_cap = (mem::size_of::<usize>() * 32) / mem::size_of::<T>(); // align to cache size
        assert!(inner_cap != 0);
        let data = vec![Vec::with_capacity(inner_cap)];
        Store {
            data: data,
            inner_cap: inner_cap,
            len: 0,
        }
    }

    pub fn insert(&mut self, index: usize, element: T) -> () {
        debug_assert!(index <= self.len, "insert");
        // Seek forward and find our place for insertion.
        let mut outer_idx = 0;
        let mut idx = index;
        while idx > self.data[outer_idx].len() {
            idx -= self.data[outer_idx].len();
            outer_idx += 1;
        }
        self.data[outer_idx].insert(idx, element);
        self.len += 1;
        // Now that we've inserted, check to see if we've gone over inner_cap on
        // this particular inner-store. If so, split it in half.
        if self.data[outer_idx].len() > self.inner_cap {
            let half = self.data[outer_idx].len() / 2;
            let nxt = self.data[outer_idx].split_off(half);
            self.data.insert(outer_idx + 1, nxt);
        }
    }

    pub fn remove(&mut self, index: usize) -> T where T: fmt::Debug {
        debug_assert!(index <= self.len, "remove");
        // Seek forward and find our place for removal.
        let mut outer_idx = 0;
        let mut idx = index;
        while idx >= self.data[outer_idx].len() {
            idx -= self.data[outer_idx].len();
            outer_idx += 1;
        }
        // println!("INNER: {:?} | LEN: {} | IDX: {}", self.data[outer_idx], self.data[outer_idx].len(), idx);
        let item = self.data[outer_idx].remove(idx);
        self.len -= 1;
        // TODO merge inner stores if two will fit into a single inner_cap
        item
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn iter(&self) -> StoreIter<T> {
        StoreIter {
            store: &self.data,
            outer_idx: 0,
            inner_idx: 0,
        }
    }
}

impl<T> IndexMut<usize> for Store<T>
where
    T: PartialEq,
{
    fn index_mut<'a>(&'a mut self, index: usize) -> &'a mut T {
        assert!(index < self.len);

        let mut outer_idx = 0;
        let mut idx = index;
        while idx > self.data[outer_idx].len() {
            idx -= self.data[outer_idx].len();
            outer_idx += 1;
        }
        &mut self.data[outer_idx][idx]
    }
}

impl<T> Index<usize> for Store<T>
where
    T: PartialEq,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        assert!(index < self.len);

        let mut outer_idx = 0;
        let mut idx = index;
        while idx > self.data[outer_idx].len() {
            idx -= self.data[outer_idx].len();
            outer_idx += 1;
        }
        &self.data[outer_idx][idx]
    }
}

pub struct StoreIter<'a, T>
where
    T: 'a + PartialEq,
{
    store: &'a Vec<Vec<T>>,
    outer_idx: usize,
    inner_idx: usize,
}

impl<'a, T> Iterator for StoreIter<'a, T>
where
    T: PartialEq + fmt::Debug,
{
    type Item = &'a T;

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
    use quickcheck::{Arbitrary, Gen, QuickCheck, TestResult};

    #[derive(Clone, Debug)]
    enum Actions {
        Insert { index: u8, value: u16 },
        Remove { index: u8 },
    }

    impl Arbitrary for Actions {
        fn arbitrary<G>(g: &mut G) -> Self
        where
            G: Gen,
        {
            match g.gen_range(0, 2) {
                0 => Actions::Insert {
                    index: Arbitrary::arbitrary(g),
                    value: Arbitrary::arbitrary(g),
                },
                1 => Actions::Remove {
                    index: Arbitrary::arbitrary(g),
                },
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn genuine_article() {
        fn inner(actions: Vec<Actions>) -> TestResult {
            let mut standard = Vec::new();
            let mut test_article = Store::new();

            for action in actions {
                match action {
                    Actions::Insert { index, value } => {
                        if (index as usize) > test_article.len() {
                            assert!((index as usize) > standard.len());
                            continue;
                        }
                        standard.insert((index as usize), value);
                        test_article.insert((index as usize), value);
                        assert_eq!(standard.len(), test_article.len());
                    }
                    Actions::Remove { index } => {
                        assert_eq!(standard.len(), test_article.len());
                        if test_article.is_empty() {
                            assert!(standard.is_empty());
                        } else if (index as usize) >= test_article.len() {
                            assert!((index as usize) >= standard.len());
                        } else {
                            debug_assert_eq!(
                                standard.remove((index as usize)),
                                test_article.remove((index as usize)),
                                "genuine_article_remove"
                            );
                        }
                        assert_eq!(standard.len(), test_article.len());
                    }
                }
            }

            for tup in standard.iter().zip(test_article.iter()) {
                assert_eq!(tup.0, tup.1)
            }

            return TestResult::passed();
        }
        QuickCheck::new().quickcheck(inner as fn(Vec<Actions>) -> TestResult);
    }

    #[test]
    fn obey_inner_cap() {
        fn inner(actions: Vec<Actions>) -> TestResult {
            let mut test_article = Store::new();

            for action in actions {
                match action {
                    Actions::Insert { index, value } => {
                        if (index as usize) <= test_article.len() {
                            test_article.insert((index as usize), value);
                        }
                    }
                    Actions::Remove { index } => {
                        if !test_article.is_empty() && !((index as usize) >= test_article.len()) {
                            test_article.remove((index as usize));
                        }
                    }
                }
            }

            for inner in test_article.data {
                assert!(inner.len() <= test_article.inner_cap);
            }

            return TestResult::passed();
        }
        QuickCheck::new().quickcheck(inner as fn(Vec<Actions>) -> TestResult);
    }
}
