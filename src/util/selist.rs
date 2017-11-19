use std::cmp;
use std::fmt;
/// A space-efficient linked list
///
/// This module implements a space efficient linked list for use in
/// summarization algorithms that require storage of many small points. A fully
/// linked list is inappropriate for our lolspeed needs but storing elements in
/// a Vec is unsustainable above a relatively small amount of points.
///
/// Implementation is based on:
/// http://opendatastructures.org/ods-java/3_3_SEList_Space_Efficient_.html
///
/// This structure is also called an 'unrolled linked list'.
use std::ptr;

#[allow(dead_code)]
#[derive(Debug)]
pub struct SEList<T> {
    max_block_size: usize,
    total_elements: usize,
    total_blocks: usize,

    head: *mut Block<T>,
}

struct Block<T> {
    elems: Vec<T>,
    next: *mut Block<T>,
    prev: *mut Block<T>,
}

#[derive(Debug, Clone, Copy)]
pub enum Error {
    IndexOutOfBounds,
}

impl<T> Block<T> {
    fn new(max_block_size: usize) -> *mut Block<T> {
        Box::into_raw(Box::new(Block {
            elems: Vec::with_capacity(max_block_size),
            next: ptr::null_mut(),
            prev: ptr::null_mut(),
        }))
    }

    fn new_from_elems(max_block_size: usize, mut elems: Vec<T>) -> *mut Block<T> {
        elems.reserve(max_block_size);
        Box::into_raw(Box::new(Block {
            elems: elems,
            next: ptr::null_mut(),
            prev: ptr::null_mut(),
        }))
    }
}

impl<T> Drop for SEList<T> {
    fn drop(&mut self) {
        let mut blk = self.head;
        let mut nxt;
        unsafe {
            while !blk.is_null() {
                assert!((*blk).prev.is_null());
                if !(*blk).next.is_null() {
                    nxt = (*blk).next;
                    (*nxt).prev = ptr::null_mut();
                    (*blk).next = ptr::null_mut();
                    let bx = Box::from_raw(blk);
                    drop(bx);
                    blk = nxt;
                } else {
                    let bx = Box::from_raw(blk);
                    drop(bx);
                    break;
                }
            }
        }
    }
}

impl<T> SEList<T>
where
    T: cmp::PartialOrd + fmt::Debug,
{
    pub fn new(max_block_size: usize) -> SEList<T> {
        assert!(max_block_size > 0);

        let head = Block::new(max_block_size);
        SEList {
            max_block_size: max_block_size,
            total_elements: 0,
            total_blocks: 0,
            head: head,
        }
    }

    pub fn search(&self, elem: &T) -> usize {
        let mut idx: usize = 0;
        let mut blk = self.head;
        unsafe {
            while !blk.is_null() {
                if (*blk).elems.is_empty() {
                    return idx;
                }
                
                let fst = &(*blk).elems[0];
                let lst = &(*blk).elems[(*blk).elems.len() - 1];

                if (fst < elem) && (lst < elem) {
                    idx += (*blk).elems.len();
                } else if (fst < elem) && (lst >= elem) {
                    for e in &(*blk).elems {
                        match e.partial_cmp(&elem).unwrap() {
                            cmp::Ordering::Less => idx += 1,
                            cmp::Ordering::Equal => return idx,
                            cmp::Ordering::Greater => return idx.saturating_sub(1),
                        }
                    }
                }

                blk = (*blk).next;
            }
        }
        idx
    }

    fn block_tidy(&mut self, block: *mut Block<T>) {
        unsafe {
            if (*block).elems.len() > self.max_block_size {
                let blk_size = match self.max_block_size {
                    0 => unreachable!(),
                    1 => 1,
                    2 => 1,
                    3 => 2,
                    i => i / 2,
                };
                let new_elems = (*block).elems.split_off(blk_size);
                let new_blk = Block::new_from_elems(self.max_block_size, new_elems);
                (*new_blk).next = (*block).next;
                (*block).next = new_blk;
                (*new_blk).prev = block;
                self.total_blocks += 1;
            }
        }
    }

    pub fn insert(&mut self, index: usize, elem: T) -> Result<(), Error> {
        let mut blk = self.head;
        let mut idx = index;
        unsafe {
            while idx != 0 {
                assert!(!blk.is_null());
                let total_elems = (*blk).elems.len();
                if idx <= total_elems {
                    break;
                } else {
                    idx -= total_elems;
                }
                blk = (*blk).next;
            }
            (*blk).elems.insert(idx, elem);
            self.block_tidy(blk);
        }
        self.total_elements += 1;
        Ok(())
    }

    pub fn remove(&mut self, index: usize) -> Result<T, Error> {
        let block = self.max_block_size / index;
        if block > self.total_blocks {
            return Err(Error::IndexOutOfBounds);
        }
        let offset = index % block;

        let mut idx = block - 1;
        let mut blk = self.head;
        unsafe {
            while idx != 0 {
                blk = (*blk).next;
                idx -= 1;
            }
            self.total_elements -= 1;
            return Ok((*blk).elems.remove(offset));
        }
    }

    fn len(&self) -> usize {
        self.total_elements
    }

    #[cfg(test)]
    fn blocks(&self) -> usize {
        self.total_blocks
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use quickcheck::{QuickCheck, TestResult};

    #[test]
    fn insert_increase_total_elements() {
        fn inner(data: Vec<u32>, max_block_size: usize) -> TestResult {
            if max_block_size == 0 {
                return TestResult::discard();
            }

            let mut selist = SEList::new(max_block_size);
            let mut expected_len = 0;

            assert_eq!(selist.len(), expected_len);
            for elem in data {
                let idx = selist.search(&elem);
                assert!(selist.insert(idx, elem).is_ok());
                expected_len += 1;
                assert_eq!(selist.len(), expected_len);
            }
            TestResult::passed()
        }
        QuickCheck::new().quickcheck(inner as fn(Vec<u32>, usize) -> TestResult);
    }

    #[test]
    fn insert_may_increase_blocks() {
        fn inner(data: Vec<u32>, max_block_size: usize) -> TestResult {
            if max_block_size == 0 {
                return TestResult::discard();
            }

            let mut selist = SEList::new(max_block_size);
            let mut expected_blocks = 0;

            assert_eq!(selist.len(), expected_blocks);
            for elem in data {
                let idx = selist.search(&elem);
                assert!(selist.insert(idx, elem).is_ok());
                if expected_blocks == selist.blocks() {
                    continue;
                } else if expected_blocks == (selist.total_blocks - 1) {
                    expected_blocks += 1;
                    continue;
                } else {
                    return TestResult::failed();
                }
            }
            TestResult::passed()
        }
        QuickCheck::new().quickcheck(inner as fn(Vec<u32>, usize) -> TestResult);
    }

    // Unless a block is the last block, then that block contains no more than b
    // elements, where b is the maximum block size.
    #[test]
    fn block_size() {
        fn inner(data: Vec<u32>, max_block_size: usize) -> TestResult {
            if max_block_size == 0 {
                return TestResult::discard();
            }

            let mut selist = SEList::new(max_block_size);
            for elem in data {
                let idx = selist.search(&elem);
                assert!(selist.insert(idx, elem).is_ok());
            }

            let mut blk = selist.head;
            let mut prev;
            while !blk.is_null() {
                unsafe {
                    assert!((*blk).elems.len() <= max_block_size);
                    prev = blk;
                    blk = (*blk).next;
                    assert!(prev != blk);
                }
            }
            TestResult::passed()
        }
        QuickCheck::new().quickcheck(inner as fn(Vec<u32>, usize) -> TestResult);
    }

    // #[test]
    // fn simple_block_size() {
    //     let data = vec![0, 0, 0, 0, 1, 0, 1];
    //     let max_block_size = 6;

    //     let mut selist = SEList::new(max_block_size);
    //     for elem in data {
    //         let idx = selist.search(&elem);
    //         // println!("ELEM: {:?} | IDX: {:?}", elem, idx);
    //         assert!(selist.insert(idx, elem).is_ok());
    //     }

    //     let mut blk = selist.head;
    //     let mut prev = ptr::null_mut();
    //     while !blk.is_null() {
    //         unsafe {
    //             // println!("{:?}", (*blk).elems);
    //             assert!((*blk).elems.len() <= max_block_size);
    //             // println!("{:?} -> {:?}", prev, blk);
    //             prev = blk;
    //             blk = (*blk).next;
    //             assert!(prev != blk);
    //         }
    //     }
    // }
}
