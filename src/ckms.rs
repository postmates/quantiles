//! This is an implementation of the algorithm presented in Cormode, Korn,
//! Muthukrishnan, Srivastava's paper "Effective Computation of Biased Quantiles
//! over Data Streams". The ambition here is to approximate quantiles on a
//! stream of data without having a boatload of information kept in memory.
//!
//! As of this writing you _must_ use the presentation in the IEEE version of
//! the paper. The authors' self-published copy of the paper is incorrect and
//! this implementation will _not_ make sense if you follow along using that
//! version. Only the 'full biased' invariant is used. The 'targeted quantiles'
//! variant of this algorithm is fundamentally flawed, an issue which the
//! authors correct in their "Space- and Time-Efficient Deterministic Algorithms
//! for Biased Quantiles over Data Streams"

use intrusive_collections::{LinkedList, LinkedListLink};
#[cfg(feature = "serde_support")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};
#[cfg(feature = "serde_support")]
use serde::de;
#[cfg(feature = "serde_support")]
use serde::de::{MapAccess, SeqAccess, Visitor};
#[cfg(feature = "serde_support")]
use serde::ser::SerializeStruct;
use std;
use std::cmp;
use std::fmt;
use std::fmt::Debug;
#[cfg(feature = "serde_support")]
use std::marker::PhantomData;
use std::ops::{Add, AddAssign, Div, Sub};

// The adapter describes how an object can be inserted into an intrusive
// collection. This is automatically generated using a macro.
intrusive_adapter!(EntryAdapter<T> = Box<Entry<T>>:
                   Entry<T> { link: LinkedListLink } where T: Copy);


#[derive(Debug, Clone)]
struct Entry<T: Copy> {
    link: LinkedListLink,
    v: T,
    g: usize,
    delta: usize,
}

impl<T> Entry<T>
where
    T: Copy,
{
    fn new(v: T, g: usize, delta: usize) -> Entry<T> {
        Entry {
            link: LinkedListLink::new(),
            v: v,
            g: g,
            delta: delta,
        }
    }
}

/// A structure to provide approximate quantiles queries in bounded memory and
/// with bounded error.
pub struct CKMS<T: Copy> {
    n: usize,

    // We follow the 'batch' method of the above paper. In this method,
    // incoming items are buffered in a priority queue, called 'buffer' here,
    // and once insert_threshold items are stored in the buffer it is drained
    // into the 'samples' collection. Insertion will cause some extranious
    // points to be held that can be merged. Once compress_threshold threshold
    // items are buffered the COMPRESS operation merges these extranious points.
    insert_threshold: usize,
    inserts: usize,

    // We aim for the full biased quantiles method. The paper this
    // implementation is based on includes a 'targeted' method but the authors
    // have granted that it is flawed in private communication. As such, all
    // queries for all quantiles will have the same error factor.
    error: f64,

    // This is the S(n) of the above paper. Entries are stored here and
    // occasionally merged. The outlined implementation uses a linked list but
    // we prefer a Vec for reasons of cache locality at the cost of worse
    // computational complexity.
    samples: LinkedList<EntryAdapter<T>>,
    total_samples: usize,

    sum: Option<T>,
    cma: Option<f64>,
    last_in: Option<T>,
}

#[cfg(feature = "serde_support")]
impl<'de, T> Deserialize<'de> for CKMS<T>
where
    T: Copy + Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<CKMS<T>, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "snake_case")]
        enum Field {
            N,
            InsertThreshold,
            Inserts,
            Error,
            Samples,
            TotalSamples,
            Sum,
            Cma,
            LastIn,
        };

        struct CKMSVisitor<T> {
            phantom: PhantomData<T>,
        }

        impl<'de, T> Visitor<'de> for CKMSVisitor<T>
        where
            T: Copy + Deserialize<'de>,
        {
            type Value = CKMS<T>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct Duration")
            }

            fn visit_seq<V>(self, mut seq: V) -> Result<CKMS<T>, V::Error>
            where
                V: SeqAccess<'de>,
                T: Deserialize<'de>,
            {
                let n = seq.next_element()?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;
                let insert_threshold = seq.next_element()?
                    .ok_or_else(|| de::Error::invalid_length(1, &self))?;
                let inserts = seq.next_element()?
                    .ok_or_else(|| de::Error::invalid_length(2, &self))?;
                let error = seq.next_element()?
                    .ok_or_else(|| de::Error::invalid_length(3, &self))?;
                let total_samples = seq.next_element()?
                    .ok_or_else(|| de::Error::invalid_length(4, &self))?;
                let sum = seq.next_element()?
                    .ok_or_else(|| de::Error::invalid_length(5, &self))?;
                let cma = seq.next_element()?
                    .ok_or_else(|| de::Error::invalid_length(6, &self))?;
                let last_in = seq.next_element()?
                    .ok_or_else(|| de::Error::invalid_length(7, &self))?;
                let enc_samples: Vec<(T, usize, usize)> = seq.next_element()?
                    .ok_or_else(|| de::Error::invalid_length(8, &self))?;
                let mut samples: LinkedList<EntryAdapter<T>> = LinkedList::new(EntryAdapter::new());
                for enc in enc_samples {
                    samples.push_back(Box::new(Entry::new(enc.0, enc.1, enc.2)));
                }
                Ok(CKMS {
                    n: n,
                    insert_threshold: insert_threshold,
                    inserts: inserts,
                    error: error,
                    samples: samples,
                    total_samples: total_samples,
                    sum: sum,
                    cma: cma,
                    last_in: last_in,
                })
            }

            fn visit_map<V>(self, mut map: V) -> Result<CKMS<T>, V::Error>
            where
                V: MapAccess<'de>,
                T: Deserialize<'de>,
            {
                let mut n = None;
                let mut insert_threshold = None;
                let mut inserts = None;
                let mut error = None;
                let mut samples = None;
                let mut total_samples = None;
                let mut sum = None;
                let mut cma = None;
                let mut last_in = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::N => {
                            if n.is_some() {
                                return Err(de::Error::duplicate_field("n"));
                            }
                            n = Some(map.next_value()?);
                        }
                        Field::InsertThreshold => {
                            if insert_threshold.is_some() {
                                return Err(de::Error::duplicate_field("insert_threshold"));
                            }
                            insert_threshold = Some(map.next_value()?);
                        }
                        Field::Inserts => {
                            if inserts.is_some() {
                                return Err(de::Error::duplicate_field("inserts"));
                            }
                            inserts = Some(map.next_value()?);
                        }
                        Field::Error => {
                            if error.is_some() {
                                return Err(de::Error::duplicate_field("error"));
                            }
                            error = Some(map.next_value()?);
                        }
                        Field::Samples => {
                            if samples.is_some() {
                                return Err(de::Error::duplicate_field("samples"));
                            }
                            samples = Some(map.next_value()?);
                        }
                        Field::TotalSamples => {
                            if total_samples.is_some() {
                                return Err(de::Error::duplicate_field("total_samples"));
                            }
                            total_samples = Some(map.next_value()?);
                        }
                        Field::Sum => {
                            if sum.is_some() {
                                return Err(de::Error::duplicate_field("sum"));
                            }
                            sum = Some(map.next_value()?);
                        }
                        Field::Cma => {
                            if cma.is_some() {
                                return Err(de::Error::duplicate_field("cma"));
                            }
                            cma = Some(map.next_value()?);
                        }
                        Field::LastIn => {
                            if last_in.is_some() {
                                return Err(de::Error::duplicate_field("last_in"));
                            }
                            last_in = Some(map.next_value()?);
                        }
                    }
                }
                let n = n.ok_or_else(|| de::Error::missing_field("n"))?;
                let insert_threshold =
                    insert_threshold.ok_or_else(|| de::Error::missing_field("insert_threshold"))?;
                let inserts = inserts.ok_or_else(|| de::Error::missing_field("inserts"))?;
                let error = error.ok_or_else(|| de::Error::missing_field("error"))?;
                let enc_samples: Vec<(T, usize, usize)> =
                    samples.ok_or_else(|| de::Error::missing_field("samples"))?;
                let mut samples: LinkedList<EntryAdapter<T>> = LinkedList::new(EntryAdapter::new());
                for enc in enc_samples {
                    samples.push_back(Box::new(Entry::new(enc.0, enc.1, enc.2)));
                }
                let total_samples =
                    total_samples.ok_or_else(|| de::Error::missing_field("total_samples"))?;
                let sum = sum.ok_or_else(|| de::Error::missing_field("sum"))?;
                let cma = cma.ok_or_else(|| de::Error::missing_field("cma"))?;
                let last_in = last_in.ok_or_else(|| de::Error::missing_field("last_in"))?;
                Ok(CKMS {
                    n: n,
                    insert_threshold: insert_threshold,
                    inserts: inserts,
                    error: error,
                    samples: samples,
                    total_samples: total_samples,
                    sum: sum,
                    cma: cma,
                    last_in: last_in,
                })
            }
        }

        const FIELDS: &'static [&'static str] = &[
            "n",
            "insert_threshold",
            "inserts",
            "error",
            "samples",
            "total_samples",
            "sum",
            "cma",
            "last_in",
        ];
        deserializer.deserialize_struct(
            "CKMS",
            FIELDS,
            CKMSVisitor {
                phantom: PhantomData,
            },
        )
    }
}

#[cfg(feature = "serde_support")]
impl<T> Serialize for CKMS<T>
where
    T: Copy + Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("CKMS", 9)?;
        state.serialize_field("n", &self.n)?;
        state.serialize_field("insert_threshold", &self.insert_threshold)?;
        state.serialize_field("inserts", &self.inserts)?;
        state.serialize_field("error", &self.error)?;
        state.serialize_field("total_samples", &self.total_samples)?;
        state.serialize_field("sum", &self.sum)?;
        state.serialize_field("cma", &self.cma)?;
        state.serialize_field("last_in", &self.last_in)?;
        let samples: Vec<(T, usize, usize)> = self.samples
            .iter()
            .map(|ent| (ent.v, ent.g, ent.delta))
            .collect();
        state.serialize_field("samples", &samples)?;
        state.end()
    }
}

impl<T> fmt::Debug for CKMS<T>
where
    T: Copy + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "CKMS {{ error: {}, sum: {:?}, cma: {:?}, count: {}, ",
            self.error,
            self.sum,
            self.cma,
            self.n
        )?;
        write!(f, "samples: [")?;
        let mut cursor = self.samples.front();
        for _ in 0..self.total_samples {
            write!(f, "{:?}, ", cursor.get().unwrap())?;
            cursor.move_next();
        }
        write!(f, "] }}")
    }
}

impl<T> AddAssign for CKMS<T>
where
    T: Copy
        + Add<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + PartialOrd
        + Debug
        + std::convert::Into<f64>,
{
    fn add_assign(&mut self, rhs: CKMS<T>) {
        self.last_in = rhs.last_in;
        self.sum = match (self.sum, rhs.sum) {
            (None, None) => None,
            (None, Some(y)) => Some(y),
            (Some(x), None) => Some(x),
            (Some(x), Some(y)) => Some(x.add(y)),
        };
        self.cma = match (self.cma, rhs.cma) {
            (None, None) => None,
            (None, Some(y)) => Some(y),
            (Some(x), None) => Some(x),
            (Some(x), Some(y)) => {
                let x_n: f64 = self.n as f64;
                let y_n: f64 = rhs.n as f64;
                Some(((x_n * x) + (y_n * y)) / (x_n + y_n))
            }
        };
        for smpl in rhs.samples {
            self.priv_insert(smpl.v);
        }
    }
}

#[inline]
fn invariant(r: f64, error: f64) -> usize {
    let i = (2.0 * error * r).floor() as usize;
    if i == 0 {
        1
    } else {
        i
    }
}

impl<
    T: Copy
        + PartialOrd
        + Debug
        + Add<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + std::convert::Into<f64>,
> CKMS<T> {
    /// Create a new CKMS
    ///
    /// A CKMS is meant to answer quantile queries with a known error bound. If
    /// the error passed here is ε and there have been `n` items inserted into
    /// CKMS then for any quantile query Φ the deviance from the true quantile
    /// will be +/- εΦn.
    ///
    /// For an error ε this structure will require T*(floor(1/(2*ε)) + O(1/ε log
    /// εn)) + f64 + usize + usize words of storage.
    ///
    /// # Examples
    /// ```
    /// use quantiles::ckms::CKMS;
    ///
    /// let mut ckms = CKMS::<u32>::new(0.001);
    /// for i in 1..1001 {
    ///     ckms.insert(i as u32);
    /// }
    /// assert_eq!(ckms.query(0.0), Some((1, 1)));
    /// assert_eq!(ckms.query(0.998), Some((998, 998)));
    /// assert_eq!(ckms.query(0.999), Some((999, 999)));
    /// assert_eq!(ckms.query(1.0), Some((1000, 1000)));
    /// ```
    ///
    /// `error` must but a value between 0 and 1, exclusive of both extremes. If
    /// you input an error <= 0.000_000_000_1 CKMS will assign an error of
    /// 0.000_000_000_1. Likewise, if your error is >= 1.0 CKMS will assign an
    /// error of 0.99.
    pub fn new(error: f64) -> CKMS<T> {
        let error = if error <= 0.000_000_000_1 {
            0.000_000_000_1
        } else if error >= 1.0 {
            0.99
        } else {
            error
        };
        let insert_threshold = 1.0 / (2.0 * error);
        let insert_threshold: usize = if insert_threshold < 1.0 {
            1
        } else {
            insert_threshold as usize
        };
        CKMS {
            n: 0,

            error: error,

            insert_threshold: insert_threshold,
            inserts: 0,

            samples: LinkedList::new(EntryAdapter::new()),
            total_samples: 0,

            last_in: None,
            sum: None,
            cma: None,
        }
    }

    /// Return the last element added to the CKMS
    ///
    /// # Example
    /// ```
    /// use quantiles::ckms::CKMS;
    ///
    /// let mut ckms = CKMS::new(0.1);
    /// ckms.insert(1.0);
    /// ckms.insert(2.0);
    /// ckms.insert(3.0);
    /// assert_eq!(Some(3.0), ckms.last());
    /// ```
    pub fn last(&self) -> Option<T> {
        self.last_in
    }

    /// Return the sum of the elements added to the CKMS
    ///
    /// # Example
    /// ```
    /// use quantiles::ckms::CKMS;
    ///
    /// let mut ckms = CKMS::new(0.1);
    /// ckms.insert(1.0);
    /// ckms.insert(2.0);
    /// ckms.insert(3.0);
    /// assert_eq!(Some(6.0), ckms.sum());
    /// ```
    pub fn sum(&self) -> Option<T> {
        self.sum
    }

    /// Return the cummulative moving average of the elements added to the CKMS
    ///
    /// # Example
    /// ```
    /// use quantiles::ckms::CKMS;
    ///
    /// let mut ckms = CKMS::new(0.1);
    /// ckms.insert(0.0);
    /// ckms.insert(100.0);
    ///
    /// assert_eq!(Some(50.0), ckms.cma());
    /// ```
    pub fn cma(&self) -> Option<f64> {
        self.cma
    }

    /// Insert a T into the CKMS
    ///
    /// Insertion will gradulally shift the approximate quantiles. This
    /// implementation is biased toward fast writes and slower queries. Storage
    /// may grow gradually, as defined in the module-level documentation, but
    /// will remain bounded.
    pub fn insert(&mut self, v: T) {
        self.sum = self.sum.map_or(Some(v), |s| Some(s.add(v)));
        self.last_in = Some(v);
        self.priv_insert(v);
        // NOTE: priv_insert increases self.n.
        let v_f64: f64 = v.into();
        let n: f64 = self.n as f64;
        self.cma = self.cma
            .map_or(Some(v_f64), |s| Some(s + ((v_f64 - s) / n)));
    }

    fn priv_insert(&mut self, v: T) {
        let s = self.total_samples;
        self.n += 1;
        self.total_samples += 1;
        if s == 0 {
            self.samples.push_front(Box::new(Entry::new(v, 1, 0)));
            return;
        }

        let mut idx = 0;
        let mut r = 0;
        for smpl in self.samples.iter() {
            match smpl.v.partial_cmp(&v).unwrap() {
                cmp::Ordering::Less => { idx += 1; r += smpl.g },
                _ => break,
            }
        }
        let delta = if idx == 0 || idx == s {
            0
        } else {
            invariant(r as f64, self.error) - 1
        };

        let entry = Box::new(Entry::new(v, 1, delta));
        if idx == 0 {
            self.samples.push_front(entry);
        } else if idx == s {
            self.samples.push_back(entry);
        } else {
            let mut cursor = self.samples.cursor_mut();
            // wind the cursor forward
            for _ in 0..idx {
                cursor.move_next();
            }
            cursor.insert_after(entry);
        }
        self.inserts = (self.inserts + 1) % self.insert_threshold;
        if self.inserts == 0 {
            self.compress();
        }
    }

    /// Query CKMS for a ε-approximate quantile
    ///
    /// This function returns an approximation to the true quantile-- +/- εΦn
    /// --for the points inserted. Argument q is valid 0. <= q <= 1.0. The
    /// minimum and maximum quantile, corresponding to 0.0 and 1.0 respectively,
    /// are always known precisely.
    ///
    /// Return
    ///
    /// # Examples
    /// ```
    /// use quantiles::ckms::CKMS;
    ///
    /// let mut ckms = CKMS::<u32>::new(0.001);
    /// for i in 0..1000 {
    ///     ckms.insert(i as u32);
    /// }
    ///
    /// assert_eq!(ckms.query(0.0), Some((1, 0)));
    /// assert_eq!(ckms.query(0.998), Some((998, 997)));
    /// assert_eq!(ckms.query(1.0), Some((1000, 999)));
    /// ```
    pub fn query(&self, q: f64) -> Option<(usize, T)> {
        let s = self.total_samples;

        if s == 0 {
            return None;
        }

        {
            let mut cursor = self.samples.front();

            let mut r = 0;
            let nphi = q * (self.n as f64);

            for _ in 1..s {
                let prev = cursor.get().unwrap();
                cursor.move_next();
                let cur = cursor.get().unwrap();

                r += prev.g;

                let lhs = (r + cur.g + cur.delta) as f64;
                let rhs = nphi + ((invariant(nphi, self.error) as f64) / 2.0);

                if lhs > rhs {
                    return Some((r, prev.v));
                }
            }
        }

        let v = self.samples.back().get().unwrap().v;
        Some((s, v))
    }

    /// Query CKMS for the count of its points
    ///
    /// This function returns the total number of points seen over the lifetime
    /// of the datastructure, _not_ the number of points currently stored in the
    /// structure.
    ///
    /// # Examples
    /// ```
    /// use quantiles::ckms::CKMS;
    ///
    /// let mut ckms = CKMS::<u32>::new(0.001);
    /// for i in 0..1000 {
    ///     ckms.insert(i as u32);
    /// }
    ///
    /// assert_eq!(ckms.count(), 1000);
    /// ```
    pub fn count(&self) -> usize {
        self.n
    }

    /// Retrieve a representative vector of points
    ///
    /// This function returns a represenative sample of points from the
    /// CKMS. Doing so consumes the CKMS.
    ///
    /// # Examples
    /// ```
    /// use quantiles::ckms::CKMS;
    ///
    /// let mut ckms = CKMS::<u32>::new(0.1);
    /// for i in 0..10 {
    ///     ckms.insert(i as u32);
    /// }
    ///
    /// assert_eq!(ckms.into_vec(), vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    /// ```
    pub fn into_vec(self) -> Vec<T> {
        self.samples.into_iter().map(|ent| ent.v).collect()
    }

    fn compress(&mut self) {
        if self.total_samples < 3 {
            return;
        }
        let mut cursor = self.samples.front_mut();
        if cursor.is_null() {
            return;
        }

        let mut i = 0;
        let mut r: f64 = 1.0;
        while i < (self.total_samples - 1) {
            let cur = cursor.get().unwrap();
            let cur_g = cur.g;
            cursor.move_next();
            let nxt = cursor.get().unwrap();
            let nxt_v = nxt.v;
            let nxt_g = nxt.g;
            let nxt_delta = nxt.delta;
            cursor.move_prev();

            if cur_g + nxt_g + nxt_delta <= invariant(r, self.error) {
                let entry = Box::new(Entry::new(nxt_v, nxt_g + cur_g, nxt_delta));
                cursor.replace_with(entry).unwrap();
                cursor.move_next();
                cursor.remove().unwrap();
                cursor.move_prev();
                self.total_samples -= 1;
            } else {
                cursor.move_next();
                i += 1;
            }
            r += 1.0;
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use quickcheck::{QuickCheck, TestResult};
    use std::f64::consts::E;

    fn percentile(data: &Vec<f64>, prcnt: f64) -> f64 {
        let idx = (prcnt * (data.len() as f64)) as usize;
        return data[idx];
    }

    #[test]
    fn test_cma() {
        fn inner(data: Vec<f64>, err: f64) -> TestResult {
            if data.is_empty() {
                return TestResult::discard();
            } else if !(err >= 0.0) || !(err <= 1.0) {
                return TestResult::discard();
            }

            let mut ckms = CKMS::<f64>::new(err);
            for d in &data {
                ckms.insert(*d);
            }

            let sum: f64 = data.iter().sum();
            let expected_mean: f64 = sum / (data.len() as f64);
            let mean = ckms.cma();
            assert!(mean.is_some());

            assert!((expected_mean - mean.unwrap()).abs() < err);
            return TestResult::passed();
        }
        QuickCheck::new()
            .tests(10000)
            .max_tests(100000)
            .quickcheck(inner as fn(Vec<f64>, f64) -> TestResult);
    }

    #[test]
    fn test_cma_add_assign() {
        fn inner(l_data: Vec<f64>, r_data: Vec<f64>, err: f64) -> TestResult {
            if !(err >= 0.0) || !(err <= 1.0) {
                return TestResult::discard();
            }

            let mut l_ckms = CKMS::<f64>::new(err);
            for d in &l_data {
                l_ckms.insert(*d);
            }
            let mut r_ckms = CKMS::<f64>::new(err);
            for d in &r_data {
                r_ckms.insert(*d);
            }

            let sum: f64 = l_data.iter().chain(r_data.iter()).sum();
            let expected_mean: f64 = sum / ((l_data.len() + r_data.len()) as f64);
            l_ckms += r_ckms;
            let mean = l_ckms.cma();
            if mean.is_some() {
                assert!((expected_mean - mean.unwrap()).abs() < err);
            }
            return TestResult::passed();
        }
        QuickCheck::new()
            .tests(10000)
            .max_tests(100000)
            .quickcheck(inner as fn(Vec<f64>, Vec<f64>, f64) -> TestResult);
    }

    #[test]
    fn error_nominal_test() {
        fn inner(mut data: Vec<f64>, prcnt: f64) -> TestResult {
            data.sort_by(|a, b| a.partial_cmp(b).unwrap());
            if !(prcnt >= 0.0) || !(prcnt <= 1.0) {
                return TestResult::discard();
            } else if data.len() < 1 {
                return TestResult::discard();
            }
            let err = 0.001;

            let mut ckms = CKMS::<f64>::new(err);
            for d in &data {
                ckms.insert(*d);
            }

            if let Some((_, v)) = ckms.query(prcnt) {
                debug_assert!(
                    (v - percentile(&data, prcnt)) < err,
                    "v: {} | percentile: {} | prcnt: {} | data: {:?}",
                    v,
                    percentile(&data, prcnt),
                    prcnt,
                    data
                );
                TestResult::passed()
            } else {
                TestResult::failed()
            }
        }
        QuickCheck::new()
            .tests(10000)
            .max_tests(100000)
            .quickcheck(inner as fn(Vec<f64>, f64) -> TestResult);
    }

    #[test]
    fn error_nominal_with_merge_test() {
        fn inner(lhs: Vec<f64>, rhs: Vec<f64>, prcnt: f64, err: f64) -> TestResult {
            if !(prcnt >= 0.0) || !(prcnt <= 1.0) {
                return TestResult::discard();
            } else if !(err >= 0.0) || !(err <= 1.0) {
                return TestResult::discard();
            } else if (lhs.len() + rhs.len()) < 1 {
                return TestResult::discard();
            }
            let mut data = lhs.clone();
            data.append(&mut rhs.clone());
            data.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let err = 0.001;

            let mut ckms = CKMS::<f64>::new(err);
            for d in &lhs {
                ckms.insert(*d);
            }
            let mut ckms_rhs = CKMS::<f64>::new(err);
            for d in &rhs {
                ckms_rhs.insert(*d);
            }
            ckms += ckms_rhs;

            if let Some((_, v)) = ckms.query(prcnt) {
                debug_assert!(
                    (v - percentile(&data, prcnt)) < err,
                    "v: {} | percentile: {} | prcnt: {} | data: {:?}",
                    v,
                    percentile(&data, prcnt),
                    prcnt,
                    data
                );
                TestResult::passed()
            } else {
                TestResult::failed()
            }
        }
        QuickCheck::new()
            .tests(10000)
            .max_tests(100000)
            .quickcheck(inner as fn(Vec<f64>, Vec<f64>, f64, f64) -> TestResult);
    }

    #[test]
    fn n_invariant_test() {
        fn n_invariant(fs: Vec<i32>) -> bool {
            let l = fs.len();

            let mut ckms = CKMS::<i32>::new(0.001);
            for f in fs {
                ckms.insert(f);
            }

            ckms.count() == l
        }
        QuickCheck::new()
            .tests(10000)
            .max_tests(100000)
            .quickcheck(n_invariant as fn(Vec<i32>) -> bool);
    }

    #[test]
    fn add_assign_test() {
        fn inner(pair: (i32, i32)) -> bool {
            let mut lhs = CKMS::<i32>::new(0.001);
            lhs.insert(pair.0);
            let mut rhs = CKMS::<i32>::new(0.001);
            rhs.insert(pair.1);

            let expected: i32 = pair.0 + pair.1;
            lhs += rhs;

            if let Some(x) = lhs.sum() {
                if x == expected {
                    if let Some(y) = lhs.last() {
                        y == pair.1
                    } else {
                        false
                    }
                } else {
                    false
                }
            } else {
                false
            }
        }
        QuickCheck::new()
            .tests(10000)
            .max_tests(100000)
            .quickcheck(inner as fn((i32, i32)) -> bool);
    }

    // prop: forany phi. (phi*n - f(phi*n, n)/2) =< r_i =< (phi*n + f(phi*n, n)/2)
    #[test]
    fn query_invariant_test() {
        fn query_invariant(f: f64, fs: Vec<i32>) -> TestResult {
            if fs.len() < 1 {
                return TestResult::discard();
            }

            let phi = (1.0 / (1.0 + E.powf(f.abs()))) * 2.0;

            let error = 0.001;
            let mut ckms = CKMS::<i32>::new(error);
            for f in fs {
                ckms.insert(f);
            }

            match ckms.query(phi) {
                None => TestResult::passed(), // invariant to check here? n*phi + f > 1?
                Some((rank, _)) => {
                    let nphi = phi * (ckms.n as f64);
                    let fdiv2 = (invariant(nphi, error) as f64) / 2.0;
                    TestResult::from_bool(
                        ((nphi - fdiv2) <= (rank as f64)) || ((rank as f64) <= (nphi + fdiv2)),
                    )
                }
            }
        }
        QuickCheck::new()
            .tests(10000)
            .max_tests(100000)
            .quickcheck(query_invariant as fn(f64, Vec<i32>) -> TestResult);
    }

    // prop: v_i-1 < v_i =< v_i+1
    #[test]
    fn asc_samples_test() {
        fn asc_samples(fs: Vec<i32>) -> TestResult {
            let mut ckms = CKMS::<i32>::new(0.001);
            let fs_len = fs.len();
            for f in fs {
                ckms.insert(f);
            }

            if ckms.total_samples == 0 && fs_len == 0 {
                return TestResult::passed();
            }

            let mut cursor = ckms.samples.front();
            if cursor.is_null() {
                return TestResult::failed();
            }

            for _ in 0..ckms.total_samples {
                let cur = cursor.get().unwrap();
                cursor.move_next();
                if cursor.is_null() {
                    break;
                }
                let nxt = cursor.get().unwrap();

                if !(cur.v <= nxt.v) {
                    return TestResult::failed();
                }
            }
            TestResult::passed()
        }
        QuickCheck::new()
            .tests(10000)
            .max_tests(100000)
            .quickcheck(asc_samples as fn(Vec<i32>) -> TestResult);
    }

    // prop: forall i. g_i + delta_i =< f(r_i, n)
    #[test]
    fn f_invariant_test() {
        fn f_invariant(fs: Vec<i32>) -> TestResult {
            let error = 0.001;
            let mut ckms = CKMS::<i32>::new(error);
            let fs_len = fs.len();
            for f in fs {
                ckms.insert(f);
            }

            if ckms.total_samples == 0 && fs_len == 0 {
                return TestResult::passed();
            }
            let mut cursor = ckms.samples.front();
            if cursor.is_null() {
                return TestResult::failed();
            }
            let mut r = 0;
            for _ in 0..ckms.total_samples {
                let ref cur = cursor.get().unwrap();

                r += cur.g;

                if !((cur.g + cur.delta) <= invariant(r as f64, error)) {
                    return TestResult::failed();
                }
                cursor.move_next();
            }
            TestResult::passed()
        }
        QuickCheck::new()
            .tests(10000)
            .max_tests(100000)
            .quickcheck(f_invariant as fn(Vec<i32>) -> TestResult);
    }

    #[test]
    fn compression_test() {
        let mut ckms = CKMS::<i32>::new(0.1);
        for i in 1..10000 {
            ckms.insert(i);
        }
        ckms.compress();

        let l = ckms.total_samples;
        let n = ckms.count();
        assert_eq!(9999, n);
        assert_eq!(316, l);
    }

    // prop: post-compression, samples is bounded above by O(1/e log^2 en)
    #[test]
    fn compression_bound_test() {
        fn compression_bound(fs: Vec<i32>) -> TestResult {
            if fs.len() < 15 {
                return TestResult::discard();
            }

            let mut ckms = CKMS::<i32>::new(0.001);
            for f in fs {
                ckms.insert(f);
            }
            ckms.compress();

            let s = ckms.total_samples as f64;
            let bound = (1.0 / ckms.error) * (ckms.error * (ckms.count() as f64)).log10().powi(2);

            if !(s <= bound) {
                println!(
                    "error: {:?} n: {:?} log10: {:?}",
                    ckms.error,
                    ckms.count() as f64,
                    (ckms.error * (ckms.count() as f64)).log10().powi(2)
                );
                println!("{:?} <= {:?}", s, bound);
                return TestResult::failed();
            }
            TestResult::passed()
        }
        QuickCheck::new()
            .tests(10000)
            .max_tests(100000)
            .quickcheck(compression_bound as fn(Vec<i32>) -> TestResult);
    }

    #[test]
    fn test_basics() {
        let mut ckms = CKMS::<i32>::new(0.001);
        for i in 1..1001 {
            ckms.insert(i as i32);
        }

        assert_eq!(ckms.query(0.00), Some((1, 1)));
        assert_eq!(ckms.query(0.05), Some((50, 50)));
        assert_eq!(ckms.query(0.10), Some((100, 100)));
        assert_eq!(ckms.query(0.15), Some((150, 150)));
        assert_eq!(ckms.query(0.20), Some((200, 200)));
        assert_eq!(ckms.query(0.25), Some((250, 250)));
        assert_eq!(ckms.query(0.30), Some((300, 300)));
        assert_eq!(ckms.query(0.35), Some((350, 350)));
        assert_eq!(ckms.query(0.40), Some((400, 400)));
        assert_eq!(ckms.query(0.45), Some((450, 450)));
        assert_eq!(ckms.query(0.50), Some((500, 500)));
        assert_eq!(ckms.query(0.55), Some((550, 550)));
        assert_eq!(ckms.query(0.60), Some((600, 600)));
        assert_eq!(ckms.query(0.65), Some((650, 650)));
        assert_eq!(ckms.query(0.70), Some((700, 700)));
        assert_eq!(ckms.query(0.75), Some((750, 750)));
        assert_eq!(ckms.query(0.80), Some((800, 800)));
        assert_eq!(ckms.query(0.85), Some((850, 850)));
        assert_eq!(ckms.query(0.90), Some((900, 900)));
        assert_eq!(ckms.query(0.95), Some((950, 950)));
        assert_eq!(ckms.query(0.99), Some((990, 990)));
        assert_eq!(ckms.query(1.00), Some((1000, 1000)));
    }
}
