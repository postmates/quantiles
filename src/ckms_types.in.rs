#[derive(Debug,Clone,PartialEq, Serialize, Deserialize)]
struct Entry<T: Copy> {
    v: T,
    g: usize,
    delta: usize,
}

/// This is an implementation of the algorithm presented in Cormode, Korn,
/// Muthukrishnan, Srivastava's paper "Effective Computation of Biased Quantiles
/// over Data Streams". The ambition here is to approximate quantiles on a
/// stream of data without having a boatload of information kept in memory.
///
/// As of this writing you _must_ use the presentation in the IEEE version of
/// the paper. The authors' self-published copy of the paper is incorrect and
/// this implementation will _not_ make sense if you follow along using that
/// version. Only the 'full biased' invariant is used. The 'targeted quantiles'
/// variant of this algorithm is fundamentally flawed, an issue which the
/// authors correct in their "Space- and Time-Efficient Deterministic Algorithms
/// for Biased Quantiles over Data Streams"
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
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
    samples: Vec<Entry<T>>,

    sum: Option<T>,
    last_in: Option<T>,
}
