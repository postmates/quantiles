#[derive(Debug,Clone,PartialEq, Serialize, Deserialize)]
struct Entry<T: Copy> {
    v: T,
    g: usize,
    delta: usize,
}

/// A structure to provide approximate quantiles queries in bounded memory and
/// with bounded error.
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
