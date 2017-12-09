use std::cmp;

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct Entry<T>
where
    T: PartialEq,
{
    pub g: u32,
    pub delta: u32,
    pub v: T,
}

// The derivation of PartialEq for Entry is not appropriate. The sole ordering
// value in an Entry is the value 'v'.
impl<T> PartialEq for Entry<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Entry<T>) -> bool {
        self.v == other.v
    }
}

impl<T> PartialOrd for Entry<T>
where
    T: PartialOrd,
{
    fn partial_cmp(&self, other: &Entry<T>) -> Option<cmp::Ordering> {
        self.v.partial_cmp(&other.v)
    }
}
