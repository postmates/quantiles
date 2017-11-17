pub mod selist;

#[derive(Debug, Clone, Copy)]
pub struct Xorshift {
    seed: u64, 
}

impl Xorshift {
    pub fn new(seed: u64) -> Xorshift {
        Xorshift {
            seed: seed,
        }
    }

    pub fn next(&mut self) -> u32 {
        // implementation inspired by
        // https://github.com/astocko/xorshift/blob/master/src/splitmix64.rs
        use std::num::Wrapping as w;
        
        let mut z = w(self.seed) + w(0x9E3779B97F4A7C15_u64);
        let nxt_seed = z.0;
        z = (z ^ (z >> 30)) * w(0xBF58476D1CE4E5B9_u64);
        z = (z ^ (z >> 27)) * w(0x94D049BB133111EB_u64);
        self.seed = nxt_seed;
        ((z ^ (z >> 31)).0 as u16) as u32
    }
}
