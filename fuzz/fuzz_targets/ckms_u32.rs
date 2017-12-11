#![no_main]
#[macro_use] extern crate libfuzzer_sys;
extern crate quantiles;
extern crate byteorder;

use std::io::Cursor;
use byteorder::{BigEndian, ReadBytesExt};

#[derive(Debug, Clone, Copy)]
pub struct Xorshift {
    seed: u64,
}

impl Xorshift {
    pub fn new(seed: u64) -> Xorshift {
        Xorshift { seed: seed }
    }

    pub fn next_val(&mut self) -> u32 {
        // implementation inspired by
        // https://github.com/astocko/xorshift/blob/master/src/splitmix64.rs
        use std::num::Wrapping as w;

        let mut z = w(self.seed) + w(0x9E37_79B9_7F4A_7C15_u64);
        let nxt_seed = z.0;
        z = (z ^ (z >> 30)) * w(0xBF58_476D_1CE4_E5B9_u64);
        z = (z ^ (z >> 27)) * w(0x94D0_49BB_1331_11EB_u64);
        self.seed = nxt_seed;
        u32::from((z ^ (z >> 31)).0 as u16)
    }
}

fuzz_target!(|data: &[u8]| {
    let mut cursor = Cursor::new(data);

    // unbounded, CKMS will adjust to within bounds 
    let error: f64 = if let Ok(res) = cursor.read_f64::<BigEndian>() {
        res
    } else {
        return;
    };
    // bounded 2**24
    let upper_bound: u32 = if let Ok(res) = cursor.read_u32::<BigEndian>() {
        res % 16_777_216
    } else {
        return;
    };
    // unbounded
    let seed: u64 = if let Ok(res) = cursor.read_u64::<BigEndian>() {
        res
    } else {
        return;
    };

    let mut xshft = Xorshift::new(seed);
    let mut ckms = quantiles::ckms::CKMS::<u32>::new(error);
    for _ in 0..(upper_bound as usize) {
        let val = xshft.next_val();
        ckms.insert(val);
    }
});
