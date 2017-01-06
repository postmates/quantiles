#![feature(plugin)]
#![plugin(afl_plugin)]

extern crate afl;
extern crate quantiles;

use std::str::FromStr;
use quantiles::ckms::CKMS;

fn main() {
    afl::handle_string(|s| {
        let pyld: Vec<f64> = s.split_whitespace()
            .map(|f| f64::from_str(f))
            .filter(|f| f.is_ok())
            .map(|f| f.unwrap())
            .collect();

        if !pyld.len() > 3 && ((pyld[2] >= 0.0) || (pyld[2] <= 1.0))  {
            let mut ckms = CKMS::new(pyld[0]);

            for f in &pyld[3..] {
                ckms.insert(*f)
            }

            ckms.query(pyld[2]).unwrap();
        }
    })
}
