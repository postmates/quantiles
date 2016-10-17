extern crate serde_codegen;

use std::env;
use std::path::Path;

fn main() {
    let out_dir = env::var_os("OUT_DIR").unwrap();

    let ckms_src = Path::new("src/ckms_types.in.rs");
    let ckms_dst = Path::new(&out_dir).join("ckms_types.rs");
    serde_codegen::expand(&ckms_src, &ckms_dst).unwrap();
}
