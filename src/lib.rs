//! This crate provides approximate quantiles over data streams in a moderate
//! amount of memory.
//!
//! Order statistics is a rough business. Exact solutions are expensive in terms
//! of memory and computation. Recent literature has advanced approximations but
//! each have fundamental tradeoffs. This crate is intended to be a collection
//! of approximate algorithms that provide guarantees around space consumption.
#![deny(missing_docs,
        missing_debug_implementations, missing_copy_implementations,
        unused_import_braces, unused_qualifications)]
#![doc(html_root_url = "https://postmates.github.io/quantiles/")]

#[cfg(test)]
#[macro_use]
extern crate quickcheck;

pub mod misra_gries;
pub mod greenwald_khanna;
pub mod ckms;
