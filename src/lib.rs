//! This crate provides approximate quantiles over data streams in a moderate
//! amount of memory.
//!
//! Order statistics is a rough business. Exact solutions are expensive in terms
//! of memory and computation. Recent literature has advanced approximations but
//! each have fundamental tradeoffs. This crate is intended to be a collection
//! of approximate algorithms that provide guarantees around space consumption.
#![deny(missing_docs, missing_debug_implementations, missing_copy_implementations, unsafe_code,
       unstable_features, unused_import_braces)]

#[cfg(test)]
#[macro_use]
extern crate quickcheck;

#[cfg(feature = "serde_support")]
#[macro_use]
extern crate serde_derive;

pub mod misra_gries;
pub mod greenwald_khanna;
pub mod ckms;
pub mod histogram;
