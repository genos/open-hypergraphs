#![forbid(missing_docs)]
#![forbid(unsafe_code)]
#![doc = include_str!("../README.md")]
pub mod array;
pub mod finite_function;
pub mod functor;
pub mod hypergraph;
pub mod indexed_coproduct;
mod macros;
pub mod open_hypergraph;
pub mod optic;

pub use array::{Backend, StdVec};
pub use finite_function::FiniteFunction;
pub use functor::{FrobeniusFunctor, Functor};
pub use hypergraph::Hypergraph;
pub use indexed_coproduct::IndexedCoproduct;
