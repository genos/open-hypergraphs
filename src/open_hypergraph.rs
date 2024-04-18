//! An open hypergraph is a cospan in Hypergraph whose feet are discrete.
use crate::{
    array::Backend,
    finite_function::{Error as FFError, FiniteFunction},
    hypergraph::{Error as HError, Hypergraph},
    indexed_coproduct::{Error as ICError, IndexedCoproduct},
    macros::impl_arith,
};
use std::{
    fmt::{Debug, Display, Formatter},
    ops::{BitOr, Shr},
};

/// An open hypergraph is a cospan in Hypergraph whose feet are discrete.
#[derive(PartialEq, Eq, Clone)]
pub struct OpenHypergraph<A: Backend> {
    /// source foot
    pub(crate) s: FiniteFunction<A>,
    /// target foot
    pub(crate) t: FiniteFunction<A>,
    /// hypergraph
    pub(crate) h: Hypergraph<A>,
}

impl<A: Backend> OpenHypergraph<A> {
    /// Check that the internal representation of this open hypergraph is correct.
    ///
    /// # Errors
    /// If `s`, `t`, and `h` don't correctly combine to form an open hypergraph.
    ///
    /// # References
    /// [John Regehr on assertions](https://blog.regehr.org/archives/1091)
    pub(crate) fn check_rep(&self) -> Result<(), Error<A>> {
        if self.s.target != self.h.num_vertices() {
            Err(Error::SHMismatch {
                t: self.s.target,
                s: self.h.num_edges(),
            })
        } else if self.t.target != self.h.num_vertices() {
            Err(Error::SHMismatch {
                t: self.t.target,
                s: self.h.num_edges(),
            })
        } else {
            Ok(())
        }
    }

    /// Safely construct a new `OpenHypergraph`.
    ///
    /// # Errors
    ///
    /// If the three arguments don't correctly combine to describe an open hypergraph.
    pub fn new(
        s: FiniteFunction<A>,
        t: FiniteFunction<A>,
        h: Hypergraph<A>,
    ) -> Result<Self, Error<A>> {
        let result = Self { s, t, h };
        result.check_rep()?;
        Ok(result)
    }

    /// Source for this open hypergraph
    ///
    /// # Errors
    /// If the internal parts are incompatible; should be impossible by construction.
    pub fn source(&self) -> Result<FiniteFunction<A>, Error<A>> {
        (&self.s >> &self.h.w).map_err(Into::into)
    }

    /// Target for this open hypergraph
    ///
    /// # Errors
    /// If the internal parts are incompatible; should be impossible by construction.
    pub fn target(&self) -> Result<FiniteFunction<A>, Error<A>> {
        (&self.t >> &self.h.w).map_err(Into::into)
    }

    /// Signature of this open hypergraph
    #[must_use]
    pub fn signature(&self) -> (FiniteFunction<A>, FiniteFunction<A>) {
        (self.h.w.to_initial(), self.h.x.to_initial())
    }

    /// Compose this open hypergraph with another.
    ///
    /// # Note
    ///
    /// This is also available via the `>>` operator, to suggest serial composition.
    ///
    /// # Errors
    ///
    /// If this open hypergraph's target doesn't match the other's source, or if they're
    /// incompatible in other ways.
    pub fn compose(&self, other: &Self) -> Result<Self, Error<A>> {
        if self.target()? == other.source()? {
            let h = (self | other)?;
            let q = self
                .t
                .inject0(other.h.num_vertices())
                .coequalizer(&other.s.inject1(self.h.num_vertices()))?;
            Ok(Self {
                s: (self.s.inject0(other.h.num_vertices()) >> &q)?,
                t: (other.t.inject1(self.h.num_vertices()) >> &q)?,
                h: h.h.coequalize_vertices(&q)?,
            })
        } else {
            Err(Error::CompositionMismatch {
                f: self.clone().into(),
                g: other.clone().into(),
            })
        }
    }

    /// Identity open hypergraph
    ///
    /// # Errors
    ///
    /// If the source of the input `x` is not zero.
    pub fn identity(w: FiniteFunction<A>, x: FiniteFunction<A>) -> Result<Self, Error<A>> {
        if x.source != 0 {
            Err(Error::SourceMustBeZero(x.clone()))
        } else {
            let s = FiniteFunction::identity(w.source);
            let t = FiniteFunction::identity(w.source);
            let h = Hypergraph::discrete(w, x)?;
            Ok(Self { s, t, h })
        }
    }

    /// Computes the tensor product.
    ///
    /// # Note
    ///
    /// This is also available via the `|` operator, to suggest parallel composition.
    ///
    /// # Errors
    ///
    /// If the hypergraphs are incompatible.
    pub fn tensor(&self, other: &Self) -> Result<Self, Error<A>> {
        Ok(Self {
            s: &self.s | &other.s,
            t: &self.t | &other.s,
            h: (&self.h + &other.h)?,
        })
    }

    /// Dagger
    pub fn dagger(&self) -> Self {
        Self {
            s: self.t.clone(),
            t: self.s.clone(),
            h: self.h.clone(),
        }
    }

    /// Frobenius spider
    pub fn spider(
        s: FiniteFunction<A>,
        t: FiniteFunction<A>,
        w: FiniteFunction<A>,
        x: FiniteFunction<A>,
    ) -> Result<Self, Error<A>> {
        let h = Hypergraph::discrete(w, x)?;
        Ok(Self { s, t, h })
    }

    /// Frobenius half spider
    pub fn half_spider(
        s: FiniteFunction<A>,
        w: FiniteFunction<A>,
        x: FiniteFunction<A>,
    ) -> Result<Self, Error<A>> {
        OpenHypergraph::spider(s, FiniteFunction::identity(w.source), w, x)
    }

    /// The N-fold tensoring of operations
    pub fn tensor_operations(
        x: FiniteFunction<A>,
        a: IndexedCoproduct<A>,
        b: IndexedCoproduct<A>,
    ) -> Result<Self, Error<A>> {
        if b.values.target != a.values.target {
            Err(Error::TargetsMismatch {
                a: a.clone(),
                b: b.clone(),
            })
        } else if x.source != a.source() {
            Err(Error::SourcesMismatch {
                f: x.clone(),
                i: a.clone(),
            })
        } else if x.source != b.source() {
            Err(Error::SourcesMismatch {
                f: x.clone(),
                i: b.clone(),
            })
        } else {
            let s = FiniteFunction::inj0(a.values.source, b.values.source);
            let t = FiniteFunction::inj1(a.values.source, b.values.source);
            let h = Hypergraph {
                s: IndexedCoproduct {
                    sources: a.sources,
                    values: s.clone(),
                },
                t: IndexedCoproduct {
                    sources: b.sources,
                    values: t.clone(),
                },
                w: (a.values + b.values)?,
                x,
            };
            Ok(Self { s, t, h })
        }
    }
}

impl_arith!(OpenHypergraph, Shr, shr, compose, true);
impl_arith!(OpenHypergraph, BitOr, bitor, tensor, true);

impl<A: Backend> Display for OpenHypergraph<A> {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "⟨{}, {}, {}⟩", self.s, self.t, self.h)
    }
}

impl<A: Backend> Debug for OpenHypergraph<A> {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        // Reuse Display cuz it's nice
        std::fmt::Display::fmt(self, f)
    }
}

/// Errors that can arise when building or computing with open hypergraphs.
#[derive(Debug, thiserror::Error)]
pub enum Error<A: Backend> {
    /// A finite function error occurred: {0}
    #[error("A finite function error occurred: {0}")]
    FiniteFunction(#[from] FFError<A>),
    /// A hypergraph error occurred: {0}
    #[error("A hypergraph error occurred: {0}")]
    Hypergraph(#[from] HError<A>),
    /// An indexed coproduct error occurred: {0}
    #[error("An indexed coproduct error occurred: {0}")]
    IndexedCoproduct(#[from] ICError<A>),
    /// The s target {t} must equal the source h {s}
    #[error("The s target {t} must equal the h source {s}")]
    SHMismatch {
        /// target: s's target
        t: usize,
        /// source: h's # nodes
        s: usize,
    },
    /// The t target {t} must equal the source h {s}
    #[error("The t target {t} must equal the h source {s}")]
    THMismatch {
        /// target: t's target
        t: usize,
        /// source: h's # nodes
        s: usize,
    },
    /// In the composition, the intermediate target {f} does not match the intermediate source of {g}.
    #[error("In the composition, the intermediate target of {f} does not match the intermediate source of {g}.")]
    CompositionMismatch {
        /// First function
        f: Box<OpenHypergraph<A>>,
        /// Second function
        g: Box<OpenHypergraph<A>>,
    },
    /// The source of {0} must be zero.
    #[error("The source of {0} must be zero.")]
    SourceMustBeZero(FiniteFunction<A>),
    /// The targets of {a} and {b} must be the same.
    #[error("The targets of {a} and {b} must be the same.")]
    TargetsMismatch {
        /// First indexed coproduct
        a: IndexedCoproduct<A>,
        /// Second indexed coproduct
        b: IndexedCoproduct<A>,
    },
    /// The source of {f} and {i} must be the same.
    #[error("The source of {f} and {i} must be the same.")]
    SourcesMismatch {
        /// Finite function
        f: FiniteFunction<A>,
        /// Indexed coproduct
        i: IndexedCoproduct<A>,
    },
}

/// `proptest` strategies for generating arbitrary open hypergraphs.
#[cfg(test)]
pub(crate) mod strategies {
    use super::*;
    use crate::{finite_function::strategies as ffs, hypergraph::strategies as hs};
    use proptest::prelude::*;

    pub(crate) fn arrows<A: Backend>() -> impl Strategy<Value = OpenHypergraph<A>> {
        hs::hypergraphs()
            .prop_flat_map(|h| {
                let w = h.num_vertices();
                (ffs::arrow_to(w), ffs::arrow_to(w), Just(h))
            })
            .prop_map(|(s, t, h)| OpenHypergraph { s, t, h })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::{NDArray1Usize, StdVec};
    use paste::paste;

    macro_rules! properties {
        ($A:ty) => {
            paste! {
                mod [<$A:snake:lower _open_hypergraph_properties>] {
                    use super::*;
                    use proptest::prelude::*;
                    proptest! {
                        #[test]
                        fn fuzz(o in strategies::arrows::<$A>()) {
                            prop_assert!(o.check_rep().is_ok());
                        }
                    }
                }
            }
        };
    }

    properties!(StdVec);
    properties!(NDArray1Usize);
}
