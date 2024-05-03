//! A hypergraph category is a symmetric monoidal category in which every object is equipped with a
//! special Frobenius monoid.
use crate::{
    array::Backend,
    finite_function::{Error as FFError, FiniteFunction},
    indexed_coproduct::{Error as ICError, IndexedCoproduct},
    macros::impl_arith,
};
use itertools::Itertools;
use std::{
    fmt::{Debug, Display, Formatter},
    ops::Add,
};

/// A hypergraph category is a symmetric monoidal category in which every object is equipped with a
/// special Frobenius monoid.
#[derive(PartialEq, Eq, Clone)]
pub struct Hypergraph<A: Backend> {
    /// sources: Σ_{x ∈ X} arity(e) → W
    pub(crate) s: IndexedCoproduct<A>,
    /// targets: Σ_{x ∈ X} coarity(e) → W
    pub(crate) t: IndexedCoproduct<A>,
    /// hypernode labels w: W → Σ₀
    pub(crate) w: FiniteFunction<A>,
    /// hyperedge labels x: X → Σ₁
    pub(crate) x: FiniteFunction<A>,
}

impl<A: Backend> Hypergraph<A> {
    /// Check that the internal representation of this hypergraph is correct.
    ///
    /// # Errors
    ///
    /// If `sources`, `targets`, `hypernode_labels`, and `hyperedge_labels` don't correctly combine
    /// to form a hypergraph.
    ///
    /// # References
    ///
    /// [John Regehr on assertions](https://blog.regehr.org/archives/1091)
    pub(crate) fn check_rep(&self) -> Result<(), Error<A>> {
        if self.s.target() != self.num_vertices() {
            Err(Error::SourcesNodesMismatch {
                sources: self.s.clone(),
                hypernode_labels: self.w.clone(),
            })
        } else if self.t.target() != self.num_vertices() {
            Err(Error::TargetsNodesMismatch {
                targets: self.t.clone(),
                hypernode_labels: self.w.clone(),
            })
        } else if self.s.source() != self.num_edges() {
            Err(Error::SourcesEdgesMismatch {
                sources: self.s.clone(),
                hyperedge_labels: self.x.clone(),
            })
        } else if self.t.source() != self.num_edges() {
            Err(Error::TargetsEdgesMismatch {
                targets: self.t.clone(),
                hyperedge_labels: self.x.clone(),
            })
        } else {
            Ok(())
        }
    }

    /// Number of vertices
    #[must_use]
    pub const fn num_vertices(&self) -> usize {
        self.w.source
    }

    /// Number of edges
    #[must_use]
    pub const fn num_edges(&self) -> usize {
        self.x.source
    }

    /// Safely construct a new `Hypergraph`.
    ///
    /// # Errors
    ///
    /// If the four arguments don't correctly combine to describe a hypergraph.
    ///
    /// # Examples
    ///
    /// ```
    /// use open_hypergraphs::{
    ///     FiniteFunction as F,
    ///     IndexedCoproduct as I,
    ///     Hypergraph as H,
    ///     StdVec as S
    /// };
    /// let h = H::new(
    ///     I::singleton(F::identity(3)), I::singleton(F::identity(3)), F::identity(3), F::<S>::identity(1)
    /// );
    /// assert!(h.is_ok());
    /// assert_eq!(
    ///     h.unwrap().to_string(),
    ///     "⟨∑([3]: 1 → ℕ): → ([0, 1, 2]: 3 → 3), ∑([3]: 1 → ℕ): → ([0, 1, 2]: 3 → 3), [0, 1, 2]: 3 → 3, [0]: 1 → 1⟩"
    /// );
    /// ```
    pub fn new(
        s: IndexedCoproduct<A>,
        t: IndexedCoproduct<A>,
        w: FiniteFunction<A>,
        x: FiniteFunction<A>,
    ) -> Result<Self, Error<A>> {
        let result = Self { s, t, w, x };
        result.check_rep()?;
        Ok(result)
    }

    /// The empty hypergraph with no hypernodes or hyperedges.
    #[must_use]
    pub fn empty(w: FiniteFunction<A>, x: FiniteFunction<A>) -> Self {
        let s = IndexedCoproduct::initial(0);
        let t = s.clone();
        Self { s, t, w, x }
    }

    /// The discrete hypergraph, consisting only of hypernodes.
    ///
    /// # Errors
    ///
    /// If the hyperedge labels have nonzero source.
    pub fn discrete(w: FiniteFunction<A>, x: FiniteFunction<A>) -> Result<Self, Error<A>> {
        if x.source == 0 {
            let s = IndexedCoproduct::initial(w.source);
            let t = s.clone();
            Ok(Self { s, t, w, x })
        } else {
            Err(Error::DiscreteRequiresNoEdges {
                hyperedge_labels: x,
            })
        }
    }

    /// Is this the discrete hypergraph?
    #[must_use]
    pub const fn is_discrete(&self) -> bool {
        self.s.source() == 0 && self.t.source() == 0 && self.x.source == 0
    }

    /// The coproduct hypergraphs is the pointwise on the components.
    ///
    /// # Note
    ///
    /// This is also available via the `+` operator.
    ///
    /// # Errors
    ///
    /// If this hypergraph's edges or vertices are incompatible.
    pub fn coproduct(&self, other: &Self) -> Result<Self, Error<A>> {
        if self.w.target != other.w.target {
            Err(Error::NodeLabelTargetsMismatch {
                first: self.w.clone(),
                second: other.w.clone(),
            })
        } else if self.x.target != other.x.target {
            Err(Error::EdgeLabelTargetsMismatch {
                first: self.x.clone(),
                second: other.x.clone(),
            })
        } else {
            Ok(Self {
                s: (&self.s | &other.s)?,
                t: (&self.t | &other.t)?,
                w: (&self.w + &other.w)?,
                x: (&self.x + &other.x)?,
            })
        }
    }

    /// Compute the n-fold coproduct of hypergraphs for n > 0.
    ///
    /// # Errors
    ///
    /// If called with an empty collection or if the hypergraphs are incompatible.
    pub fn coproduct_of(mut hs: impl Iterator<Item = Self>) -> Result<Self, Error<A>> {
        if let Some(h) = hs.next() {
            let (ss, ts, ws, xs): (
                Vec<IndexedCoproduct<_>>,
                Vec<IndexedCoproduct<_>>,
                Vec<FiniteFunction<_>>,
                Vec<FiniteFunction<_>>,
            ) = std::iter::once(h)
                .chain(hs)
                .map(|h| (h.s, h.t, h.w, h.x))
                .multiunzip();
            let sources = IndexedCoproduct::tensor_of(ss.into_iter())?;
            let targets = IndexedCoproduct::tensor_of(ts.into_iter())?;
            let hypernode_labels = FiniteFunction::coproduct_of(ws.into_iter())?;
            let hyperedge_labels = FiniteFunction::coproduct_of(xs.into_iter())?;
            Ok(Self {
                s: sources,
                t: targets,
                w: hypernode_labels,
                x: hyperedge_labels,
            })
        } else {
            Err(Error::EmptyCoproduct)
        }
    }

    /// Coequalize the vertices of this hypergraph via the given function `q`.
    ///
    /// # Errors
    ///
    /// If the number of edges isn't the same as the source of `q`, or if the finite function
    /// calculations involved throw an error.
    pub fn coequalize_vertices(&self, q: &FiniteFunction<A>) -> Result<Self, Error<A>> {
        if self.num_edges() == q.source {
            let w = q.coequalizer_universal(&self.w)?;
            let s = self.s.map_values(q)?;
            let t = self.t.map_values(q)?;
            Ok(Self {
                s,
                t,
                w,
                x: self.x.clone(),
            })
        } else {
            Err(Error::CoequalizerDifferentSource {
                q: q.clone(),
                edges: self.num_edges(),
            })
        }
    }

    /// Permute the nodes and edges of a hypergraph.
    ///
    /// # Errors
    /// If `w` isn't a permutation on the number of edges or `x` isn't a permutation on the number
    /// of nodes.
    pub fn permute(&self, w: &FiniteFunction<A>, x: &FiniteFunction<A>) -> Result<Self, Error<A>> {
        if w.source != self.num_vertices() || w.target != self.num_vertices() {
            Err(Error::NotAPermutation {
                p: w.clone(),
                n: self.num_vertices(),
            })
        } else if x.source != self.num_edges() || x.target != self.num_edges() {
            Err(Error::NotAPermutation {
                p: x.clone(),
                n: self.num_edges(),
            })
        } else {
            // https://github.com/statusfailed/open-hypergraphs/blob/afbc51c26ce615bdf6d3022e56cc33160d537745/open_hypergraphs/hypergraph.py#L93C1-L105C47
            // (w, x) is an ACSet/hypergraph morphism G ⇒ H
            // So we must have that both w and x preserve labels.
            // i.e.,
            //   G.w == w >> H.w
            // whence:
            //   H.w == w⁻¹ ; G.w
            // similarly
            //   H.x == x⁻¹ ; G.x
            // source/target maps
            //   Σ_{i ∈ X} f(i)
            // become
            //   Σ_{i ∈ X} g(x(i)) = Σ_{i ∈ X} f(i)
            let x = x.argsort();
            Ok(Self {
                s: self.s.map_indexes(&x)?.map_values(w)?,
                t: self.t.map_indexes(&x)?.map_values(w)?,
                w: (w.argsort() >> &self.w)?,
                x: (x >> &self.x)?,
            })
        }
    }
}

impl<A: Backend> Display for Hypergraph<A> {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "⟨{}, {}, {}, {}⟩", self.s, self.t, self.w, self.x)
    }
}

impl<A: Backend> Debug for Hypergraph<A> {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        // Reuse Display cuz it's nice
        Display::fmt(self, f)
    }
}

impl_arith!(Hypergraph, Add, add, coproduct, true);

/// Errors that can arise when building or computing with hypergraphs.
#[derive(Debug, thiserror::Error)]
pub enum Error<A: Backend> {
    /// A finite function error occurred: {0}
    #[error("A finite function error occurred: {0}")]
    FiniteFunction(#[from] FFError<A>),
    /// A finite function error occurred: {0}
    #[error("A indexed coproduct error occurred: {0}")]
    IndexedCoproduct(#[from] ICError<A>),
    /// Sources/Hypernodes mismatch: sources = {`sources`}, hypernode labels = {`hypernode_labels`}.
    #[error(
        "Sources/Hypernodes mismatch: sources = {sources}, hypernode labels = {hypernode_labels}."
    )]
    SourcesNodesMismatch {
        /// Potential sources
        sources: IndexedCoproduct<A>,
        /// Potential hypernode labels
        hypernode_labels: FiniteFunction<A>,
    },
    /// Targets/Hypernodes mismatch: targets = {`targets`}, hypernode labels = {`hypernode_labels`}.
    #[error(
        "Targets/Hypernodes mismatch: sources = {targets}, hypernode labels = {hypernode_labels}."
    )]
    TargetsNodesMismatch {
        /// Potential targets
        targets: IndexedCoproduct<A>,
        /// Potential hypernode labels
        hypernode_labels: FiniteFunction<A>,
    },
    /// Sources/Hyperedges mismatch: sources = {`sources`}, hyperedge labels = {`hyperedge_labels`}.
    #[error(
        "Sources/Hyperedges mismatch: sources = {sources}, hyperedge labels = {hyperedge_labels}."
    )]
    SourcesEdgesMismatch {
        /// Potential sources
        sources: IndexedCoproduct<A>,
        /// Potential hyperedge labels
        hyperedge_labels: FiniteFunction<A>,
    },
    /// Targets/Hyperedges mismatch: targets = {`targets`}, hyperedge labels = {`hyperedge_labels`}.
    #[error(
        "Targets/Hyperedges mismatch: targets = {targets}, hyperedge labels = {hyperedge_labels}."
    )]
    TargetsEdgesMismatch {
        /// Potential targets
        targets: IndexedCoproduct<A>,
        /// Potential hyperedge labels
        hyperedge_labels: FiniteFunction<A>,
    },
    /// The discrete hypergraph requires no edges, was given hyperedge labels = {`hyperedge_labels`}.
    #[error("The discrete hypergraph requires no edges, was given hyperedge labels = {hyperedge_labels}.")]
    DiscreteRequiresNoEdges {
        /// Potential hyperedge labels
        hyperedge_labels: FiniteFunction<A>,
    },
    /// The target of the node label {`first`} doesn't match the target of {`second`}.
    #[error("The target of the node label {first} doesn't match the target of {second}.")]
    NodeLabelTargetsMismatch {
        /// First hypernode labels
        first: FiniteFunction<A>,
        /// Second hypernode labels
        second: FiniteFunction<A>,
    },
    /// The target of the edge label {`first`} doesn't match the target of {`second`}.
    #[error("The target of the edge label {first} doesn't match the target of {second}.")]
    EdgeLabelTargetsMismatch {
        /// First hyperedge labels
        first: FiniteFunction<A>,
        /// Second hyperedge labels
        second: FiniteFunction<A>,
    },
    /// The n-fold coproduct of hypergraphs is only valid for n > 0.
    #[error("The n-fold coproduct of hypergraphs is only valid for n > 0.")]
    EmptyCoproduct,
    /// The purported coequalizer {`q`} has a different source than the number of edges {`edges`}.
    #[error(
        "The purported coequalizer {q} has a different source than the number of edges {edges}."
    )]
    CoequalizerDifferentSource {
        /// Potential coequalizer
        q: FiniteFunction<A>,
        /// Number of hyperedges
        edges: usize,
    },
    /// The function {p} is not a permutation of size {n}.
    #[error("The function {p} is not a permutation of size {n}.")]
    NotAPermutation {
        /// Potential permutation
        p: FiniteFunction<A>,
        /// Needed number of items
        n: usize,
    },
}

/// `proptest` strategies for generating arbitrary hypgraphs.
#[cfg(test)]
pub(crate) mod strategies {
    use super::*;
    use crate::{
        finite_function::{strategies as ffs, FiniteFunction},
        indexed_coproduct::strategies as ics,
    };
    use proptest::prelude::*;

    pub fn num_hypernodes(num_hyperedges: usize) -> impl Strategy<Value = usize> {
        usize::from(num_hyperedges > 0)..32
    }

    pub fn labels<A: Backend>() -> impl Strategy<Value = (FiniteFunction<A>, FiniteFunction<A>)> {
        ffs::arrows_nz()
            .prop_flat_map(|x| {
                let s = x.source;
                (num_hypernodes(s), Just(x))
            })
            .prop_flat_map(|(w, x)| (ffs::arrow_from(w), Just(x)))
    }

    pub fn hypergraph<A: Backend>(
        w: FiniteFunction<A>,
        x: FiniteFunction<A>,
    ) -> impl Strategy<Value = Hypergraph<A>> {
        let ws = w.source;
        let xs = x.source;
        (
            Just(w),
            Just(x),
            ics::indexed_coproduct(xs, ws),
            ics::indexed_coproduct(xs, ws),
        )
            .prop_map(|(hypernode_labels, hyperedge_labels, sources, targets)| {
                Hypergraph {
                    s: sources,
                    t: targets,
                    w: hypernode_labels,
                    x: hyperedge_labels,
                }
            })
    }

    pub fn hypergraphs<A: Backend>() -> impl Strategy<Value = Hypergraph<A>> {
        labels().prop_flat_map(|(w, x)| hypergraph(w, x))
    }

    pub fn hypergraph_and_permutation<A: Backend>(
    ) -> impl Strategy<Value = (Hypergraph<A>, FiniteFunction<A>, FiniteFunction<A>)> {
        hypergraphs().prop_flat_map(|h| {
            let vs = h.num_vertices();
            let es = h.num_edges();
            (Just(h), ffs::permutation(vs), ffs::permutation(es))
        })
    }

    pub fn discrete<A: Backend>() -> impl Strategy<Value = Hypergraph<A>> {
        labels().prop_flat_map(|(w, x)| hypergraph(w, x.to_initial()))
    }

    pub fn valid_coproduct<A: Backend>() -> impl Strategy<Value = (Hypergraph<A>, Hypergraph<A>)> {
        labels().prop_flat_map(|(w, x)| (hypergraph(w.clone(), x.clone()), hypergraph(w, x)))
    }

    pub fn coproductable<A: Backend>() -> impl Strategy<Value = Vec<Hypergraph<A>>> {
        (1..32usize, labels())
            .prop_flat_map(|(n, (w, x))| proptest::collection::vec(hypergraph(w, x), n))
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
                mod [<$A:snake:lower _hypergraph_properties>] {
                    use super::*;
                    use proptest::prelude::*;
                    proptest! {
                        #[test]
                        fn sources_targets_bounded(h in strategies::hypergraphs::<$A>()) {
                            prop_assert!(h.check_rep().is_ok());
                            prop_assert_eq!(h.s.target(), h.w.source);
                            prop_assert_eq!(h.t.target(), h.w.source);
                        }
                        #[test]
                        fn empty((w, x) in strategies::labels::<$A>()) {
                            let h = Hypergraph::empty(w.to_initial(), x.to_initial());
                            prop_assert!(h.check_rep().is_ok());
                            prop_assert_eq!(h.num_vertices(), 0);
                            prop_assert_eq!(h.num_edges(), 0);
                        }
                        #[test]
                        fn discrete((w, x) in strategies::labels::<$A>()) {
                            let h = Hypergraph::discrete(w, x.clone());
                            if x.source > 0 {
                                prop_assert!(h.is_err());
                            } else {
                                let h = h.unwrap();
                                prop_assert_eq!(h.s.source(), 0);
                                prop_assert_eq!(h.t.source(), 0);
                                prop_assert_eq!(h.x.source, 0);
                            }
                        }
                        #[test]
                        fn discrete_is_discrete(h in strategies::discrete::<$A>()) {
                            prop_assert!(h.check_rep().is_ok());
                            prop_assert!(h.is_discrete());
                        }
                        #[test]
                        fn coproduct((g0, g1) in strategies::valid_coproduct::<$A>()) {
                            let h = (&g0 + &g1);
                            prop_assert!(h.is_ok());
                            let h = h.unwrap();
                            prop_assert!(h.check_rep().is_ok());
                            prop_assert_eq!(h.s.source(), g0.s.source() + g1.s.source());
                            prop_assert_eq!(h.t.source(), g0.t.source() + g1.t.source());
                            prop_assert_eq!(h.s.values, &g0.s.values | &g1.s.values);
                            prop_assert_eq!(h.t.values, &g0.t.values | &g1.t.values);
                            let gw = &g0.w + &g1.w;
                            prop_assert!(gw.is_ok());
                            prop_assert_eq!(h.w.clone(), gw.unwrap());
                            let gx = &g0.x + &g1.x;
                            prop_assert!(gx.is_ok());
                            prop_assert_eq!(h.x.clone(), gx.unwrap());
                        }
                        #[test]
                        fn coproduct_of(mut gs in strategies::coproductable::<$A>()) {
                            let actual = Hypergraph::coproduct_of(gs.iter().cloned());
                            prop_assert!(actual.is_ok());
                            let g = gs.remove(0);
                            let expected = gs.iter().fold(Ok(g), |acc, g| acc.and_then(|a| a + g));
                            prop_assert!(expected.is_ok());
                            prop_assert_eq!(actual.unwrap(), expected.unwrap());
                        }
                        #[test]
                        fn hypergraph_perm((g, w, x) in strategies::hypergraph_and_permutation::<$A>()) {
                            let h = g.permute(&w, &x);
                            prop_assert!(h.is_ok());
                            let h = h.unwrap();
                            prop_assert!(h.check_rep().is_ok());
                            // vertices
                            prop_assert_eq!(h.num_vertices(), g.num_vertices());
                            let i = h.w.argsort();
                            let j = g.w.argsort();
                            let hw = (&i >> &h.w);
                            prop_assert!(hw.is_ok());
                            let gw = (&j >> &g.w);
                            prop_assert!(gw.is_ok());
                            prop_assert_eq!(hw.unwrap(), gw.unwrap());
                            // edges
                            prop_assert_eq!(h.num_edges(), g.num_edges());
                            let i = h.x.argsort();
                            let j = g.x.argsort();
                            let hx = (&i >> &h.x);
                            prop_assert!(hx.is_ok());
                            let gx = (&j >> &g.x);
                            prop_assert!(gx.is_ok());
                            prop_assert_eq!(hx.unwrap(), gx.unwrap());
                        }
                    }
                }
            }
        };
    }

    properties!(StdVec);
    properties!(NDArray1Usize);
}
