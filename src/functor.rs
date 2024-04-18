//! Hypergraph functors
use crate::{
    array::Backend,
    finite_function::{Error as FFError, FiniteFunction},
    indexed_coproduct::{Error as ICError, IndexedCoproduct},
    open_hypergraph::{Error as OHError, OpenHypergraph},
};

/// Strict symmetric monoidal hypergraph functors
pub trait Functor<A: Backend> {
    ///  Map objects
    fn map_objects(&self, objects: &FiniteFunction<A>) -> Result<IndexedCoproduct<A>, Error<A>>;
    /// Map arrow
    fn map_arrow(&self, f: &OpenHypergraph<A>) -> Result<OpenHypergraph<A>, Error<A>>;
}

/// Errors that can occur when building or computing with functors.
#[derive(Debug, thiserror::Error)]
pub enum Error<A: Backend> {
    /// A finite function error occured
    #[error("A finite function error occured: {0}")]
    FiniteFunction(#[from] FFError<A>),
    /// A finite function error occured
    #[error("A indexed coproduct error occured: {0}")]
    IndexedCoproduct(#[from] ICError<A>),
    /// An open hypergraph error occured
    #[error("An open hypergraph error occured: {0}")]
    OpenHypergraph(#[from] OHError<A>),
}

// This is not *quite* what we want! It only maps the values; but we also need the new sources!
fn map_half_spider<A: Backend>(
    fw: &IndexedCoproduct<A>,
    f: &FiniteFunction<A>,
) -> Result<FiniteFunction<A>, Error<A>> {
    fw.sources.injections(f).map_err(Into::into)
}

/// Map a tensoring of operations into an open hypergraph
pub trait FrobeniusFunctor<A: Backend>: Functor<A> {
    /// Compute F(x₁) ● F(x₂) ● ... ● F(xn), where each x ∈ Σ₁ is an operation, and sources and
    /// targets are the types of each operation.
    fn map_operations(
        &self,
        x: &FiniteFunction<A>,
        sources: &IndexedCoproduct<A>,
        targets: &IndexedCoproduct<A>,
    ) -> Result<OpenHypergraph<A>, Error<A>>;

    /// Implement map_arrow based on map_operations!
    fn map_arrow(&self, f: &OpenHypergraph<A>) -> Result<OpenHypergraph<A>, Error<A>> {
        // Ff: the tensoring of operations F(x₀) ● F(x₁) ● ... ● F(xn)
        let sources = f.h.s.map_values(&f.h.w)?; // source types
        let targets = f.h.t.map_values(&f.h.w)?; // target types
        let fx = self.map_operations(&f.h.x, &sources, &targets)?;

        // Fw is the tensoring of objects F(w₀) ● F(w₁) ● ... ● F(wn)
        let fw = self.map_objects(&f.h.w)?;

        // Signature
        let (_w, x) = f.signature();
        // Identity map on wires of F(w)
        let i = OpenHypergraph::identity(fw.values.clone(), x.clone())?;

        let fs = map_half_spider(&fw, &f.s)?;
        let fe_s = map_half_spider(&fw, &f.h.s.values)?;
        let sx = OpenHypergraph::spider(fs, (i.t.clone() + fe_s)?, i.h.w.clone(), x.clone())?;

        let ft = map_half_spider(&fw, &f.t)?;
        let fe_t = map_half_spider(&fw, &f.h.t.values)?;
        let yt = OpenHypergraph::spider((i.s.clone() + fe_t)?, ft, i.h.w.clone(), x)?;

        ((sx >> (i | fx)?)? >> yt).map_err(Into::into)
    }
}
