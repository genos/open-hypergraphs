//! Hypergraph optics

use crate::{
    array::Backend,
    finite_function::{Error as FFError, FiniteFunction},
    functor::{Error as FError, FrobeniusFunctor, Functor},
    indexed_coproduct::{Error as ICError, IndexedCoproduct},
    open_hypergraph::{Error as OHError, OpenHypergraph},
};

/// Forward and reverse maps
pub trait Optic<A: Backend>: FrobeniusFunctor<A> {
    /// Forward map
    fn fwd(&self) -> impl FrobeniusFunctor<A>;
    /// Reverse map
    fn rev(&self) -> impl FrobeniusFunctor<A>;
    /// Compute the object M for each operation x[i] : a[i] → b[i]
    fn residual(
        &self,
        x: &FiniteFunction<A>,
        a: &IndexedCoproduct<A>,
        b: &IndexedCoproduct<A>,
    ) -> Result<IndexedCoproduct<A>, Error<A>>;
    /// Map objects
    fn map_objects(&self, objects: &FiniteFunction<A>) -> Result<IndexedCoproduct<A>, Error<A>> {
        let fa = self.fwd().map_objects(objects)?;
        let ra = self.rev().map_objects(objects)?;
        if fa.source() == ra.source() {
            let paired = (&fa + &ra)?;
            let p = FiniteFunction::transpose(2, fa.source());
            let sources = FiniteFunction {
                source: fa.sources.table.len(),
                target: usize::MAX,
                table: fa.sources.table + ra.sources.table,
            };
            let values = paired.map_indexes(&p)?.values;
            Ok(IndexedCoproduct { sources, values })
        } else {
            Err(Error::SourcesMismatch {
                fa: fa.clone(),
                ra: ra.clone(),
            })
        }
    }
    /// Map operations
    fn map_operations(
        &self,
        x: &FiniteFunction<A>,
        a: &IndexedCoproduct<A>,
        b: &IndexedCoproduct<A>,
    ) -> Result<OpenHypergraph<A>, Error<A>> {
        // F(x₀) ● F(x₁) ... F(xn)   :   FA₀ ● FA₁ ... FAn   →   (FB₀ ● M₀) ● (FB₁ ● M₁) ... (FBn ● Mn)
        let fwd = self.fwd().map_operations(x, a, b)?;

        // R(x₀) ● R(x₁) ... R(xn)   :   (M₀ ● RB₀) ● (M₁ ● RB₁) ... (Mn ● RBn)   →   RA₀ ● RA₁ ... RAn
        let rev = self.rev().map_operations(x, a, b)?;

        // We'll need these types to build identities and interleavings
        let fa = self.fwd().map_objects(&a.values)?;
        let fb = self.fwd().map_objects(&b.values)?;
        let ra = self.rev().map_objects(&a.values)?;
        let rb = self.rev().map_objects(&b.values)?;
        let m = self.residual(&x, &a, &b)?;

        // NOTE: we use flatmap here to ensure that each "block" of FB, which
        // might be e.g., F(B₀ ● B₁ ● ... ● Bn) is correctly interleaved:
        // consider that if M = I, then we would need to interleave
        let fwd_interleave = self
            .interleave_blocks(&b.flatmap(&fb)?, &m, &x.to_initial())?
            .dagger();
        let rev_cointerleave = self.interleave_blocks(&m, &b.flatmap(&rb)?, &x.to_initial())?;

        let i_fb = OpenHypergraph::identity(fb.clone().values, x.to_initial())?;
        let i_rb = OpenHypergraph::identity(rb.clone().values, x.to_initial())?;

        // Make this diagram "c":
        //
        //       ┌────┐
        //       │    ├──────────────── FB
        // FA ───┤ Ff │  M
        //       │    ├───┐  ┌────┐
        //       └────┘   └──┤    │
        //                   │ Rf ├──── RA
        // RB ───────────────┤    │
        //                   └────┘
        let lhs = ((fwd >> fwd_interleave)? | i_rb)?;
        let rhs = (i_fb | (rev_cointerleave >> rev)?)?;
        let c = (lhs >> rhs)?;

        // now adapt so that the wires labeled RB and RA are 'bent around'.
        let d = partial_dagger(&c, &fa, &fb, &ra, &rb)?;

        // finally interleave the FA with RA and FB with RB
        let lhs = self.interleave_blocks(&fa, &ra, &x.to_initial())?.dagger();
        let rhs = self.interleave_blocks(&fb, &rb, &x.to_initial())?;
        ((lhs >> d)? >> rhs).map_err(Into::into)
    }

    /// An OpenHypergraph whose source is A+B and whose target is the "interleaving" (A₀ + B₀) + (A₁ + B₁) + ... (An + Bn)
    fn interleave_blocks(
        &self,
        a: &IndexedCoproduct<A>,
        b: &IndexedCoproduct<A>,
        x: &FiniteFunction<A>,
    ) -> Result<OpenHypergraph<A>, Error<A>> {
        if a.source() != b.source() {
            Err(Error::InterleaveDiffLengths {
                a: a.clone(),
                b: b.clone(),
            })
        } else if x.source != 0 {
            Err(Error::NotIntial(x.clone()))
        } else {
            let ab = (a + b)?;
            let s = FiniteFunction::identity(ab.values.source);
            // NOTE: t is the dagger of transpose(2, N) because it appears in the target position!
            let t = ab
                .sources
                .injections(&FiniteFunction::transpose(2, a.source()))?;
            OpenHypergraph::spider(s, t, ab.values, x.clone()).map_err(Into::into)
        }
    }
}

/// Bend around the A₁ and B₁ wires of a map like c:
///         ┌─────┐
/// FA  ────┤     ├──── FB
///         │  c  │
/// RB  ────┤     ├──── RA
///         └─────┘
///
/// ... to get a map of type FA ● RA → FB ● RB
fn partial_dagger<A: Backend>(
    c: &OpenHypergraph<A>,
    fa: &IndexedCoproduct<A>,
    fb: &IndexedCoproduct<A>,
    ra: &IndexedCoproduct<A>,
    rb: &IndexedCoproduct<A>,
) -> Result<OpenHypergraph<A>, Error<A>> {
    let s_i = (FiniteFunction::inj0(fa.values.source, rb.values.source) >> &c.s)?;
    let s_o = (FiniteFunction::inj1(fb.values.source, ra.values.source) >> &c.t)?;
    let s = (s_i + s_o)?;

    let t_i = (FiniteFunction::inj0(fb.values.source, ra.values.source) >> &c.t)?;
    let t_o = (FiniteFunction::inj1(fa.values.source, rb.values.source) >> &c.s)?;
    let t = (t_i + t_o)?;

    Ok(OpenHypergraph {
        s,
        t,
        h: c.h.clone(),
    })
}

/// Errors that can occur when building or computing with functors.
#[derive(Debug, thiserror::Error)]
pub enum Error<A: Backend> {
    /// A finite function error occured
    #[error("A finite function error occured: {0}")]
    FiniteFunction(#[from] FFError<A>),
    /// A functor error occured
    #[error("A functor error occured: {0}")]
    Functor(#[from] FError<A>),
    /// A finite function error occured
    #[error("A indexed coproduct error occured: {0}")]
    IndexedCoproduct(#[from] ICError<A>),
    /// An open hypergraph error occured
    #[error("An open hypergraph error occured: {0}")]
    OpenHypergraph(#[from] OHError<A>),
    /// Sources mismatch between {fa} & {ra}
    #[error("Sources mismatch between {fa} & {ra}")]
    SourcesMismatch {
        /// Forward mapped objects
        fa: IndexedCoproduct<A>,
        /// Reverse mapped objects
        ra: IndexedCoproduct<A>,
    },
    /// Can't interleave types of unequal lengths, but got {a} & {b}
    #[error("Can't interleave types of unequal lengths, but got {a} & {b}")]
    InterleaveDiffLengths {
        /// First indexed coproduct
        a: IndexedCoproduct<A>,
        /// Second indexed coproduct
        b: IndexedCoproduct<A>,
    },
    /// Interleaving requires an initial x, but got {0}
    #[error("Interleaving requires an initial x, but got {0}")]
    NotIntial(FiniteFunction<A>),
}
