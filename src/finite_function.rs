//! Finite functions implemented as arrays.
//! Finite functions can be thought of as a thin wrapper around integer arrays whose elements are
//! within a specified range.
use crate::{array::Backend, macros::impl_arith};
use std::{
    fmt::{Debug, Display, Formatter},
    ops::{Add, BitOr, Index, Shr},
};

/// Finite functions implemented as arrays.
///
/// Finite functions can be thought of as a thin wrapper around integer arrays whose elements are
/// within a specified range.
/// Each `FiniteFunction` consists of a source (the domain as an unsigned integer), a target
/// (codomain as an unsigned integer), and a table of values respecting some invariants:
///  - |table| = source
///  - 0 ≤ t < target ∀ t ∈ table
#[derive(PartialEq, Eq, Clone)]
pub struct FiniteFunction<A: Backend> {
    pub(crate) source: usize,
    pub(crate) target: usize,
    pub(crate) table: A,
}

impl<A: Backend> FiniteFunction<A> {
    /// Check that the internal representation of this function is correct.
    ///
    /// # Errors
    ///
    /// If the source, target, and table don't correctly combine to describe a finite function.
    ///
    /// # References
    ///
    /// [John Regehr on assertions](https://blog.regehr.org/archives/1091)
    pub(crate) fn check_rep(&self) -> Result<(), Error<A>> {
        if self.table.len() != self.source {
            return Err(Error::IncorrectNumSource {
                num_values: self.table.len(),
                src: self.source,
            });
        }
        for i in 0..self.source {
            if self.table[i] >= self.target {
                return Err(Error::TargetValueTooBig {
                    too_big: self.table[i],
                    input: i,
                    target: self.target,
                });
            }
        }
        Ok(())
    }

    /// Safely construct a new `FiniteFunction`.
    ///
    /// # Errors
    ///
    /// If the source, target, and table don't correctly combine to describe a finite function.
    ///
    /// # Examples
    ///
    /// ```
    /// use open_hypergraphs::{Backend, FiniteFunction, StdVec};
    /// // Ok example
    /// let f = FiniteFunction::<StdVec>::new(3, 5, StdVec::array([0, 1, 4].into_iter()));
    /// assert!(f.is_ok());
    /// assert_eq!(f.unwrap().to_string(), "[0, 1, 4]: 3 → 5");
    /// // Too small a table
    /// let g = FiniteFunction::<StdVec>::new(3, 5, StdVec::arange(0, 2));
    /// assert!(!g.is_ok());
    /// assert_eq!(
    ///     format!("{}", g.unwrap_err()),
    ///     "The table has an incorrect number of values (2) to be from source 3."
    /// );
    /// // Target value out of range
    /// let h = FiniteFunction::<StdVec>::new(3, 5, StdVec::array([0, 1, 7].into_iter()));
    /// assert!(!h.is_ok());
    /// assert_eq!(
    ///     format!("{}", h.unwrap_err()),
    ///     "The target value 7 for input 2 is larger than or equal to 5."
    /// );
    /// ```
    pub fn new(source: usize, target: usize, table: A) -> Result<Self, Error<A>> {
        let result = Self {
            source,
            target,
            table,
        };
        result.check_rep()?;
        Ok(result)
    }

    /// The type of a finite function f: A → B is the pair (A, B).
    #[must_use]
    pub const fn get_type(&self) -> (usize, usize) {
        (self.source, self.target)
    }

    /// Compose this finite function with another.
    ///
    /// # Note
    ///
    /// This is also available via the `>>` operator, to suggest serial composition.
    ///
    /// # Errors
    ///
    /// If this function's target doesn't match the other's source.
    ///
    /// # Examples
    ///
    /// ```
    /// use open_hypergraphs::{Backend, FiniteFunction, StdVec};
    /// let f = FiniteFunction::new(3, 5, StdVec::array([0, 1, 4].into_iter())).unwrap();
    /// let g = FiniteFunction::new(5, 2, StdVec::array([0, 1, 0, 1, 0].into_iter())).unwrap();
    /// let fg0 = f.compose(&g);
    /// let fg1 = f >> g;
    /// assert!(fg0.is_ok());
    /// assert!(fg1.is_ok());
    /// assert_eq!(fg0.as_ref().unwrap(), fg1.as_ref().unwrap());
    /// assert_eq!(fg0.unwrap().to_string(), "[0, 1, 0]: 3 → 2");
    /// ```
    pub fn compose(&self, other: &Self) -> Result<Self, Error<A>> {
        if self.target == other.source {
            let table = other.table.select(&self.table);
            Ok(Self {
                source: self.source,
                target: other.target,
                table,
            })
        } else {
            Err(Error::CompositionMismatch {
                f: self.clone(),
                g: other.clone(),
            })
        }
    }

    /// Given another function with the same target, compute the coproduct; specifically, if our
    /// function is f: A₀ → B, given another g: A₁ → B, this computes f.coproduct(g): A₀ + A₁ → B.
    ///
    /// # Note
    ///
    /// This is also available via the `+` operator.
    ///
    /// # Errors
    ///
    /// If this function's target doesn't match the other's target.
    ///
    /// # Examples
    ///
    /// ```
    /// use open_hypergraphs::{Backend, FiniteFunction, StdVec};
    /// let f = FiniteFunction::<StdVec>::new(3, 5, StdVec::array([0, 1, 4].into_iter())).unwrap();
    /// let g = FiniteFunction::<StdVec>::new(2, 5, StdVec::array([0, 2].into_iter())).unwrap();
    /// let fg0 = f.coproduct(&g);
    /// let fg1 = f + g;
    /// assert!(fg0.is_ok());
    /// assert!(fg1.is_ok());
    /// assert_eq!(fg0.as_ref().unwrap(), fg1.as_ref().unwrap());
    /// assert_eq!(fg0.unwrap().to_string(), "[0, 1, 4, 0, 2]: 5 → 5");
    /// ```
    pub fn coproduct(&self, other: &Self) -> Result<Self, Error<A>> {
        if self.target == other.target {
            let table = self.table.append(&other.table);
            Ok(Self {
                source: table.len(),
                target: self.target,
                table,
            })
        } else {
            Err(Error::CoproductMismatch {
                f: self.clone(),
                g: other.clone(),
            })
        }
    }

    /// Computes the tensor product; if our function is f: A₀ → B₀, given g: A₁ → B₁, this computes
    /// f.tensor(g): A₀ + A₁ → B₀ + B₁.
    ///
    /// # Note
    ///
    /// This is also available via the `|` operator, to suggest parallel composition.
    ///
    /// # Examples
    ///
    /// ```
    /// use open_hypergraphs::{Backend, FiniteFunction, StdVec};
    /// let f = FiniteFunction::<StdVec>::new(3, 5, StdVec::array([0, 1, 4].into_iter())).unwrap();
    /// let g = FiniteFunction::<StdVec>::new(2, 4, StdVec::arange(2, 4)).unwrap();
    /// let fg0 = f.tensor(&g);
    /// let fg1 = f | g;
    /// assert_eq!(fg0, fg1);
    /// assert_eq!(fg0.to_string(), "[0, 1, 4, 7, 8]: 5 → 9");
    /// ```
    #[must_use]
    pub fn tensor(&self, other: &Self) -> Self {
        let table = self.table.append(&(other.table.clone() + self.target));
        Self {
            source: table.len(),
            target: self.target + other.target,
            table,
        }
    }

    /// Directly compute (f ; ι₀) instead of by composition.
    #[must_use]
    pub fn inject0(&self, b: usize) -> Self {
        Self {
            source: self.source,
            target: self.target + b,
            table: self.table.clone(),
        }
    }

    /// Directly compute (f ; ι1) instead of by composition.
    #[must_use]
    pub fn inject1(&self, a: usize) -> Self {
        Self {
            source: self.source,
            target: self.target + a,
            table: self.table.clone() + a,
        }
    }

    /// The identity finite function of type N → N.
    ///
    /// # Examples
    ///
    /// ```
    /// use open_hypergraphs::{FiniteFunction, StdVec};
    /// assert_eq!(FiniteFunction::<StdVec>::identity(2).to_string(), "[0, 1]: 2 → 2");
    /// ```
    #[must_use]
    pub fn identity(n: usize) -> Self {
        Self {
            source: n,
            target: n,
            table: A::arange(0, n),
        }
    }

    /// The initial map ?: 0 → B.
    ///
    /// # Examples
    ///
    /// ```
    /// use open_hypergraphs::{FiniteFunction, StdVec};
    /// assert_eq!(FiniteFunction::<StdVec>::initial(3).to_string(), "[]: 0 → 3");
    /// ```
    #[must_use]
    pub fn initial(n: usize) -> Self {
        Self {
            source: 0,
            target: n,
            table: A::empty(),
        }
    }

    /// Turn a finite function f: A → B into the initial map ?: 0 → B.
    #[must_use]
    pub fn to_initial(&self) -> Self {
        Self::initial(self.target)
    }

    /// The terminal map !: A → 1.
    ///
    /// # Examples
    ///
    /// ```
    /// use open_hypergraphs::{FiniteFunction, StdVec};
    /// assert_eq!(FiniteFunction::<StdVec>::terminal(2).to_string(), "[0, 0]: 2 → 1");
    /// ```
    #[must_use]
    pub fn terminal(n: usize) -> Self {
        Self {
            source: n,
            target: 1,
            table: A::zeros(n),
        }
    }

    /// The injection ι₀: A → A + B.
    #[must_use]
    pub fn inj0(a: usize, b: usize) -> Self {
        Self {
            source: a,
            target: a + b,
            table: A::arange(0, a),
        }
    }

    /// The injection ι₁: B → A + B.
    #[must_use]
    pub fn inj1(a: usize, b: usize) -> Self {
        Self {
            source: b,
            target: a + b,
            table: A::arange(a, a + b),
        }
    }

    /// The singleton array containing just x whose domain is B.
    ///
    /// # Errors
    ///
    /// If the value is too large for the target codomain.
    ///
    /// # Examples
    ///
    /// ```
    /// use open_hypergraphs::{FiniteFunction, StdVec};
    /// let f = FiniteFunction::<StdVec>::singleton(2, 5);
    /// assert!(f.is_ok());
    /// assert_eq!(f.unwrap().to_string(), "[2]: 1 → 5");
    /// ```
    pub fn singleton(x: usize, b: usize) -> Result<Self, Error<A>> {
        Self::constant(x, 1, b)
    }

    /// The constant function of type A → B mapping all inputs to the value x.
    ///
    /// # Errors
    ///
    /// If the value is too large for the target codomain.
    ///
    /// # Examples
    ///
    /// ```
    /// use open_hypergraphs::{FiniteFunction, StdVec};
    /// let f = FiniteFunction::<StdVec>::constant(1, 2, 3);
    /// assert!(f.is_ok());
    /// assert_eq!(f.unwrap().to_string(), "[1, 1]: 2 → 3");
    /// ```
    pub fn constant(x: usize, a: usize, b: usize) -> Result<Self, Error<A>> {
        if x < b {
            Ok(Self {
                source: a,
                target: b,
                table: A::constant(x, a),
            })
        } else {
            Err(Error::TargetValueTooBig {
                too_big: x,
                input: 0,
                target: b,
            })
        }
    }

    /// A permutation as the array whose ith position denotes "where to send" value i.
    ///
    /// # Examples
    ///
    /// ```
    /// use open_hypergraphs::{FiniteFunction, StdVec};
    /// assert_eq!(FiniteFunction::<StdVec>::twist(2, 3).to_string(), "[3, 4, 0, 1, 2]: 5 → 5");
    /// ```
    #[must_use]
    pub fn twist(a: usize, b: usize) -> Self {
        Self {
            source: a + b,
            target: a + b,
            table: A::arange(b, b + a).append(&A::arange(0, b)),
        }
    }

    /// The permutation that would stably sort this finite function's array representation; i.e.
    /// for the function f: A → B, return the stable sorting permutation p: A → A such that p ; f
    /// is monotonic.
    #[must_use]
    pub fn argsort(&self) -> Self {
        Self {
            source: self.source,
            target: self.source,
            table: self.table.argsort(),
        }
    }

    /// Given an b*a-dimensional input thought of as a matrix in row-major order with b
    /// rows and a columns, compute the "target indices" in the transpose.
    ///
    /// So if we have a matrix M: A → B and a matrix N: B → A, then setting indexes
    /// N[transpose(a, b)] = M is the same as writing N = M.T
    #[must_use]
    pub fn transpose(a: usize, b: usize) -> Self {
        let mut table = A::zeros(a * b);
        for i in 0..table.len() {
            table[i] = (i % a) * b + i / a;
        }
        Self {
            source: a * b,
            target: a * b,
            table,
        }
    }

    /// Given finite functions f, g: A → B, return the coequalizer q: B → Q which is the unique
    /// arrow such that f ; q = g ; q having a unique arrow to any other such map.
    ///
    /// # Errors
    ///
    /// If the two functions aren't of the same type.
    pub fn coequalizer(&self, other: &Self) -> Result<Self, Error<A>> {
        if self.get_type() == other.get_type() {
            let (num_components, table) =
                A::connected_components(&self.table, &other.table, self.target);
            Ok(Self {
                source: table.len(),
                target: num_components,
                table,
            })
        } else {
            Err(Error::CoequalizerMismatch {
                f: self.clone(),
                g: other.clone(),
            })
        }
    }

    /// Given a coequalizer q: B → Q of morphisms a, b: A → B and some f: B → B' such that f(a) =
    /// f(b), Compute the universal map u: Q → B' such that q ; u = f.
    ///
    /// # Errors
    ///
    /// If the constructed universal morphism is ill-formed, if the computation of q ; u fails, or
    /// if q ; u ≠ f.
    pub fn coequalizer_universal(&self, other: &Self) -> Result<Self, Error<A>> {
        if self.source == other.source {
            let mut table = A::zeros(self.target);
            for i in 0..self.table.len() {
                table[self.table[i]] = other.table[i];
            }
            let u = Self {
                source: table.len(),
                target: other.target,
                table,
            };
            if (self >> &u)? == *other {
                Ok(u)
            } else {
                Err(Error::DoesntCommute { q: self.clone(), u })
            }
        } else {
            Err(Error::DifferentLengths {
                first: self.table.clone(),
                second: other.table.clone(),
            })
        }
    }

    /// Compute the coproduct of a collection of finite functions. O(n) in the size of the result.
    ///
    /// # Errors
    ///
    /// If any of the finite functions don't have the same target.
    ///
    /// # Examples
    ///
    /// ```
    /// use open_hypergraphs::{Backend, FiniteFunction, StdVec};
    /// let f = FiniteFunction::coproduct_of(
    ///     std::iter::repeat(FiniteFunction::<StdVec>::identity(3)).take(3)
    /// );
    /// assert!(f.is_ok());
    /// assert_eq!(f.unwrap().to_string(), "[0, 1, 2, 0, 1, 2, 0, 1, 2]: 9 → 3");
    /// let g = FiniteFunction::coproduct_of(
    ///     [
    ///         FiniteFunction::<StdVec>::new(0, 27, StdVec::empty()).unwrap(),
    ///         FiniteFunction::<StdVec>::new(1, 27, StdVec::zeros(1)).unwrap(),
    ///         FiniteFunction::<StdVec>::new(1, 27, StdVec::ones(1)).unwrap(),
    ///     ].into_iter()
    /// );
    /// assert!(g.is_ok());
    /// assert_eq!(g.unwrap().to_string(), "[0, 1]: 2 → 27");
    /// ```
    pub fn coproduct_of(mut fs: impl Iterator<Item = Self>) -> Result<Self, Error<A>> {
        if let Some(mut f) = fs.next() {
            for g in fs {
                f = f.coproduct(&g)?;
            }
            Ok(f)
        } else {
            Ok(Self::initial(0))
        }
    }

    /// Compute the tensor product of a collection of finite functions. O(n) in the size of the
    /// result.
    /// # Examples
    ///
    /// ```
    /// use open_hypergraphs::{FiniteFunction, StdVec};
    /// let f = FiniteFunction::<StdVec>::tensor_of(
    ///     std::iter::repeat(FiniteFunction::<StdVec>::identity(3)).take(3))
    /// ;
    /// assert_eq!(f.to_string(), "[0, 1, 2, 3, 4, 5, 6, 7, 8]: 9 → 9");
    /// ```
    pub fn tensor_of(fs: impl Iterator<Item = Self>) -> Self {
        fs.fold(Self::initial(0), |f, g| f | g)
    }

    /// Given a finite function s: N → K representing the objects of the coproduct ∑_{n ∈ N} s(n)
    /// whose injections have the type ιₓ: s(x) → ∑_{n ∈ N} s(n), and another finite map a: A → N,
    /// compute the coproduct of injections
    ///     injections(s, a): Σ_{x ∈ A} s(x) → Σ_{n ∈ N} s(n) = Σ_{x ∈ A} ιₐ(x)
    /// so that injections(s, id) = id.
    /// Note that when a is a permutation, injections(s, a) is a blockwise version of that
    /// permutation with block sizes equal to s.
    ///
    /// # Errors
    ///
    /// If the other function doesn't (pre) compose with this one.
    pub fn injections(&self, other: &Self) -> Result<Self, Error<A>> {
        let p = self.table.prefix_sum();
        let k = other.compose(self)?;
        let r = k.table.segmented_arange();
        let target = p[p.len() - 1];
        let permuted = p.select(&other.table);
        let table = permuted.repeat(&k.table) + r;
        Ok(Self {
            source: table.len(),
            target,
            table,
        })
    }
}

impl<A: Backend> Index<usize> for FiniteFunction<A> {
    type Output = usize;
    fn index(&self, i: usize) -> &Self::Output {
        self.table.index(i)
    }
}

impl<A: Backend> Display for FiniteFunction<A> {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "{}: {} → {}", self.table, self.source, self.target)
    }
}

impl<A: Backend> Debug for FiniteFunction<A> {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        // Reuse Display cuz it's nice
        std::fmt::Display::fmt(self, f)
    }
}

impl_arith!(FiniteFunction, Shr, shr, compose, true);
impl_arith!(FiniteFunction, Add, add, coproduct, true);
impl_arith!(FiniteFunction, BitOr, bitor, tensor, false);

/// Errors that can arise when building or computing with finite functions.
#[derive(Debug, thiserror::Error)]
pub enum Error<A: Backend> {
    /// The table has an incorrect number of values ({`num_values`}) to be from source {`src`}.
    #[error("The table has an incorrect number of values ({num_values}) to be from source {src}.")]
    IncorrectNumSource {
        /// The number of values in the table
        num_values: usize,
        /// The source domain
        src: usize,
    },
    /// The target value {`too_big`} for input {`input`} is larger than or equal to {`target`}.
    #[error("The target value {too_big} for input {input} is larger than or equal to {target}.")]
    TargetValueTooBig {
        /// The value that was too big
        too_big: usize,
        /// The input (index in the table) of that value
        input: usize,
        /// The target codomain
        target: usize,
    },
    /// In the composition, the intermediate target {`f`} does not match the intermediate source of {`g`}.
    #[error("In the composition, the intermediate target of {f} does not match the intermediate source of {g}.")]
    CompositionMismatch {
        /// First function
        f: FiniteFunction<A>,
        /// Second function
        g: FiniteFunction<A>,
    },
    /// In the coproduct, the target of {`f`} does not match the target of {`g`}.
    #[error("In the coproduct, the target of {f} does not match the target of {g}.")]
    CoproductMismatch {
        /// The first function
        f: FiniteFunction<A>,
        /// The second function
        g: FiniteFunction<A>,
    },
    /// In the coequalizer, the type of {`f`} doesn't match the type of {`g`}.
    #[error("In the coequalizer, the type of {f} doesn't match the type of {g}.")]
    CoequalizerMismatch {
        /// The first function
        f: FiniteFunction<A>,
        /// The second function
        g: FiniteFunction<A>,
    },
    /// The universal morphism doesn't make {`q`} ; {`u`} commute. Is q really a coequalizer?
    #[error("The universal morphism doesn't make {q} ; {u} commute. Is q really a coequalizer?")]
    DoesntCommute {
        /// Supposed coequalizer
        q: FiniteFunction<A>,
        /// Supposed universal morphism
        u: FiniteFunction<A>,
    },
    /// The tables {`first`} and {`second`} have different lengths.
    #[error("The tables {first} and {second} have different lengths.")]
    DifferentLengths {
        /// The first table
        first: A,
        /// The second table
        second: A,
    },
}

/// `proptest` strategies for generating arbitray finite functions.
#[cfg(test)]
pub(crate) mod strategies {
    use super::*;
    use crate::array::strategies as a;
    use proptest::prelude::*;

    pub fn object(allow_initial: bool) -> impl Strategy<Value = usize> {
        usize::from(!allow_initial)..32
    }

    pub fn arrows<A: Backend>() -> impl Strategy<Value = FiniteFunction<A>> {
        (object(true), object(true))
            .prop_flat_map(|(s, t)| {
                let s = if t == 0 { 0 } else { s };
                (Just(s), Just(t), a::array(0, t, s))
            })
            .prop_map(|(source, target, table)| FiniteFunction {
                source,
                target,
                table,
            })
    }

    pub fn arrows_nz<A: Backend>() -> impl Strategy<Value = FiniteFunction<A>> {
        (object(false), object(false))
            .prop_flat_map(|(s, t)| {
                let s = if t == 0 { 0 } else { s };
                (Just(s), Just(t), a::array(0, t, s))
            })
            .prop_map(|(source, target, table)| FiniteFunction {
                source,
                target,
                table,
            })
    }

    pub fn arrow<A: Backend>(
        source: usize,
        target: usize,
    ) -> impl Strategy<Value = FiniteFunction<A>> {
        (Just(source), Just(target), a::array(0, target, source)).prop_map(
            |(source, target, table)| FiniteFunction {
                source,
                target,
                table,
            },
        )
    }

    pub fn arrow_from<A: Backend>(
        source: usize,
    ) -> impl Strategy<Value = FiniteFunction<A>> {
        (Just(source), object(false)).prop_flat_map(|(s, t)| arrow(s, t))
    }

    pub fn arrow_to<A: Backend>(target: usize) -> impl Strategy<Value = FiniteFunction<A>> {
        (object(true), Just(target)).prop_flat_map(|(s, t)| {
            let s = if t == 0 { 0 } else { s };
            arrow(s, t)
        })
    }

    pub fn permutation<A: Backend>(n: usize) -> impl Strategy<Value = FiniteFunction<A>> {
        a::permutation(n).prop_map(|table: A| {
            let source = table.len();
            let target = table.len();
            FiniteFunction {
                source,
                target,
                table,
            }
        })
    }

    pub fn two_parallel<A: Backend>(
    ) -> impl Strategy<Value = (FiniteFunction<A>, FiniteFunction<A>)> {
        arrows().prop_flat_map(|f| (Just(f.clone()), arrow(f.source, f.target)))
    }

    pub fn two_composite<A: Backend>(
    ) -> impl Strategy<Value = (FiniteFunction<A>, FiniteFunction<A>)> {
        object(true)
            .prop_flat_map(arrow_from)
            .prop_flat_map(|f| (Just(f.clone()), arrow_from(f.target)))
    }

    pub fn three_composite<A: Backend>(
    ) -> impl Strategy<Value = (FiniteFunction<A>, FiniteFunction<A>, FiniteFunction<A>)> {
        two_composite().prop_flat_map(|(f, g)| (Just(f), Just(g.clone()), arrow_from(g.target)))
    }

    pub fn same_target<A: Backend>() -> impl Strategy<Value = Vec<FiniteFunction<A>>> {
        (object(true), object(true))
            .prop_flat_map(|(target, num)| proptest::collection::vec(arrow_to(target), num))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::{NDArray1Usize, StdVec};
    use paste::paste;

    macro_rules! by_hand {
        ($A:ty) => {
            paste! {
                mod [<$A:snake:lower _finite_function_by_hand>] {
                    use super::*;
                    #[test]
                    fn id_ok() {
                        let i: FiniteFunction<$A> = FiniteFunction::identity(3);
                        assert!(i.check_rep().is_ok());
                        assert_eq!(i.source, 3);
                        assert_eq!(i.target, 3);
                        assert_eq!(i.table, $A::arange(0, 3));
                        assert_eq!(i.to_string(), "[0, 1, 2]: 3 → 3");
                    }

                    #[test]
                    fn initial_ok() {
                        let i: FiniteFunction<$A> = FiniteFunction::initial(3);
                        assert!(i.check_rep().is_ok());
                        assert_eq!(i.source, 0);
                        assert_eq!(i.target, 3);
                        assert_eq!(i.table.len(), 0);
                        assert_eq!(i.to_string(), "[]: 0 → 3");
                    }

                    #[test]
                    fn terminal_ok() {
                        let t: FiniteFunction<$A> = FiniteFunction::terminal(3);
                        assert!(t.check_rep().is_ok());
                        assert_eq!(t.source, 3);
                        assert_eq!(t.target, 1);
                        assert_eq!(t.table, $A::zeros(3));
                        assert_eq!(t.to_string(), "[0, 0, 0]: 3 → 1");
                    }

                    #[test]
                    fn inj0_ok() {
                        assert!(FiniteFunction::<$A>::inj0(3, 5).check_rep().is_ok());
                        let f: FiniteFunction<$A> = FiniteFunction::identity(3);
                        let g = &f >> FiniteFunction::inj0(f.source, 5);
                        assert!(g.is_ok());
                        assert_eq!(f.inject0(5), g.unwrap());
                    }

                    #[test]
                    fn inj1_ok() {
                        assert!(FiniteFunction::<$A>::inj1(3, 5).check_rep().is_ok());
                        let f: FiniteFunction<$A> = FiniteFunction::identity(3);
                        let g = &f >> FiniteFunction::inj1(5, f.target);
                        assert!(g.is_ok());
                        assert_eq!(f.inject1(5), g.unwrap());
                    }

                    #[test]
                    fn tensor_ok() {
                        let f: FiniteFunction<$A> = FiniteFunction::identity(3);
                        let g = FiniteFunction::identity(5);
                        let h = f.inject0(g.target) + g.inject1(f.target);
                        assert!(h.is_ok());
                        assert_eq!(f | g, h.unwrap());
                    }
                }
            }
        };
    }

    macro_rules! properties {
        ($A:ty) => {
                paste! {
                mod [<$A:snake:lower _finite_function_properties>] {
                    use super::*;
                    use proptest::prelude::*;
                    proptest! {
                        #[test]
                        fn equality_reflexive(f in strategies::arrows::<$A>()) {
                            prop_assert!(f.check_rep().is_ok());
                            prop_assert_eq!(f.clone(), f);
                        }

                        #[test]
                        fn equality_of_parallel((f, g) in strategies::two_parallel::<$A>()) {
                            prop_assert!(f.check_rep().is_ok());
                            prop_assert!(g.check_rep().is_ok());
                            prop_assert_eq!(f == g, f.table == g.table);
                        }

                        #[test]
                        fn initial_unique(f in strategies::arrow_to::<$A>(0)) {
                            prop_assert!(f.check_rep().is_ok());
                            prop_assert_eq!(FiniteFunction::initial(f.source), f);
                        }

                        #[test]
                        fn category_id_left(f in strategies::arrows::<$A>()) {
                            prop_assert!(f.check_rep().is_ok());
                            let id = FiniteFunction::identity(f.source);
                            prop_assert!(id.check_rep().is_ok());
                            let g = id >> &f;
                            prop_assert!(g.is_ok());
                            prop_assert_eq!(f, g.unwrap());
                        }

                        #[test]
                        fn category_id_right(f in strategies::arrows::<$A>()) {
                            prop_assert!(f.check_rep().is_ok());
                            let id = FiniteFunction::identity(f.target);
                            prop_assert!(id.check_rep().is_ok());
                            let g = &f >> id;
                            prop_assert!(g.is_ok());
                            prop_assert_eq!(f, g.unwrap());
                        }

                        #[test]
                        fn composition_requires_match(f in strategies::arrows::<$A>(), g in strategies::arrows()) {
                            prop_assert!(f.check_rep().is_ok());
                            prop_assert!(g.check_rep().is_ok());
                            prop_assert_eq!(f.target == g.source, (f >> g).is_ok());
                        }

                        #[test]
                        fn fuzz_composition((f, g) in strategies::two_composite::<$A>()) {
                            prop_assert!(f.check_rep().is_ok());
                            prop_assert!(g.check_rep().is_ok());
                            prop_assert!((f >> g).is_ok());
                        }

                        #[test]
                        fn category_associative((f, g, h) in strategies::three_composite::<$A>()) {
                            prop_assert!(f.check_rep().is_ok());
                            prop_assert!(g.check_rep().is_ok());
                            prop_assert!(h.check_rep().is_ok());
                            let left = (&f >> &g)? >> &h;
                            prop_assert!(left.is_ok());
                            let right = f >> (g >> h)?;
                            prop_assert!(right.is_ok());
                            prop_assert_eq!(left.unwrap(), right.unwrap());
                        }

                        #[test]
                        fn tensor_vs_injections(f in strategies::arrows::<$A>(), g in strategies::arrows::<$A>()) {
                            prop_assert!(f.check_rep().is_ok());
                            prop_assert!(g.check_rep().is_ok());
                            let h = f.inject0(g.target) + g.inject1(f.target);
                            prop_assert!(h.is_ok());
                            prop_assert_eq!(f | g, h.unwrap());
                        }

                        #[test]
                        fn twist_inverse(a in strategies::object(true), b in strategies::object(true)) {
                            let f: FiniteFunction<$A> = FiniteFunction::twist(a, b);
                            let g: FiniteFunction<$A> = FiniteFunction::twist(b, a);
                            prop_assert!(f.check_rep().is_ok());
                            prop_assert!(g.check_rep().is_ok());
                            let (lhs, rhs) = (&f >> &g, &g >> &f);
                            prop_assert!(lhs.is_ok());
                            prop_assert!(rhs.is_ok());
                            let id = FiniteFunction::identity(a + b);
                            prop_assert!(id.check_rep().is_ok());
                            prop_assert_eq!(&id, &lhs.unwrap());
                            prop_assert_eq!(&id, &rhs.unwrap());
                        }

                        #[test]
                        fn argsort_monotonic(f in strategies::arrows::<$A>()) {
                            prop_assert!(f.check_rep().is_ok());
                            let p = f.argsort();
                            prop_assert!(p.check_rep().is_ok());
                            let g = p >> f;
                            prop_assert!(g.is_ok());
                            let g = g.unwrap();
                            if !g.table.is_empty() {
                                for i in 0..g.table.len() - 1 {
                                    prop_assert!(g.table[i] <= g.table[i + 1]);
                                }
                            }
                        }

                        #[test]
                        fn transpose_inverse(a in strategies::object(true), b in strategies::object(true)) {
                            let f: FiniteFunction<$A> = FiniteFunction::transpose(a, b);
                            let g: FiniteFunction<$A> = FiniteFunction::transpose(b, a);
                            let id = FiniteFunction::identity(a * b);
                            prop_assert!(f.check_rep().is_ok());
                            prop_assert!(g.check_rep().is_ok());
                            prop_assert!(id.check_rep().is_ok());
                            let fg = f >> g;
                            prop_assert!(fg.is_ok());
                            prop_assert_eq!(fg.unwrap(), id);
                        }

                        #[test]
                        fn fuzz_coproduct_of(fs in strategies::same_target::<$A>()) {
                            prop_assert!(FiniteFunction::coproduct_of(fs.into_iter()).is_ok());
                        }

                        #[test]
                        fn fuzz_injections((f, g) in strategies::two_composite::<$A>()) {
                            prop_assert!(f.check_rep().is_ok());
                            prop_assert!(g.check_rep().is_ok());
                            prop_assert!(g.injections(&f).is_ok());
                        }

                        #[test]
                        fn fuzz_cc((a, b) in strategies::two_parallel::<$A>()) {
                            let n = a.target;
                            let (num_components, labels) = $A::connected_components(&a.table, &b.table, n);
                            prop_assert!(num_components <= n);
                            for i in 0..labels.len() {
                                prop_assert!(labels[i] < n);
                            }
                        }

                    }
                }
            }
        }
    }

    by_hand!(StdVec);
    properties!(StdVec);
    by_hand!(NDArray1Usize);
    properties!(NDArray1Usize);
}
