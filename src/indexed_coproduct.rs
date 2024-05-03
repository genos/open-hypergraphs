//! A finite coproduct of finite functions.
use crate::{
    array::Backend,
    finite_function::{Error as FFError, FiniteFunction},
    macros::impl_arith,
};
use std::{
    convert::TryFrom,
    fmt::{Debug, Display, Formatter},
    ops::{Add, BitOr, Shr},
};

/// A finite coproduct of finite functions.
///
/// You can think of it simply as a segmented array.
/// Categorically, it represents a finite coproduct Σ_{i ∈ N} fᵢ: s(fᵢ) → Y as a pair of maps:
/// - sources: N → ℕ  (which means we ignore `sources.target` or set it to `usize::MAX`)
/// - values: ∑ sources → ∑₀
#[derive(Eq, Clone)]
pub struct IndexedCoproduct<A: Backend> {
    /// An array of segment sizes
    pub(crate) sources: FiniteFunction<A>,
    /// The values of the coproduct
    pub(crate) values: FiniteFunction<A>,
}

/// Custom implementation to ignore `sources.target`
impl<A: Backend> PartialEq for IndexedCoproduct<A> {
    fn eq(&self, other: &Self) -> bool {
        self.sources.source == other.sources.source
            && self.sources.table == other.sources.table
            && self.values.table == other.values.table
    }
}

impl<A: Backend> IndexedCoproduct<A> {
    /// Check that the internal representation of this indexed coproduct is correct.
    ///
    /// # Errors
    /// If the sources and values don't correctly combine to describe an indexed coproduct.
    ///
    /// # References
    /// [John Regehr on assertions](https://blog.regehr.org/archives/1091)
    pub(crate) fn check_rep(&self) -> Result<(), Error<A>> {
        let sum = self.sources.table.sum();
        if self.values.source == sum {
            Ok(())
        } else {
            Err(Error::SourcesSumValuesSourceMismatch {
                sum,
                values: self.values.clone(),
            })
        }
    }

    /// Safely construct a new `IndexedCoproduct`.
    ///
    /// # Errors
    /// If the sources and values don't correctly combine to describe an indexed coproduct.
    ///
    /// # Examples
    ///
    /// ```
    /// use open_hypergraphs::{FiniteFunction, IndexedCoproduct, StdVec};
    /// // Ok example
    /// let sources = FiniteFunction::<StdVec>::identity(3);
    /// let values = FiniteFunction::<StdVec>::twist(2, 1);
    /// assert!(IndexedCoproduct::new(sources, values).is_ok());
    /// // Mismatch between sum of sources table and source of values table
    /// let sources = FiniteFunction::<StdVec>::identity(3);
    /// let values = FiniteFunction::<StdVec>::identity(5);
    /// assert!(!IndexedCoproduct::new(sources, values).is_ok());
    /// ```
    pub fn new(sources: FiniteFunction<A>, values: FiniteFunction<A>) -> Result<Self, Error<A>> {
        let result = Self { sources, values };
        result.check_rep()?;
        Ok(result)
    }

    /// The domain of this indexed coproduct.
    ///
    /// # Examples
    ///
    /// ```
    /// use open_hypergraphs::{IndexedCoproduct, StdVec};
    /// assert_eq!(IndexedCoproduct::<StdVec>::initial(3).source(), 0);
    /// ```
    #[must_use]
    pub const fn source(&self) -> usize {
        self.sources.source
    }

    /// The codomain of this indexed coproduct.
    ///
    /// # Examples
    ///
    /// ```
    /// use open_hypergraphs::{IndexedCoproduct, StdVec};
    /// assert_eq!(IndexedCoproduct::<StdVec>::initial(3).target(), 3);
    /// ```
    #[must_use]
    pub const fn target(&self) -> usize {
        self.values.target
    }

    /// The initial map.
    ///
    /// # Examples
    ///
    /// ```
    /// use open_hypergraphs::{IndexedCoproduct, StdVec};
    /// assert_eq!(IndexedCoproduct::<StdVec>::initial(3).to_string(), "∑([]: 0 → ℕ): → ([]: 0 → 3)");
    /// ```
    #[must_use]
    pub fn initial(n: usize) -> Self {
        Self {
            sources: FiniteFunction::initial(usize::MAX),
            values: FiniteFunction::initial(n),
        }
    }

    /// Turn a `FiniteFunction` f: A → B into an `IndexedCoproduct` ∑_{i ∈ 1} fᵢ: A → B.
    ///
    /// # Panics
    ///
    /// If there is an issue constructing a singleton `FiniteFunction` from A and ℕ, but there
    /// shouldn't be one.
    ///
    /// # Examples
    ///
    /// ```
    /// use open_hypergraphs::{FiniteFunction, IndexedCoproduct, StdVec};
    /// let i = IndexedCoproduct::singleton(FiniteFunction::<StdVec>::identity(3));
    /// assert_eq!(i.to_string(), "∑([3]: 1 → ℕ): → ([0, 1, 2]: 3 → 3)");
    /// ```
    #[must_use]
    pub fn singleton(values: FiniteFunction<A>) -> Self {
        let sources = FiniteFunction::singleton(values.source, usize::MAX).unwrap();
        Self { sources, values }
    }

    /// Turn a `FiniteFunction` f: A → B into an `IndexedCoproduct` ∑_{a ∈ A} fₐ: A → B.
    ///
    /// # Errors
    ///
    /// If there is an issue constructing a singleton `FiniteFunction` from A and B, like if the
    /// finite function is empty or has only one element.
    ///
    /// # Examples
    ///
    /// ```
    /// use open_hypergraphs::{FiniteFunction, IndexedCoproduct, StdVec};
    /// let i = IndexedCoproduct::elements(FiniteFunction::<StdVec>::identity(3));
    /// assert!(i.is_ok());
    /// assert_eq!(i.unwrap().to_string(), "∑([1, 1, 1]: 3 → ℕ): → ([0, 1, 2]: 3 → 3)");
    /// assert!(IndexedCoproduct::elements(FiniteFunction::<StdVec>::initial(0)).is_err());
    /// assert!(IndexedCoproduct::elements(FiniteFunction::<StdVec>::terminal(1)).is_err());
    /// ```
    pub fn elements(values: FiniteFunction<A>) -> Result<Self, Error<A>> {
        let sources = FiniteFunction::constant(1, values.source, values.source)?;
        Ok(Self { sources, values })
    }

    /// Computes the coproduct of two indexed coproducts.
    ///
    /// # Note
    ///
    /// This is also available via the `+` operator.
    ///
    ///
    /// # Errors
    ///
    /// If the sources or values of these indexed coproducts are incompatible
    pub fn coproduct(&self, other: &Self) -> Result<Self, Error<A>> {
        Ok(Self {
            sources: (&self.sources + &other.sources)?,
            values: (&self.values + &other.values)?,
        })
    }

    /// Computes the tensor product of two indexed coproducts.
    ///
    /// # Note
    ///
    /// This is also available via the `|`, to suggest parallel composition.
    ///
    /// # Errors
    ///
    /// If the sources of the these indexed coproducts are incompatible.
    pub fn tensor(&self, other: &Self) -> Result<Self, Error<A>> {
        Ok(Self {
            sources: (&self.sources + &other.sources)?,
            values: &self.values | &other.values,
        })
    }

    /// Compose this indexed coproduct with another.
    ///
    /// # Note
    ///
    /// This is also available via the `>>` operator, to suggest serial composition.
    ///
    /// # Errors
    /// If the source of this coproduct's values is not the same as the source of the other.
    pub fn compose(&self, other: &Self) -> Result<Self, Error<A>> {
        self.flatmap(other)
    }

    /// Compose this indexed coproduct with another.
    ///
    /// # Note
    ///
    /// This is also available via the `>>` operator, to suggest serial composition.
    ///
    /// # Errors
    ///
    /// If the source of this coproduct's values is not the same as the source of the other.
    pub fn flatmap(&self, other: &Self) -> Result<Self, Error<A>> {
        if self.values.source == other.source() {
            let ss = self.sources.table.segmented_sum(&other.sources.table);
            let sources = FiniteFunction::new(ss.len(), usize::MAX, ss)?;
            let values = other.values.clone();
            Ok(Self { sources, values })
        } else {
            Err(Error::SourceMistmatch {
                values: self.values.clone(),
                src_0: self.values.source,
                src_1: other.source(),
            })
        }
    }

    /// Compute the tensor product of a nonempty collection of indexed coproducts. O(n) in the size
    /// of the result.
    ///
    /// # Errors
    ///
    /// If the collection was empty or if the underying finite functions are incompatible.
    pub fn tensor_of(mut is: impl Iterator<Item = Self>) -> Result<Self, Error<A>> {
        if let Some(first) = is.next() {
            let (ss, vs): (Vec<_>, Vec<_>) = std::iter::once(first)
                .chain(is)
                .map(|i| (i.sources, i.values))
                .unzip();
            let sources = FiniteFunction::coproduct_of(ss.into_iter())?;
            let values = FiniteFunction::tensor_of(vs.into_iter());
            Ok(Self { sources, values })
        } else {
            Err(Error::EmptyTensorOf)
        }
    }

    /// Given an indexed coproduct ∑_{i ∈ X} fᵢ: ∑_{i ∈ X} Aᵢ → B and a finite function g: B → C,
    /// return a new indexed coproduct ∑_{i ∈ X} fᵢ: ∑_{i ∈ X}Aᵢ → C.
    ///
    /// # Errors
    ///
    /// If the values of this indexed coproduct don't compose with g.
    pub fn map_values(&self, g: &FiniteFunction<A>) -> Result<Self, Error<A>> {
        Ok(Self {
            sources: self.sources.clone(),
            values: (&self.values >> g)?,
        })
    }

    /// Given an indexed coproduct ∑_{i ∈ X} fᵢ: ∑_{i ∈ X} Aᵢ → B and a finite function g: W → X,
    /// return a new indexed coproduct ∑_{i ∈ X} fᵢ: ∑_{i ∈ W}A_{g(i)} → C.
    ///
    /// # Errors
    ///
    /// If the sources of this indexed coproduct don't compose with g.
    pub fn map_indexes(&self, g: &FiniteFunction<A>) -> Result<Self, Error<A>> {
        Ok(Self {
            sources: (g >> &self.sources)?,
            values: self.indexed_values(g)?,
        })
    }

    /// Like `IndexedCoproduct::map_indexes`, but only computes the values array.
    ///
    /// # Errors
    ///
    /// If the domain of g isn't the same as the source of this coproduct or if there's an error
    /// computing injections.
    pub fn indexed_values(&self, f: &FiniteFunction<A>) -> Result<FiniteFunction<A>, Error<A>> {
        if f.target == self.sources.source {
            let inj = self.sources.injections(f)?;
            Ok((&inj >> &self.values)?)
        } else {
            Err(Error::IndexedValuesMismatch {
                c: self.clone(),
                f: f.clone(),
            })
        }
    }
}

impl<A: Backend> Display for IndexedCoproduct<A> {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(
            f,
            "∑({}ℕ): → ({})",
            self.sources
                .to_string()
                .trim_end_matches(|c: char| c.is_numeric()),
            self.values
        )
    }
}

impl<A: Backend> Debug for IndexedCoproduct<A> {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        // Reuse Display cuz it's nice
        Display::fmt(self, f)
    }
}

impl<A: Backend> TryFrom<Vec<FiniteFunction<A>>> for IndexedCoproduct<A> {
    type Error = Error<A>;
    fn try_from(fs: Vec<FiniteFunction<A>>) -> Result<Self, Self::Error> {
        let mut target = usize::MAX;
        let mut table = A::zeros(fs.len());
        for (i, f) in fs.iter().enumerate() {
            if i > 0 && f.target != target {
                return Err(Error::TargetMismatch {
                    first: target,
                    second: f.target,
                });
            }
            target = f.target;
            *table.index_mut(i) = f.source;
        }
        let sources = FiniteFunction::new(table.len(), usize::MAX, table)?;
        let values = FiniteFunction::coproduct_of(fs.iter().cloned())?;
        Ok(Self { sources, values })
    }
}

impl<A: Backend> IntoIterator for IndexedCoproduct<A> {
    type Item = FiniteFunction<A>;
    type IntoIter = ICStack<A>;
    fn into_iter(self) -> Self::IntoIter {
        let mut i = 0;
        let mut finite_functions = (0..self.sources.table.len())
            .map(|j| {
                let x = self.sources.table[j];
                i += x;
                FiniteFunction {
                    source: x,
                    target: self.target(),
                    table: self.values.table.slice(i - x, i),
                }
            })
            .collect::<Vec<_>>();
        finite_functions.reverse();
        ICStack { finite_functions }
    }
}

/// An iterator over the finite functions in an indexed coproduct.
pub struct ICStack<A: Backend> {
    finite_functions: Vec<FiniteFunction<A>>,
}

impl<A: Backend> Iterator for ICStack<A> {
    type Item = FiniteFunction<A>;
    fn next(&mut self) -> Option<Self::Item> {
        self.finite_functions.pop()
    }
}

impl_arith!(IndexedCoproduct, Shr, shr, compose, true);
impl_arith!(IndexedCoproduct, Add, add, coproduct, true);
impl_arith!(IndexedCoproduct, BitOr, bitor, tensor, true);

/// Errors that can arise when building or computing with indexed coproducts
#[derive(Debug, thiserror::Error)]
pub enum Error<A: Backend> {
    /// A finite function error occurred: {0}
    #[error("A finite function error occurred: {0}")]
    FiniteFunction(#[from] FFError<A>),
    /// The sum {`sum`} of the sources table doesn't match the source of values {`values`}
    #[error("The sum {sum} of the sources table doesn't match the source of values {values}")]
    SourcesSumValuesSourceMismatch {
        /// The sum of the sources table
        sum: usize,
        /// The values finite function
        values: FiniteFunction<A>,
    },
    /// The target {`first`} doesn't match {`second`}
    #[error("The target {first} doesn't match {second}")]
    TargetMismatch {
        /// First target
        first: usize,
        /// Second target
        second: usize,
    },
    /// Tried to take the tensor of an empty collection of indexed coproducts
    #[error("Tried to take the tensor of an empty collection of indexed coproducts")]
    EmptyTensorOf,
    /// The values {`values`} source {`src_0`} doesn't match the other's source {`src_1`}
    #[error("The values {values} source {src_0} doesn't match the other's source {src_1}")]
    SourceMistmatch {
        /// The values field of the first indexed coproduct
        values: FiniteFunction<A>,
        /// The source of that values field
        src_0: usize,
        /// The source of the second indexed coproduct
        src_1: usize,
    },
    /// The target of the finite function {f} doesn't match the target of the sources of {c}
    #[error(
        "The target of the finite function {f} doesn't match the target of the sources of {c}"
    )]
    IndexedValuesMismatch {
        /// The indexed coproduct
        c: IndexedCoproduct<A>,
        /// The function with which we tried `c.indexed_values(&f)`
        f: FiniteFunction<A>,
    },
}

/// `proptest` strategies for generating arbitrary indexed coproducts.
#[cfg(test)]
pub(crate) mod strategies {
    use super::*;
    use crate::finite_function::strategies as ffs;
    use proptest::prelude::*;

    pub fn coproduct_indexes() -> impl Strategy<Value = usize> {
        0..32usize
    }

    pub fn indexed_coproducts<A: Backend>() -> impl Strategy<Value = IndexedCoproduct<A>> {
        coproduct_indexes().prop_flat_map(indexed_coproduct_from)
    }

    pub fn indexed_coproduct_from<A: Backend>(
        n: usize,
    ) -> impl Strategy<Value = IndexedCoproduct<A>> {
        ffs::arrow_from(n)
            .prop_flat_map(|mut sources: FiniteFunction<A>| {
                let source = sources.table.sum();
                sources.target = usize::MAX;
                (Just(sources), ffs::arrow_from(source))
            })
            .prop_map(|(sources, values)| IndexedCoproduct { sources, values })
    }

    pub fn indexed_coproduct<A: Backend>(
        source: usize,
        target: usize,
    ) -> impl Strategy<Value = IndexedCoproduct<A>> {
        ffs::arrow_from(source)
            .prop_flat_map(move |mut sources: FiniteFunction<A>| {
                let source = sources.table.sum();
                sources.target = usize::MAX;
                (Just(sources), ffs::arrow(source, target))
            })
            .prop_map(|(sources, values)| IndexedCoproduct { sources, values })
    }

    pub fn coproduct_list<A: Backend>() -> impl Strategy<Value = Vec<IndexedCoproduct<A>>> {
        (1..5usize).prop_flat_map(|n| proptest::collection::vec(indexed_coproducts(), n))
    }

    pub fn composable_indexed_coproducts<A: Backend>(
    ) -> impl Strategy<Value = (IndexedCoproduct<A>, IndexedCoproduct<A>)> {
        indexed_coproducts().prop_flat_map(|c| {
            let n = c.values.source;
            (Just(c), indexed_coproduct_from(n))
        })
    }

    pub fn map_with_indexed_coproduct<A: Backend>(
    ) -> impl Strategy<Value = (IndexedCoproduct<A>, FiniteFunction<A>)> {
        indexed_coproducts().prop_flat_map(|c| {
            let n = c.source();
            (Just(c), ffs::arrow_to(n))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        array::{Backend, NDArray1Usize, StdVec},
        finite_function::strategies as ffs,
    };
    use paste::paste;

    macro_rules! properties {
        ($A:ty) => {
            paste! {
                mod [<$A:snake:lower _indexed_coproduct_properties>] {
                    use super::*;
                    use proptest::prelude::*;
                    proptest! {
                        #[test]
                        fn fuzz_try_from(fs in ffs::same_target::<$A>()) {
                            let c = IndexedCoproduct::try_from(fs.clone());
                            prop_assert!(c.is_ok());
                            let c = c.unwrap();
                            prop_assert!(c.check_rep().is_ok());
                            prop_assert_eq!(c.into_iter().collect::<Vec<_>>(), fs);
                        }

                        #[test]
                        fn singleton_ok(f in ffs::arrows::<$A>()) {
                            let s = IndexedCoproduct::singleton(f.clone());
                            prop_assert!(s.check_rep().is_ok());
                            prop_assert_eq!(s.source(), 1);
                            prop_assert_eq!(s.values, f);
                        }

                        #[test]
                        fn elements_ok(f in ffs::arrows::<$A>()) {
                            let s = IndexedCoproduct::elements(f.clone());
                            if f.source > 1 {
                                prop_assert!(s.is_ok());
                                let s = s.unwrap();
                                prop_assert_eq!(s.source(), f.source);
                                prop_assert_eq!(s.values, f);
                            }
                        }

                        #[test]
                        fn coproduct_of_injections_is_id(s in ffs::arrows::<$A>()) {
                            let i = FiniteFunction::identity(s.source);
                            let n = s.table.sum();
                            let inj = s.injections(&i);
                            prop_assert!(inj.is_ok());
                            let inj = inj.unwrap();
                            prop_assert!(inj.check_rep().is_ok());
                            prop_assert_eq!(inj, FiniteFunction::identity(n));
                        }

                        #[test]
                        fn roundtrip(c in strategies::indexed_coproducts::<$A>()) {
                            prop_assert!(c.check_rep().is_ok());
                            let fs = c.clone().into_iter().collect::<Vec<_>>();
                            let d = IndexedCoproduct::try_from(fs);
                            prop_assert!(d.is_ok());
                            let d = d.unwrap();
                            prop_assert!(d.check_rep().is_ok());
                            prop_assert_eq!(c, d);
                        }

                        #[test]
                        fn tensor_of_ok(cs in strategies::coproduct_list::<$A>()) {
                            let actual = IndexedCoproduct::tensor_of(cs.clone().into_iter());
                            prop_assert!(actual.is_ok());
                            let actual = actual.unwrap();
                            prop_assert!(actual.check_rep().is_ok());
                            let mut cs = cs;
                            cs.reverse();
                            let mut expected = cs.pop().unwrap();
                            while let Some(next) = cs.pop() {
                                let step = expected.tensor(&next);
                                prop_assert!(step.is_ok());
                                expected = step.unwrap();
                                prop_assert!(expected.check_rep().is_ok());
                            }
                            prop_assert_eq!(actual, expected);
                        }

                        #[test]
                        fn coproduct_map((c, x) in strategies::map_with_indexed_coproduct::<$A>()) {
                            prop_assert!(c.check_rep().is_ok());
                            let fs = c.clone().into_iter().collect::<Vec<_>>();
                            let table = $A::array((0..x.source).map(|i| fs[x[i]].source));
                            let sources = FiniteFunction::new(table.len(), usize::MAX, table);
                            prop_assert!(sources.is_ok());
                            let sources = sources.unwrap();
                            let values = FiniteFunction::coproduct_of((0..x.source).map(|i| fs[x[i]].clone()));
                            prop_assert!(values.is_ok());
                            let values = values.unwrap();
                            prop_assert_eq!(c.source(), fs.len());
                            let d = c.map_indexes(&x);
                            prop_assert!(d.is_ok());
                            let d = d.unwrap();
                            prop_assert_eq!(d.source(), x.source);
                            prop_assert_eq!(d.sources, sources);
                            if d.values.source > 0 {
                                prop_assert_eq!(d.values, values);
                            } else {
                                prop_assert_eq!(d.values.source, values.source);
                                prop_assert_eq!(d.values.table, values.table);
                            }
                        }

                        #[test]
                        fn flatmap((x, y) in strategies::composable_indexed_coproducts::<$A>()) {
                            prop_assert!(x.check_rep().is_ok());
                            prop_assert!(y.check_rep().is_ok());
                            let actual = x.flatmap(&y);
                            prop_assert!(actual.is_ok());
                            let actual = actual.unwrap();
                            prop_assert!(actual.check_rep().is_ok());
                            prop_assert_eq!(actual.source(), x.source());
                            prop_assert_eq!(actual.values.source, y.values.source);
                        }
                    }
                }
            }
        }
    }

    properties!(StdVec);
    properties!(NDArray1Usize);
}
