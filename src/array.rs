//! Array backend
use ndarray::{Array1, Axis};
use std::{
    collections::BTreeMap,
    fmt::{Debug, Display, Formatter, Result},
    ops::{Add, Index, IndexMut, Sub},
};

/// Array backend: 1D arrays.
pub trait Backend:
    Index<usize, Output = usize>
    + IndexMut<usize, Output = usize>
    + Add<Output = Self>
    + Add<usize, Output = Self>
    + Sub<Output = Self>
    + Sub<usize, Output = Self>
    + Debug
    + Display
    + PartialEq
    + Eq
    + Clone
    + Sized
{
    /// Create an empty array.
    #[must_use]
    fn empty() -> Self;
    /// Constant array of a given size.
    #[must_use]
    fn constant(item: usize, size: usize) -> Self;
    /// All-zero array.
    #[must_use]
    fn zeros(size: usize) -> Self {
        Self::constant(0, size)
    }
    /// All-one array.
    #[must_use]
    fn ones(size: usize) -> Self {
        Self::constant(1, size)
    }
    /// Construct an array out of an iterator of elements.
    #[must_use]
    fn array(xs: impl Iterator<Item = usize>) -> Self;
    /// Iterator over the elements of this array.
    #[must_use]
    fn iter(&self) -> impl Iterator<Item = usize>;
    /// Construct the array from start to end.
    #[must_use]
    fn arange(start: usize, end: usize) -> Self {
        Self::array(start..end)
    }
    /// Is this the empty array?
    #[must_use]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Number of elements in the array.
    #[must_use]
    fn len(&self) -> usize;
    /// Sum of elements in the array.
    #[must_use]
    fn sum(&self) -> usize;
    /// Subarray xs[lo..hi]
    #[must_use]
    fn slice(&self, lo: usize, hi: usize) -> Self;
    /// Prefix sum, including leading zero.
    #[must_use]
    fn prefix_sum(&self) -> Self;
    /// Concatenate the two arrays together.
    #[must_use]
    fn append(&self, other: &Self) -> Self {
        Self::array(self.iter().chain(other.iter()))
    }
    /// Repeat this array's elements as dictated by the second.
    #[must_use]
    fn repeat(&self, repeats: &Self) -> Self {
        Self::array(
            self.iter()
                .zip(repeats.iter())
                .flat_map(|(x, r)| std::iter::repeat(x).take(r)),
        )
    }
    /// Select the entries in this array according to the index.
    #[must_use]
    fn select(&self, index: &Self) -> Self;
    /// The indices that would sort this array.
    #[must_use]
    fn argsort(&self) -> Self {
        let mut indices = (0..self.len()).collect::<Vec<_>>();
        indices.sort_by_key(|&i| self[i]);
        Self::array(indices.into_iter())
    }
    /// Given a graph of n vertices whose edges are encoded as a pair of arrays, compute the connected
    /// components and return (# of components, mapping from vertex to component)
    ///
    /// # Assumptions
    ///
    /// - |sources| = |targets|
    /// - 0 ≤ v < n ∀ v ∈ sources
    /// - 0 ≤ v < n ∀ v ∈ targets
    fn connected_components(sources: &Self, targets: &Self, n: usize) -> (usize, Self) {
        let mut adj = BTreeMap::new();
        for (s, t) in sources.iter().zip(targets.iter()) {
            adj.entry(s).or_insert_with(Vec::new).push(t);
        }
        let (mut label, mut labels) = (0, Self::zeros(n));
        while let Some((node, mut stack)) = adj.pop_first() {
            labels[node] = label;
            while let Some(v) = stack.pop() {
                labels[v] = label;
                stack.extend(&adj.remove(&v).unwrap_or_default());
            }
            label += 1;
        }
        (label, labels)
    }
    /// Given an array of sizes [x₀, x₁, …], output the concatenation arange(x₀) || arange(x₁) || …
    #[must_use]
    fn segmented_arange(&self) -> Self {
        let mut ptr = self.prefix_sum().iter().collect::<Vec<_>>();
        ptr.pop().map_or_else(Self::empty, |n| {
            Self::arange(0, n) - Self::array(ptr.into_iter()).repeat(self)
        })
    }
    /// Sum within segments dictated by other array.
    #[must_use]
    fn segmented_sum(&self, segments: &Self) -> Self {
        let ptr = self.prefix_sum();
        let n = ptr.len();
        let sums = segments.prefix_sum();
        Self::select(&sums, &Self::slice(&ptr, 1, n))
            - Self::select(&sums, &Self::slice(&ptr, 0, n - 1))
    }
}

#[cfg(test)]
pub(crate) mod strategies {
    use super::*;
    use proptest::prelude::*;
    pub fn array<A: Backend>(min: usize, max: usize, len: usize) -> impl Strategy<Value = A> {
        proptest::collection::vec(min..max, len).prop_map(|xs| A::array(xs.into_iter()))
    }
    pub fn permutation<A: Backend>(n: usize) -> impl Strategy<Value = A> {
        Just((0..n).collect::<Vec<usize>>())
            .prop_shuffle()
            .prop_map(|xs| A::array(xs.into_iter()))
    }
}

/// The usual `Vec`
#[derive(PartialEq, Eq, Clone)]
pub struct StdVec(Vec<usize>);

impl Display for StdVec {
    fn fmt(&self, f: &mut Formatter) -> Result {
        f.write_str("[")?;
        for (i, x) in self.0.iter().enumerate() {
            let sep = if i > 0 { ", " } else { "" };
            write!(f, "{sep}{x}")?;
        }
        f.write_str("]")
    }
}

impl Debug for StdVec {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        // Reuse Display cuz it's nice
        Display::fmt(self, f)
    }
}

impl Index<usize> for StdVec {
    type Output = usize;
    fn index(&self, i: usize) -> &Self::Output {
        self.0.index(i)
    }
}

impl IndexMut<usize> for StdVec {
    fn index_mut(&mut self, i: usize) -> &mut usize {
        self.0.index_mut(i)
    }
}

impl Add for StdVec {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self(
            self.0
                .iter()
                .zip(other.0.iter())
                .map(|(&i, &j)| i + j)
                .collect(),
        )
    }
}

impl Add<usize> for StdVec {
    type Output = Self;
    fn add(self, scalar: usize) -> Self {
        Self(self.0.iter().map(|&x| x + scalar).collect())
    }
}

impl Sub for StdVec {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self(
            self.0
                .iter()
                .zip(other.0.iter())
                .map(|(&i, &j)| i - j)
                .collect(),
        )
    }
}

impl Sub<usize> for StdVec {
    type Output = Self;
    fn sub(self, scalar: usize) -> Self {
        Self(self.0.iter().map(|&x| x - scalar).collect())
    }
}

impl Backend for StdVec {
    /// Create an empty array.
    fn empty() -> Self {
        Self(Vec::new())
    }
    /// Constant array of a given size.
    fn constant(item: usize, size: usize) -> Self {
        Self(vec![item; size])
    }
    /// Construct an array out of an iterator of elements.
    fn array(xs: impl Iterator<Item = usize>) -> Self {
        Self(xs.collect())
    }
    /// Iterator over the elements of this array.
    fn iter(&self) -> impl Iterator<Item = usize> {
        self.0.iter().copied()
    }
    /// Number of elements in the array.
    fn len(&self) -> usize {
        self.0.len()
    }
    /// Sum of elements in the array.
    fn sum(&self) -> usize {
        self.0.iter().sum()
    }
    /// Subarray xs[lo..hi]
    fn slice(&self, lo: usize, hi: usize) -> Self {
        Self(self.0[lo..hi].to_vec())
    }
    /// Prefix sum, including leading zero.
    fn prefix_sum(&self) -> Self {
        let mut result = vec![0; self.len() + 1];
        for (i, x) in (1..).zip(self.0.iter()) {
            result[i] += x + result[i - 1];
        }
        Self(result)
    }
    /// Select the entries in this array according to the index.
    fn select(&self, index: &Self) -> Self {
        Self::array(index.iter().map(|i| self[i]))
    }
}

/// `ndarray::Array1<usize>`
pub type NDArray1Usize = Array1<usize>;

impl Backend for NDArray1Usize {
    fn empty() -> Self {
        Self::from_vec(Vec::new())
    }
    fn constant(item: usize, size: usize) -> Self {
        Self::from_elem(size, item)
    }
    fn array(xs: impl Iterator<Item = usize>) -> Self {
        Self::from_iter(xs)
    }
    fn iter(&self) -> impl Iterator<Item = usize> {
        self.iter().copied()
    }
    fn len(&self) -> usize {
        Self::len(self)
    }
    fn sum(&self) -> usize {
        Self::sum(self)
    }
    fn slice(&self, lo: usize, hi: usize) -> Self {
        // ndarray's usual slice is unsafe, not that we do any bounds checking here either
        Self::from_iter((lo..hi).map(|i| self[i]))
    }
    fn prefix_sum(&self) -> Self {
        let mut out = <Self as Backend>::zeros(1).append(self);
        out.accumulate_axis_inplace(Axis(0), |&prev, curr| *curr += prev);
        out
    }
    fn append(&self, other: &Self) -> Self {
        let mut out = self.clone();
        Self::append(&mut out, Axis(0), other.view()).unwrap();
        out
    }
    fn select(&self, index: &Self) -> Self {
        index.mapv(|i| self[i])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cc_ok() {
        //   0 → 1  3 → 4
        //   ↓      ↑   ↓
        //   2      6 ← 5
        let sources = StdVec(vec![0, 0, 3, 4, 5, 6]);
        let targets = StdVec(vec![1, 2, 4, 5, 6, 3]);
        let n = 7;
        let (num_components, labels) = StdVec::connected_components(&sources, &targets, n);
        assert_eq!(num_components, 2);
        assert_eq!(labels.0, [0, 0, 0, 1, 1, 1, 1]);
    }

    #[test]
    fn cumsum_zero_ok() {
        assert_eq!(StdVec::ones(3).prefix_sum().0, [0, 1, 2, 3]);
    }

    #[test]
    fn segmented_arange_ok() {
        assert_eq!(
            StdVec::array([5, 2, 3, 1].into_iter()).segmented_arange().0,
            [0, 1, 2, 3, 4, 0, 1, 0, 1, 2, 0]
        );
    }

    #[test]
    fn repeat_ok() {
        assert_eq!(
            StdVec::arange(1, 4).repeat(&StdVec::arange(1, 4)).0,
            [1, 2, 2, 3, 3, 3]
        );
    }

    #[test]
    fn segmented_sum_ok() {
        assert_eq!(
            StdVec::array([5, 4, 2, 1, 3].into_iter())
                .segmented_sum(&StdVec::arange(0, 17))
                .0,
            [10, 26, 19, 11, 39]
        );
        assert_eq!(StdVec::empty().segmented_sum(&StdVec::arange(0, 17)).0, []);
    }
}
