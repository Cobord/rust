//! Arithmetic on distributions if the sampled type
//! `T` supports that particular operation

use core::ops::{Add, Deref, DerefMut, Mul, Sub, Neg};

use crate::marker::PhantomData;
pub use core::random::*;

/// Arithmetic on distributions if the sampled type
/// `T` supports that particular operation
pub trait DistributionArithExt<T> : Distribution<T> + Sized { 
    /// Turn something that `impl Distribution<T>` into
    /// a struct wrapping it which also impls `Distribution<T>`
    /// that can implement the foreign traits
    /// `Add`, `Sub`, `Mul` as appropriate for `T`
    fn arithmetize(self) -> ArithmeticDistribution<T,Self> {
        self.into()
    }
}

impl<T,W : Distribution<T>> DistributionArithExt<T> for W {
}

/// A struct wrapping a `w` which impls `Distribution<T>`
/// that can implement `Add`, `Sub`, `Mul` as appropriate
/// for `T`
#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct ArithmeticDistribution<T,W: Distribution<T>> {
    w: W,
    t: PhantomData<T>,
}

impl<T,W: Distribution<T>> From<W> for ArithmeticDistribution<T,W> {
    fn from(w: W) -> Self {
        Self {
            w,
            t: PhantomData
        }
    }
}

impl<T,W> Deref for ArithmeticDistribution<T,W>
where
    W: Distribution<T>
{
    type Target = W;
    fn deref(&self) -> &Self::Target {
        &self.w
    }
}

impl<T,W> DerefMut for ArithmeticDistribution<T,W>
where
    W: Distribution<T>
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.w
    }
}

impl<T,W: Distribution<T>> Distribution<T> for ArithmeticDistribution<T,W> {
    fn sample(&self, source: &mut (impl RandomSource + ?Sized)) -> T {
        self.w.sample(source)
    }
}

/// The distribution which is obtained by
/// taking the sum of samples taken from two
/// other distributions on a type where
/// addition is defined.
#[derive(Debug, Clone)]
pub struct AddDistribution<T,W1,W2>
where
    T: Add<T,Output=T>,
    W1: Distribution<T>,
    W2: Distribution<T>,
{
    operand_1: W1,
    operand_2: W2,
    _output_type: PhantomData<T>,
}

impl<T,W1,W2> Distribution<T> for AddDistribution<T,W1,W2>
where
    T: Add<T,Output=T>,
    W1: Distribution<T>,
    W2: Distribution<T>,
{
    fn sample(&self, source: &mut (impl RandomSource + ?Sized)) -> T {
        let operand_1_sampled = self.operand_1.sample(source);
        let operand_2_sampled = self.operand_2.sample(source);
        operand_1_sampled + operand_2_sampled
    }
}

impl<T,W1,W2> Add<W2> for ArithmeticDistribution<T,W1>
where
    T: Add<T,Output=T>,
    W1: Distribution<T>,
    W2: Distribution<T>
{
    type Output = ArithmeticDistribution<T,AddDistribution<T,W1,W2>>;

    fn add(self, rhs: W2) -> Self::Output {
        (AddDistribution {
            operand_1: self.w,
            operand_2: rhs,
            _output_type: PhantomData
        }).into()
    }
}

/// The distribution which is obtained by
/// taking the difference of samples taken from two
/// other distributions on a type where
/// subtraction is defined.
#[derive(Debug, Clone)]
pub struct SubDistribution<T,W1,W2>
where
    T: Sub<T,Output=T>,
    W1: Distribution<T>,
    W2: Distribution<T>,
{
    operand_1: W1,
    operand_2: W2,
    _output_type: PhantomData<T>,
}


impl<T,W1,W2> Distribution<T> for SubDistribution<T,W1,W2>
where
    T: Sub<T,Output=T>,
    W1: Distribution<T>,
    W2: Distribution<T>,
{
    fn sample(&self, source: &mut (impl RandomSource + ?Sized)) -> T {
        let operand_1_sampled = self.operand_1.sample(source);
        let operand_2_sampled = self.operand_2.sample(source);
        operand_1_sampled - operand_2_sampled
    }
}

impl<T,W1,W2> Sub<W2> for ArithmeticDistribution<T,W1>
where
    T: Sub<T,Output=T>,
    W1: Distribution<T>,
    W2: Distribution<T>
{
    type Output = ArithmeticDistribution<T,SubDistribution<T,W1,W2>>;

    fn sub(self, rhs: W2) -> Self::Output {
        (SubDistribution {
            operand_1: self.w,
            operand_2: rhs,
            _output_type: PhantomData
        }).into()
    }
}

/// The distribution which is obtained by
/// taking the negation of a sample taken
/// from a distributions on a type where
/// negation is defined.
#[derive(Debug, Clone)]
pub struct NegDistribution<T,W>
where
    T: Neg<Output=T>,
    W: Distribution<T>,
{
    operand_1: W,
    _output_type: PhantomData<T>,
}


impl<T,W> Distribution<T> for NegDistribution<T,W>
where
    T: Neg<Output=T>,
    W: Distribution<T>,
{
    fn sample(&self, source: &mut (impl RandomSource + ?Sized)) -> T {
        let operand_1_sampled = self.operand_1.sample(source);
        - operand_1_sampled
    }
}

impl<T,W> Neg for ArithmeticDistribution<T,W>
where
    T: Neg<Output=T>,
    W: Distribution<T>,
{
    type Output = ArithmeticDistribution<T,NegDistribution<T,W>>;

    fn neg(self) -> Self::Output {
        (NegDistribution {
            operand_1: self.w,
            _output_type: PhantomData
        }).into()
    }
}

/// The distribution which is obtained by
/// taking the product of samples taken from two
/// other distributions on a type where
/// multiplication is defined.

#[derive(Debug, Clone)]
pub struct MulDistribution<T,W1,W2>
where
    T: Mul<T,Output=T>,
    W1: Distribution<T>,
    W2: Distribution<T>,
{
    operand_1: W1,
    operand_2: W2,
    _output_type: PhantomData<T>,
}

impl<T,W1,W2> Distribution<T> for MulDistribution<T,W1,W2>
where
    T: Mul<T,Output=T>,
    W1: Distribution<T>,
    W2: Distribution<T>,
{
    fn sample(&self, source: &mut (impl RandomSource + ?Sized)) -> T {
        let operand_1_sampled = self.operand_1.sample(source);
        let operand_2_sampled = self.operand_2.sample(source);
        operand_1_sampled * operand_2_sampled
    }
}

impl<T,W1,W2> Mul<W2> for ArithmeticDistribution<T,W1>
where
    T: Mul<T,Output=T>,
    W1: Distribution<T>,
    W2: Distribution<T>
{
    type Output = ArithmeticDistribution<T,MulDistribution<T,W1,W2>>;

    fn mul(self, rhs: W2) -> Self::Output {
        (MulDistribution {
            operand_1: self.w,
            operand_2: rhs,
            _output_type: PhantomData
        }).into()
    }
}