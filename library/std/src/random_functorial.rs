//! Monadic operations on random value generation.

use core::marker::PhantomData;
pub use core::random::*;

/// Monadic operations and strength (pairing into tuples) of random value generation
pub trait DistributionExt<T> : Distribution<T> + Sized {
    /// Construct the `StrengthDistribution`
    /// by providing the distributions on
    /// both factors
    fn pair<T2,W2: Distribution<T2>>(self, w2: W2) -> StrengthDistribution<T,T2,Self,W2> {
        StrengthDistribution {
            w1: self,
            w2,
            _output_type: PhantomData
        }
    }

    /// Construct the `PushforwardDistribution`
    /// by providing the distribution on the source
    /// and the function from source to target
    fn fmap<T2>(self, pushforward_function: fn(T) -> T2) -> PushforwardDistribution<T,T2,Self> {
        PushforwardDistribution {
            before_f: self,
            pushforward_function
        }
    }

    /// Construct the `PushforwardDistribution`
    /// by providing the distribution on the source
    /// and the function from source to target
    fn pushforward<T2>(self, pushforward_function: fn(T) -> T2) -> PushforwardDistribution<T,T2,Self> {
        self.fmap(pushforward_function)
    }

    /// Construct the `PushforwardDistribution`
    /// by providing the distribution on the source
    /// and use `into` for conversion
    fn fmap_into<T2: From<T>>(self) -> PushforwardDistribution<T, T2, Self> {
        self.fmap(T2::from)
    }

    /// Construct the `PushforwardDistribution`
    /// by providing the distribution on the source
    /// and use `into` for conversion
    fn pushforward_into<T2: From<T>>(self) -> PushforwardDistribution<T, T2, Self> {
        self.fmap_into()
    }

    /// Construct the `BindDistribution`
    /// by providing the distribution on the source
    /// and the function from source to distributions on the target
    fn bind<T2,W2: Distribution<T2>>(self, bind_f: fn(T) -> W2) -> BindDistrubition<T,T2,Self,W2> {
        BindDistrubition {
            before_f: self,
            bind_f,
            _output_type: PhantomData
        }
    }

    /// Given a `Distribution (Distribution T1)`
    /// we can flatten that in order to just
    /// give a `Distribution T1`
    fn return_map<T1>(self) -> BindDistrubition<T, T1, Self, T>
    where
        T: Distribution<T1>
    {
        self.bind(|x| x)
    }

    /// Draw independent samples from the same `Distribution<T>` until the
    /// result passes the `filter` with a `Some` variant
    /// where the `filter` is just a restriction of the identity to a partial
    /// function.
    /// It is expected that the likelihood of getting through the filter
    /// when sampling from `self` is high. We are rejecting and retrying
    /// and so we don't want it to get stuck in that loop too long.
    /// Use only in that context.
    fn filter(self, filter: fn(T) -> Option<T>, give_up_iterations: Option<usize>) -> FilterMapDistribution<T,T,Self>
    where 
        T: Clone
    {
        FilterMapDistribution {
            sample_one: self,
            filter,
            give_up_iterations,
            _output_type: PhantomData
        }
    }

    /// Draw independent samples from the same `Distribution<T>` until the
    /// result passes the `filter` with a `Some` variant
    fn filter_map<T2>(self, filter: fn(T) -> Option<T2>, give_up_iterations: Option<usize>) -> 
        FilterMapDistribution<T,T2,Self>
    {
        FilterMapDistribution {
            sample_one: self,
            filter,
            give_up_iterations,
            _output_type: PhantomData
        }
    }

    /// Either take a single sample and look at that result.
    /// Or use the provided `T`.
    /// Return a delta distribution which ignores the
    /// randomness and always returns that value upon sampling.
    fn freeze(self, source: Result<&mut (impl RandomSource + ?Sized), T>) -> ZeroEntropyDistribution<T>
    where 
        T: Clone
    {
        let fixed_val = match source {
            Ok(randomness_source) => {
                self.sample(randomness_source)
            },
            Err(fixed_val) => {
                fixed_val
            }
        };
        delta_dist(fixed_val)
    }

    /// Construct the `IndependentSamples`
    /// by providing the distribution on
    /// one sample
    fn repeat_n<const N: usize>(self) -> IndependentSamples<N,T,Self> {
        IndependentSamples {
            sample_one: self,
            _output_type: PhantomData
        }
    }

}

impl<T,W : Distribution<T>> DistributionExt<T> for W {

}

/// Promote a single value of T to a distribution over T.
/// When sampling, the source of randomness is disregarded.
#[derive(Debug, Clone)]
pub struct ZeroEntropyDistribution<T: Clone>(T);

impl<T: Clone> Distribution<T> for ZeroEntropyDistribution<T> {
    fn sample(&self, _source: &mut (impl RandomSource + ?Sized)) -> T {
        self.0.clone()
    }
}

impl<T: Clone> From<T> for ZeroEntropyDistribution<T> {
    fn from(t: T) -> Self {
        Self(t)
    }
}

impl<T: Default + Clone> Default for ZeroEntropyDistribution<T> {
    fn default() -> Self {
        T::default().into()
    }
}

/// An alternative way of creating this distribution with
/// a shorter name than the full type of the struct
pub fn delta_dist<T: Clone>(t: T) -> ZeroEntropyDistribution<T> {
    ZeroEntropyDistribution::from(t)
}

/// Turn two distributions that sample two types into a single distribution
/// that samples from the tuple type
#[derive(Debug, Clone)]
pub struct StrengthDistribution<T1,T2,W1,W2>
where
    W1: Distribution<T1>,
    W2: Distribution<T2>,
{
    w1: W1,
    w2: W2,
    _output_type: PhantomData<(T1,T2)>,
}

impl<T1,T2,W1,W2> Distribution<(T1,T2)> for StrengthDistribution<T1,T2,W1,W2>
where
    W1: Distribution<T1>,
    W2: Distribution<T2>,
{
    fn sample(&self, source: &mut (impl RandomSource + ?Sized)) -> (T1,T2) {
        let w1_sampled = self.w1.sample(source);
        let w2_sampled = self.w2.sample(source);
        (w1_sampled , w2_sampled)
    }
}

/// For a function `pushforward_function` and
/// a distribution `before_f`
/// give `pushforward_function^* before_f`
#[derive(Debug, Clone)]
pub struct PushforwardDistribution<T1,T2,W1>
where
    W1: Distribution<T1>,
{
    before_f: W1,
    pushforward_function: fn(T1) -> T2
}

impl<T1,T2,W1> Distribution<T2> for PushforwardDistribution<T1,T2,W1>
where
    W1: Distribution<T1>,
{
    fn sample(&self, source: &mut (impl RandomSource + ?Sized)) -> T2 {
        let before_f_sampled = self.before_f.sample(source);
        (self.pushforward_function)(before_f_sampled)
    }
}

/// For a `morphism`, `T1 -> Tout`
/// turn it into a function that can take
/// distributions on `T1` to distributions on `Tout`.
/// The kinds of distribution on `T1` are
/// of type `W1` because `Distribution T1` is
/// not actually a type since that would make
/// `Distribution` of kind `* -> *` which is not present.
pub fn fmap_morphism<'a,T1: 'a,Tout: 'a,W1>(morphism: fn(T1)->Tout) -> impl Fn(W1)
    -> PushforwardDistribution<T1,Tout,W1> + 'a
where 
    W1: Distribution<T1>,
{
    move |before_f| before_f.fmap(morphism)
}


/// For a `morphism`, `T1,T2 -> Tout`
/// turn it into a function that can take
/// distributions on `T1` and `T2` to distributions on `Tout`.
pub fn fmap_morphism_2<'a,T1,T2,Tout,W1,W2>(morphism_tuple: fn((T1,T2))->Tout)
    -> impl Fn(W1,W2) -> PushforwardDistribution<(T1,T2),Tout,StrengthDistribution<T1,T2,W1,W2>> + 'a
where
    T1: 'a,
    T2: 'a,
    Tout: 'a,
    W1: Distribution<T1>,
    W2: Distribution<T2>,    
{
    let post_compose_with = fmap_morphism(morphism_tuple);
    move |w1,w2| post_compose_with(w1.pair(w2))
}

/// For a function `bind_f` and
/// a distribution `before_f`
/// give the result of sampling
/// from `bind_f (x)` where `x` is
/// sampled from `before_f`
#[derive(Debug, Clone)]
pub struct BindDistrubition<T1,T2,W1,W2>
where
    W1: Distribution<T1>,
    W2: Distribution<T2>,
{
    before_f: W1,
    bind_f: fn(T1) -> W2,
    _output_type: PhantomData<T2>,
}

impl<T1,T2,W1,W2> Distribution<T2> for BindDistrubition<T1,T2,W1,W2>
where
    W1: Distribution<T1>,
    W2: Distribution<T2>,
{
    fn sample(&self, source: &mut (impl RandomSource + ?Sized)) -> T2 {
        let before_f_sampled = self.before_f.sample(source);
        let w = (self.bind_f)(before_f_sampled);
        w.sample(source)
    }
}

/// Draw `N` independent samples from the same `W: Distribution<T>`
#[derive(Debug, Clone)]
pub struct IndependentSamples<const N: usize, T,W>
where
    W: Distribution<T>,
{
    sample_one: W,
    _output_type: PhantomData<T>,
}

impl<const N: usize, T,W> Distribution<[T;N]> for IndependentSamples<N,T,W>
where
    W: Distribution<T>,
{
    fn sample(&self, source: &mut (impl RandomSource + ?Sized)) -> [T;N] {
        core::array::from_fn(|_| {
            self.sample_one.sample(source)
        })
    }
}

/// Draw independent samples from the same `Distribution<T>` until the
/// result passes the `filter` with a `Some` variant
#[derive(Debug, Clone)]
pub struct FilterMapDistribution<T,T2,W>
where
    W: Distribution<T>,
{
    sample_one: W,
    filter: fn(T) -> Option<T2>,
    give_up_iterations: Option<usize>,
    _output_type: PhantomData<T2>,
}

impl<T,T2,W> Distribution<Option<T2>> for FilterMapDistribution<T,T2,W>
where
    W: Distribution<T>,
{
    fn sample(&self, source: &mut (impl RandomSource + ?Sized)) -> Option<T2> {
        if let Some(max_iterations) = self.give_up_iterations {
            for _ in 0..max_iterations {
                let cur_try = self.sample_one.sample(source);
                if let Some(final_result) = (self.filter)(cur_try) {
                    return Some(final_result);
                }
            }
            return None;
        } else {
            loop {
                let cur_try = self.sample_one.sample(source);
                if let Some(final_result) = (self.filter)(cur_try) {
                    return Some(final_result);
                }
            }
        }
    }
}

/// Choose from 2 possible distributions with probability determined by
/// a `Distribution<bool>`
#[derive(Debug, Clone)]
pub struct IfDistribution<T,B,Wtrue,Wfalse>
where
    B: Distribution<bool>,
    Wtrue: Distribution<T>,
    Wfalse: Distribution<T>,
{
    conditional: B,
    true_branch: Wtrue,
    false_branch: Wfalse,
    _output_type: PhantomData<T>,
}

impl<T,B,Wtrue,Wfalse> IfDistribution<T,B,Wtrue,Wfalse>
where
    B: Distribution<bool>,
    Wtrue: Distribution<T>,
    Wfalse: Distribution<T>,
{
    /// Construct the `IfDistribution`
    /// by providing the distribution on
    /// the conditional and the two branches
    pub fn if_expr(conditional: B, true_branch: Wtrue, false_branch: Wfalse) -> Self {
        Self {
            conditional,
            true_branch,
            false_branch,
            _output_type: PhantomData
        }
    }
}

impl<T,B,Wtrue,Wfalse> Distribution<T> for IfDistribution<T,B,Wtrue,Wfalse>
where
    B: Distribution<bool>,
    Wtrue: Distribution<T>,
    Wfalse: Distribution<T>,
{
    fn sample(&self, source: &mut (impl RandomSource + ?Sized)) -> T {
        let use_true = self.conditional.sample(source);
        if use_true {
            self.true_branch.sample(source)
        } else {
            self.false_branch.sample(source)
        }
    }
}