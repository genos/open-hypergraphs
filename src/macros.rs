macro_rules! impl_arith(
    ($type:ident, $operation:ident, $method:ident, $impl:ident, true) => {
        impl<A: Backend> $operation for $type<A> {
            type Output = Result<$type<A>, Error<A>>;
            fn $method(self, other: $type<A>) -> Self::Output {
                self.$impl(&other)
            }
        }
        impl<A: Backend> $operation<&$type<A>> for $type<A> {
            type Output = Result<$type<A>, Error<A>>;
            fn $method(self, other: &$type<A>) -> Self::Output {
                self.$impl(other)
            }
        }
        impl<A: Backend> $operation<$type<A>> for &$type<A> {
            type Output = Result<$type<A>, Error<A>>;
            fn $method(self, other: $type<A>) -> Self::Output {
                self.$impl(&other)
            }
        }
        impl<A: Backend> $operation<&$type<A>> for &$type<A> {
            type Output = Result<$type<A>, Error<A>>;
            fn $method(self, other: &$type<A>) -> Self::Output {
                self.$impl(other)
            }
        }
    };
    ($type:ident, $operation:ident, $method:ident, $impl:ident, false) => {
        impl<A: Backend> $operation for $type<A> {
            type Output = $type<A>;
            fn $method(self, other: $type<A>) -> Self::Output {
                self.$impl(&other)
            }
        }
        impl<A: Backend> $operation<&$type<A>> for $type<A> {
            type Output = $type<A>;
            fn $method(self, other: &$type<A>) -> Self::Output {
                self.$impl(other)
            }
        }
        impl<A: Backend> $operation<$type<A>> for &$type<A> {
            type Output = $type<A>;
            fn $method(self, other: $type<A>) -> Self::Output {
                self.$impl(&other)
            }
        }
        impl<A: Backend> $operation<&$type<A>> for &$type<A> {
            type Output = $type<A>;
            fn $method(self, other: &$type<A>) -> Self::Output {
                self.$impl(other)
            }
        }
    };
);

pub(crate) use impl_arith;
