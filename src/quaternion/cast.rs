extern crate num_traits;

use super::Quaternion;
use num_traits::{AsPrimitive, FromPrimitive, NumCast, ToPrimitive, Signed};
use Numeral;

macro_rules! impl_to_primitive {
    ($ty:ty, $to:ident) => {
        #[inline]
        fn $to(&self) -> Option<$ty> {
            if self.b.is_zero() &&
               self.c.is_zero() &&
               self.d.is_zero() { 
                   self.a.$to() 
            } else { 
                   None 
            }
        }
    }
} // impl_to_primitive

// Returns None if Complex part is non-zero
impl<T> ToPrimitive for Quaternion<T> where T: Numeral + ToPrimitive + Signed {
    impl_to_primitive!(usize, to_usize);
    impl_to_primitive!(isize, to_isize);
    impl_to_primitive!(u8, to_u8);
    impl_to_primitive!(u16, to_u16);
    impl_to_primitive!(u32, to_u32);
    impl_to_primitive!(u64, to_u64);
    impl_to_primitive!(i8, to_i8);
    impl_to_primitive!(i16, to_i16);
    impl_to_primitive!(i32, to_i32);
    impl_to_primitive!(i64, to_i64);
    #[cfg(has_i128)]
    impl_to_primitive!(u128, to_u128);
    #[cfg(has_i128)]
    impl_to_primitive!(i128, to_i128);
    impl_to_primitive!(f32, to_f32);
    impl_to_primitive!(f64, to_f64);
}

macro_rules! impl_from_primitive {
    ($ty:ty, $from_xx:ident) => {
        #[inline]
        fn $from_xx(n: $ty) -> Option<Self> {
            T::$from_xx(n).map(|re| Quaternion {
                a: re,
                b: T::zero(),
                c: T::zero(),
                d: T::zero(),
            })
        }
    };
} // impl_from_primitive

impl<T> FromPrimitive for Quaternion<T> where T: Numeral + FromPrimitive + Signed {
    impl_from_primitive!(usize, from_usize);
    impl_from_primitive!(isize, from_isize);
    impl_from_primitive!(u8, from_u8);
    impl_from_primitive!(u16, from_u16);
    impl_from_primitive!(u32, from_u32);
    impl_from_primitive!(u64, from_u64);
    impl_from_primitive!(i8, from_i8);
    impl_from_primitive!(i16, from_i16);
    impl_from_primitive!(i32, from_i32);
    impl_from_primitive!(i64, from_i64);
    #[cfg(has_i128)]
    impl_from_primitive!(u128, from_u128);
    #[cfg(has_i128)]
    impl_from_primitive!(i128, from_i128);
    impl_from_primitive!(f32, from_f32);
    impl_from_primitive!(f64, from_f64);
}

impl<T> NumCast for Quaternion<T> where T: Numeral + NumCast + Signed {
    fn from<U: ToPrimitive>(n: U) -> Option<Self> {
        T::from(n).map(|re| Quaternion {
            a:re,
            b: T::zero(),
            c: T::zero(),
            d: T::zero(),
        })
    }
}

impl<T, U> AsPrimitive<U> for Quaternion<T>
where
    T: AsPrimitive<U> + Numeral + Signed,
    U: 'static + Copy,
{
    fn as_(self) -> U {
        self.a.as_()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_to_primitive() {
        let a: Quaternion<i32> = Quaternion { a: 3, b: 0, c: 0, d: 0 };
        assert_eq!(a.to_i32(), Some(3_i32));
        let b: Quaternion<i32> = Quaternion { a: 3, b: 1, c: 0, d: 0 };
        assert_eq!(b.to_i32(), None);
        let x: Quaternion<f32> = Quaternion { a: 1.0, b: 0.1, c: 0.0, d: 0.0 };
        assert_eq!(x.to_f32(), None);
        let y: Quaternion<f32> = Quaternion { a: 1.0, b: 0.0, c: 0.0, d: 0.0 };
        assert_eq!(y.to_f32(), Some(1.0));
        let z: Quaternion<f32> = Quaternion { a: 1.0, b: 0.0, c: 0.0, d: 0.0 };
        assert_eq!(z.to_i32(), Some(1));
    }

    #[test]
    fn test_from_primitive() {
        let a: Quaternion<f32> = FromPrimitive::from_i32(2).unwrap();
        assert_eq!(a, Quaternion { a: 2.0, b: 0.0, c: 0.0, d: 0.0 });
    }

    #[test]
    fn test_num_cast() {
        let a: Quaternion<f32> = NumCast::from(2_i32).unwrap();
        assert_eq!(a, Quaternion { a: 2.0, b: 0.0, c: 0.0, d: 0.0 });
    }

    #[test]
    fn test_as_primitive() {
        let a: Quaternion<f32> = Quaternion { a: 2.0, b: 0.2, c: 0.0, d: 0.0 };
        let a_: i32 = a.as_();
        assert_eq!(a_, 2_i32);
    }
}
