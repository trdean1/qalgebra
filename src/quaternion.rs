extern crate test;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std;
use std::fmt;
use Numeral;
use num_traits::sign::Signed;
use num_traits::{Zero,One};

use AlmostEq;

#[derive(Copy, Clone)]
pub struct Quaternion<T: Numeral>{
    a: T,
    b: T,
    c: T,
    d: T,
}

impl<T> Quaternion<T> where T: Numeral {
    pub fn from_vals( a: T, b: T, c: T, d: T ) 
        -> Quaternion<T> {
        Quaternion{ 
           a, b, c, d 
        }
    }

    pub fn from_vec( x: &Vec<T> )
        -> Quaternion<T> {
        Quaternion {
            a: x[0],
            b: x[1],
            c: x[2],
            d: x[3]
        }
    }

    pub fn to_vec ( self ) -> Vec<T> {
        vec![self.a, self.b, self.c, self.d]
    }
}

impl<T> Zero for Quaternion<T> where T: Numeral {
    fn zero() -> Quaternion<T> {
        Quaternion {
            a: T::zero(),
            b: T::zero(),
            c: T::zero(),
            d: T::zero(),
        }
    }

    fn is_zero(&self) -> bool {
        self.a.is_zero() && self.b.is_zero() &&
        self.c.is_zero() && self.d.is_zero()
    }
}

impl<T> One for Quaternion<T> where T: Numeral {
    fn one() -> Quaternion<T> {
        Quaternion {
            a: T::one(),
            b: T::zero(),
            c: T::zero(),
            d: T::zero(),
        }
    }

    fn is_one(&self) -> bool {
        self.a.is_one() && self.b.is_zero() &&
        self.c.is_zero() && self.d.is_zero()
    }
}

/////////////////////////////////////////////////////////////////
/// Scalar operations
/////////////////////////////////////////////////////////////////

trait ScalarOps<T> {
    fn scalar_add( &mut self, T );

    fn scalar_mul( &mut self, T );
}

impl<T> ScalarOps<T> for Quaternion<T> where T: Numeral {
    fn scalar_add( &mut self, rhs: T ) {
        self.a += rhs;  self.b += rhs;
        self.c += rhs;  self.d += rhs;
    }

    default fn scalar_mul( &mut self, rhs: T ) {
        self.a *= rhs;  self.b *= rhs;
        self.c *= rhs;  self.d *= rhs;
    }
}

#[cfg(target_arch="x86_64")]
impl ScalarOps<f32> for Quaternion<f32> {
    default fn scalar_mul( &mut self, rhs: f32 ) {
       let unpacked: (f32, f32, f32, f32);
       unsafe {
            let selfm128 = _mm_set_ps( self.d, self.c, self.b, self.a );
            let rhsm128 = _mm_set_ps( rhs, rhs, rhs, rhs );
            let res = _mm_mul_ps( selfm128, rhsm128 );
            unpacked = std::mem::transmute( res );
       }
       self.a = unpacked.3; self.b = unpacked.2;
       self.c = unpacked.1; self.a = unpacked.0;
    }
}

/////////////////////////////////////////////////////////////////
/// Conjugation and Negation
/////////////////////////////////////////////////////////////////

//XXX: Should this return a quaternion rather than act mutably?
trait Conjugate<T> {
    fn conjugate( &mut self );
}

impl<T> Conjugate<T> for Quaternion<T> where T: Numeral + Signed {
    fn conjugate( &mut self ) {
        self.b = -self.b;
        self.c = -self.c;
        self.d = -self.d;
    }
}

impl<T> std::ops::Neg for Quaternion<T> where T: Numeral + 
                                                 std::ops::Neg + 
                                                 std::ops::Neg<Output=T> {
    type Output = Quaternion<T>;

    #[inline(always)]
    fn neg(self) -> Quaternion<T> {
        Quaternion {
            a: -self.a,
            b: -self.b,
            c: -self.c,
            d: -self.d,
        }
    }
}

/////////////////////////////////////////////////////////////////
/// Arithmetic 
/////////////////////////////////////////////////////////////////

impl<T> std::ops::Add for Quaternion<T> where T: Numeral {
    type Output = Quaternion<T>;

    fn add(self, rhs: Quaternion<T>) -> Quaternion<T> {
        Quaternion {
            a: self.a + rhs.a,
            b: self.b + rhs.b,
            c: self.c + rhs.c,
            d: self.d + rhs.d,
        }
    }
}

impl<T> std::ops::AddAssign for Quaternion<T> where T: Numeral {
    fn add_assign(&mut self, rhs: Quaternion<T>) {
        *self = *self + rhs;
    }
}

impl<T> std::ops::Sub for Quaternion<T> where T: Numeral {
    type Output = Quaternion<T>;

    fn sub(self, rhs: Quaternion<T>) -> Quaternion<T> {
        Quaternion {
            a: self.a - rhs.a,
            b: self.b - rhs.b,
            c: self.c - rhs.c,
            d: self.d - rhs.d,
        }
    }
}

impl<T> std::ops::SubAssign for Quaternion<T> where T: Numeral {
    fn sub_assign(&mut self, rhs: Quaternion<T>) {
        *self = *self - rhs;
    }
}

impl<T> std::ops::Mul for Quaternion<T> where T: Numeral {
    type Output = Quaternion<T>;

    default fn mul( self, rhs: Quaternion<T> ) -> Quaternion<T> {
    	let a = self.a * rhs.a - self.b * rhs.b - self.c * rhs.c - self.d * rhs.d;
	    let b = self.a * rhs.b + self.b * rhs.a + self.c * rhs.d - self.d * rhs.c;
	    let c = self.a * rhs.c - self.b * rhs.d + self.c * rhs.a - self.d * rhs.b;
	    let d = self.a * rhs.d + self.b * rhs.c - self.c * rhs.b + self.d * rhs.a;

        Quaternion {
            a, b, c, d 
        }
    }
}

#[cfg(target_arch="x86_64")]
impl std::ops::Mul for Quaternion<f32> {

    #[inline(always)]
    default fn mul(self, rhs: Quaternion<f32>) -> Quaternion<f32> {
        let unpacked: (f32, f32, f32, f32);
        unsafe {
            let a = _mm_set_ps( self.d, self.c, self.b, self.a );
            let b = _mm_set_ps( rhs.d, rhs.c, rhs.b, rhs.a );
            let a1123 = _mm_shuffle_ps(a,a,0xE5);
            let a2231 = _mm_shuffle_ps(a,a,0x7A);
            let b1000 = _mm_shuffle_ps(b,b,0x01);
            let b2312 = _mm_shuffle_ps(b,b,0x9E);
            let t1    = _mm_mul_ps(a1123, b1000);
            let t2    = _mm_mul_ps(a2231, b2312);
            let t12   = _mm_add_ps(t1, t2);
            let mask =_mm_set_epi32(0,0,0,-2147483648i32); //0x80000000
            let t12m  = _mm_xor_ps(t12, _mm_castsi128_ps(mask)); // flip sign bits
            let a3312 = _mm_shuffle_ps(a,a,0x9F);
            let b3231 = _mm_shuffle_ps(b,b,0x7B);
            let a0000 = _mm_shuffle_ps(a,a,0x00);
            let t3    = _mm_mul_ps(a3312, b3231);
            let t0    = _mm_mul_ps(a0000, b);
            let t03   = _mm_sub_ps(t0, t3);
            let res  = _mm_add_ps(t03, t12m);
            unpacked = std::mem::transmute( res );
        }

        Quaternion {
            a: unpacked.3,
            b: unpacked.2,
            c: unpacked.1,
            d: unpacked.0,
        }
    }
}

impl<T> std::ops::MulAssign for Quaternion<T> where  T: Numeral {
    fn mul_assign( &mut self, rhs: Quaternion<T> ) {
        *self = *self * rhs;
    }
}


/////////////////////////////////////////////////////////////////
/// Unit Tests 
/////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;

    #[test]
    fn basic_add() {
        let mut a = Quaternion::from_vals( 1.0, -2.0, 3.0, -4.0 );
        let bvec = vec![1.0f32, 1.0, 1.0, 1.0];
        let b = Quaternion::from_vec( &bvec );

        a += b;
        
        let mut res = a.to_vec();
        let mut tv = vec![2.0f32, -1.0, 4.0, -3.0];
        assert!(res.iter().zip(tv).all( |(x,y)| (x - y).abs() < 1e-9 ));

        a = a + b;

        res = a.to_vec();
        tv = vec![3.0f32, 0.0, 5.0, -2.0];
        assert!(res.iter().zip(tv).all( |(x,y)| (x - y).abs() < 1e-9 ));
    }

    #[test]
    fn conjugate() {
        let mut a = Quaternion::from_vals( 1.0, 1.0, 1.0, 1.0 );
        a.conjugate();

        let tv = vec![1.0f32, -1.0, -1.0, -1.0];
        assert!( a.to_vec().iter().zip(tv).all( |(x,y)| (x - y).abs() < 1e-9 ));
    }

    #[bench]
    fn mul_f64(b: &mut Bencher) {
        let mut x = Quaternion::from_vals( 0.5f64, -0.5, 0.5, -0.5 );
        let y = Quaternion::from_vals( 0.5725695175851593f64, 0.5597722931244762, 
                                       0.5597722931244762, 0.2132465878569433 );

        b.iter(|| x *= y )
    }

    #[bench]
    fn mul_f32(b: &mut Bencher) {
        let mut x = Quaternion::from_vals( 0.5f32, -0.5, 0.5, -0.5 );
        let y = Quaternion::from_vals( 0.5725695175851593f32, 0.5597722931244762, 
                                       0.5597722931244762, 0.2132465878569433 );

        b.iter(|| x *= y )
    }
}


