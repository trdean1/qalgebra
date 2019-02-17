extern crate test;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std;
use std::fmt;
use std::error::Error;
use num_traits::sign::Signed;
use num_traits::{Num,Zero,One};
use num_traits::cast::ToPrimitive;

use Numeral;
use AlmostEq;

mod cast;

#[derive(Copy, Clone, Debug)]
pub struct Quaternion<T: Numeral + Signed>{
    a: T,
    b: T,
    c: T,
    d: T,
}

impl<T> Quaternion<T> where T: Numeral + Signed {
    pub fn from_vals( a: T, b: T, c: T, d: T ) 
        -> Self {
        Self {
           a, b, c, d 
        }
    }

    pub fn from_vec( x: &Vec<T> )
        -> Self {
        Self {
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

impl<T> Zero for Quaternion<T> where T: Numeral + Signed {
    fn zero() -> Self {
        Self {
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

/////////////////////////////////////////////////////////////////
/// Misc Traits
/////////////////////////////////////////////////////////////////

impl<T> One for Quaternion<T> where T: Numeral + Signed {
    fn one() -> Self {
        Self {
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

impl<T> std::cmp::PartialEq for Quaternion<T> where T: Numeral + Signed {
    fn eq( &self, other: &Self ) -> bool {
        self.a == other.a && self.b == other.b &&
        self.c == other.c && self.d == other.d
    }
}

//Consider overloading this for different types...i.e. accept lower
//precision for f32 vs f64
impl<T> AlmostEq for Quaternion<T> where T: Numeral + ToPrimitive + Signed {
    fn almost_eq( &self, other: &Self ) -> bool {
        let norm_diff = (*self - *other).norm();
         norm_diff < 1e-5
    }
}

//Norm should map a quaternion of any type to R so I think f64 is a reasonable 
//return type.  Might consider changing if arch is not x86_64
trait Norm<T> {
    fn norm( self ) -> f64;
}

impl<T> Norm<T> for Quaternion<T> where T: Numeral + ToPrimitive + Signed {
    fn norm( self ) -> f64 {
        let norm_sq = self.a * self.a + self.b * self.b + 
                      self.c * self.c + self.d * self.d;
        let nf64_maybe = norm_sq.to_f64();
        if let Some(norm_sq) = nf64_maybe {
            return norm_sq.sqrt();
        } else {
            return std::f64::NAN;
        }
    }
}

impl<T> fmt::Display for Quaternion<T> where T: Numeral + Signed {
    default fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut s = String::new();
        s += &format!("({} + {}*I + {}*J + {}*K)", self.a, self.b, 
                                                   self.c, self.d);
        write!(f, "{}", s)
    }
}

impl<T> Num for Quaternion<T> where T: Numeral + Signed {
    type FromStrRadixErr = ParseQuaternionError<T::FromStrRadixErr>;

    /// Parses `a +/- bi`; `ai +/- b`; `a`; or `bi` where `a` and `b` are of type `T`
    fn from_str_radix(_s: &str, _radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        Err( ParseQuaternionError { kind: QuaternionErrorKind::NotImplementedError } ) 
    }
}

/////////////////////////////////////////////////////////////////
/// Scalar operations
/////////////////////////////////////////////////////////////////

trait ScalarOps<T> {
    fn scalar_add( &mut self, T );

    fn scalar_mul( &mut self, T );
}

impl<T> ScalarOps<T> for Quaternion<T> where T: Numeral + Signed {
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
    fn conjugate( self ) -> Self;
}

impl<T> Conjugate<T> for Quaternion<T> where T: Numeral + Signed {
    fn conjugate( self ) -> Self {
        Self {
            a:  self.a,
            b: -self.b,
            c: -self.c,
            d: -self.d,
        }
    }
}

impl<T> std::ops::Neg for Quaternion<T> where T: Numeral + Signed + 
                                                 std::ops::Neg + 
                                                 std::ops::Neg<Output=T> {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        Self {
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

impl<T> std::ops::Add for Quaternion<T> where T: Numeral + Signed {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            a: self.a + rhs.a,
            b: self.b + rhs.b,
            c: self.c + rhs.c,
            d: self.d + rhs.d,
        }
    }
}

impl<T> std::ops::AddAssign for Quaternion<T> where T: Numeral + Signed {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<T> std::ops::Sub for Quaternion<T> where T: Numeral + Signed {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            a: self.a - rhs.a,
            b: self.b - rhs.b,
            c: self.c - rhs.c,
            d: self.d - rhs.d,
        }
    }
}

impl<T> std::ops::SubAssign for Quaternion<T> where T: Numeral + Signed {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<T> std::ops::Mul for Quaternion<T> where T: Numeral + Signed {
    type Output = Self;

    default fn mul( self, rhs: Self ) -> Self {
    	let a = self.a * rhs.a - self.b * rhs.b - self.c * rhs.c - self.d * rhs.d;
	    let b = self.a * rhs.b + self.b * rhs.a + self.c * rhs.d - self.d * rhs.c;
	    let c = self.a * rhs.c - self.b * rhs.d + self.c * rhs.a + self.d * rhs.b;
	    let d = self.a * rhs.d + self.b * rhs.c - self.c * rhs.b + self.d * rhs.a;

        Self {
            a, b, c, d 
        }
    }
}

#[cfg(target_arch="x86_64")]
impl std::ops::Mul for Quaternion<f32> {

    #[inline(always)]
    default fn mul(self, rhs: Self) -> Self {
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

        Self {
            a: unpacked.0,
            b: unpacked.1,
            c: unpacked.2,
            d: unpacked.3,
        }
    }
}

impl<T> std::ops::MulAssign for Quaternion<T> where  T: Numeral + Signed {
    fn mul_assign( &mut self, rhs: Self ) {
        *self = *self * rhs;
    }
}


//XXX: Need to think through what generic multiplication should be.  In general,
//it should be multiplication by the inverse of rhs, but inverse is hard to generalize.
//This trades of generality for speed --- would be faster to multiply by inverse rather
//than divide in final assignment but this wouldn't work for int's.  
//Will override for floats anyways.  
//Quaternions over Z+ are certainly not closed so I think requiring signed is fine
impl<T> std::ops::Div for Quaternion<T> where T: Numeral + Signed {
    type Output = Self;

    default fn div( self, rhs: Self ) -> Self {
        let rhs_conj = rhs.clone().conjugate();

        let res = self * rhs_conj;

        let norm_sq = rhs.a * rhs.a + rhs.b * rhs.b +
                      rhs.c * rhs.c + rhs.d * rhs.d;

        Self {
            a : res.a / norm_sq,
            b : res.b / norm_sq,
            c : res.c / norm_sq,
            d : res.d / norm_sq,
        }
    }
}

//This actaully tkes the same amount of time as the unoverloaded div
//so it probably only saves in terms of code length
#[cfg(target_arch="x86_64")]
impl std::ops::Div for Quaternion<f32> {
    #[inline(always)]
    default fn div( self, rhs: Self ) -> Self {
        let unpacked: (f32, f32, f32, f32);
        unsafe {
            let a = _mm_set_ps( self.d, self.c, self.b, self.a );
            let mut b = _mm_set_ps( rhs.d, rhs.c, rhs.b, rhs.a );

            //Conjugate b
            let mask  = _mm_set_epi32(-2147483648i32,-2147483648i32,-2147483648i32,0); //Sign bits of lower three registers 
            b = _mm_xor_ps(b, _mm_castsi128_ps(mask)); // flip sign bits    

            //multiply a*conj(b)
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
            let prod  = _mm_add_ps(t03, t12m);

            //Compute norm b
            let bb = _mm_mul_ps( b, b ); //b^2

            //Horizontal add of aa
            let shuf = _mm_shuffle_ps ( bb, bb, 0x1b ); //reverse order
            let sums = _mm_add_ps(bb, shuf); //0+3, 1+2, 1+2, 0+3
            let swap = _mm_shuffle_ps( sums, sums, 0x4e); // high half -> low half
            let normsq  = _mm_add_ps(sums, swap);

            //Divide by norm --- WARNING: SLOW hard to avoid...it's division
            let res = _mm_div_ps( prod, normsq );

            unpacked = std::mem::transmute( res );
        }

        Self {
            a: unpacked.0,
            b: unpacked.1,
            c: unpacked.2,
            d: unpacked.3
        }
    }
}

impl<T> std::ops::DivAssign for Quaternion<T> where T: Numeral + Signed {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

// Attempts to identify the quaternion equivalent of a gaussian integer
// whose product with `modulus` is closest to `self`.
impl<T> std::ops::Rem for Quaternion<T> where T: Numeral + Signed {
    type Output = Self;

    #[inline]
    fn rem(self, modulus: Self) -> Self {
        let Self { a, b, c, d } = self.clone() / modulus.clone();
        // This is the gaussian integer corresponding to the true ratio
        // rounded towards zero.
        let (a0, b0, c0, d0) = (a.clone() - a % T::one(), b.clone() - b % T::one(), 
                                c.clone() - c % T::one(), d.clone() - d % T::one() );

        self - modulus * Self::from_vals(a0, b0, c0, d0)
    }
}

impl<T> std::ops::RemAssign for Quaternion<T> where T: Numeral + Signed {
    fn rem_assign( &mut self, modulus: Self ) {
        *self = *self % modulus;
    }
}

/////////////////////////////////////////////////////////////////
/// Parse Errors 
/////////////////////////////////////////////////////////////////

#[derive(Debug, PartialEq)]
pub struct ParseQuaternionError<E> {
    kind: QuaternionErrorKind<E>,
}

#[derive(Debug, PartialEq)]
#[allow(dead_code)]
enum QuaternionErrorKind<E> {
    ParseError(E),
    ExprError,
    NotImplementedError,
}

#[allow(dead_code)]
impl<E> ParseQuaternionError<E> {
    fn new() -> Self {
        ParseQuaternionError {
            kind: QuaternionErrorKind::ExprError,
        }
    }

    fn from_error(error: E) -> Self {
        ParseQuaternionError {
            kind: QuaternionErrorKind::ParseError(error),
        }
    }
}

impl<E: Error> Error for ParseQuaternionError<E> {
    fn description(&self) -> &str {
        match self.kind {
            QuaternionErrorKind::ParseError(ref e) => e.description(),
            QuaternionErrorKind::ExprError => "invalid or unsupported complex expression",
            QuaternionErrorKind::NotImplementedError => "not yet implemented",
        }
    }
}

impl<E: fmt::Display> fmt::Display for ParseQuaternionError<E> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.kind {
            QuaternionErrorKind::ParseError(ref e) => e.fmt(f),
            QuaternionErrorKind::ExprError => "invalid or unsupported complex expression".fmt(f),
            QuaternionErrorKind::NotImplementedError => "not yet implemented".fmt(f),
        }
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
    fn basic_sub() {
        let mut a = Quaternion::from_vals( 1.0f32, -2.0, 3.0, -4.0 );
        let bvec = vec![1.0f32, 1.0, 1.0, 1.0];
        let b = Quaternion::from_vec( &bvec );

        a -= b;
        
        let mut res = a.to_vec();
        let mut tv = vec![0.0f32, -3.0, 2.0, -5.0];
        assert!(res.iter().zip(tv).all( |(x,y)| (x - y).abs() < 1e-9 ));

        a = a - b;

        res = a.to_vec();
        tv = vec![-1.0f32, -4.0, 1.0, -6.0];
        assert!(res.iter().zip(tv).all( |(x,y)| (x - y).abs() < 1e-9 ));
    }

    #[test]
    fn basic_mul32() {
        let bvec = vec![1.0f32, -2.0, 3.0, -4.0];
        let mut a = Quaternion::from_vals( -2.0, 2.0, 2.0, 2.0 );
        let mut b = Quaternion::from_vec( &bvec );

        a *= b;
        
        let mut tv = Quaternion::from_vals( 4.0f32, -8.0, 0.0, 20.0 );

        assert!( a.almost_eq( &tv ) );
        
        b = Quaternion::from_vals( -0.25, 0.125, 0.1, 0.125 );
        a = a * b;

        tv = Quaternion::from_vals( -2.5, 0.5, 3.9, -5.3 );

        assert!( a.almost_eq(&tv) );
    }

    #[test]
    fn basic_mul64() {
        let bvec = vec![1.0f64, 2.0, 3.0, 4.0];
        let mut a = Quaternion::from_vals( 2.0, 2.0, 2.0, 2.0 );
        let mut b = Quaternion::from_vec( &bvec );

        a *= b;
        
        let mut tv = Quaternion::from_vals( -16.0f64, 8.0, 4.0, 12.0 );

        assert!( a.almost_eq( &tv ) );
        
        b = Quaternion::from_vals( -0.25, 0.125, 0.1, 0.125 );
        a = a * b;

        tv = Quaternion::from_vals( 1.1, -4.7, -2.1, -4.7 );

        assert!( a.almost_eq(&tv) );
    }

    #[test]
    fn basic_div32() {
        let mut a = Quaternion::from_vals( 2.0f32, 2.0, 2.0, 2.0 );
        let bvec = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut b = Quaternion::from_vec( &bvec );

        a = a / b;
        
        let mut tv = Quaternion::from_vals(2.0f32/3.0, -2.0/15.0, 0.0, -4.0/15.0);
        assert!( a.almost_eq( &tv ) );

        b = Quaternion::from_vals( -0.25, 0.125, 0.1, 0.125 );
        a /= b;
        tv = Quaternion::from_vals(-520.0f32/249.0, -184.0/249.0, -40.0/83.0, -8.0/249.0);

        println!(" a  = {}\n tv = {}", a, tv);
        println!(" diff = {}", (a - tv).norm() );
        assert! (a.almost_eq( &tv ) );
    }

    #[test]
    fn basic_div64() {
        let mut a = Quaternion::from_vals( 2.0f64, 2.0, 2.0, 2.0 );
        let bvec = vec![1.0f64, 2.0, 3.0, 4.0];
        let mut b = Quaternion::from_vec( &bvec );

        a = a / b;
        
        let mut tv = Quaternion::from_vals(2.0f64/3.0, -2.0/15.0, 0.0, -4.0/15.0);
        assert!( a.almost_eq( &tv ) );

        b = Quaternion::from_vals( -0.25, 0.125, 0.1, 0.125 );
        a /= b;
        tv = Quaternion::from_vals(-520.0f64/249.0, -184.0/249.0, -40.0/83.0, -8.0/249.0);

        println!(" a  = {}\n tv = {}", a, tv);
        println!(" diff = {}", (a - tv).norm() );
        assert! (a.almost_eq( &tv ) );
    }

    #[test]
    fn conjugate() {
        let a = Quaternion::from_vals( 1.0, 1.0, 1.0, 1.0 );
        let b = a.conjugate();

        let tv = vec![1.0f32, -1.0, -1.0, -1.0];
        assert!( b.to_vec().iter().zip(tv).all( |(x,y)| (x - y).abs() < 1e-9 ));
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

    #[bench]
    fn div_f64(b: &mut Bencher) {
        let mut x = Quaternion::from_vals( 0.5f64, -0.5, 0.5, -0.5 );
        let y = Quaternion::from_vals( 0.5725695175851593f64, 0.5597722931244762, 
                                       0.5597722931244762, 0.2132465878569433 );

        b.iter(|| x /= y )
    }

    #[bench]
    fn div_f32(b: &mut Bencher) {
        let mut x = Quaternion::from_vals( 0.5f32, -0.5, 0.5, -0.5 );
        let y = Quaternion::from_vals( 0.5725695175851593f32, 0.5597722931244762, 
                                       0.5597722931244762, 0.2132465878569433 );

        b.iter(|| x /= y )
    }
}


