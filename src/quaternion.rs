extern crate test;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std;
use std::fmt;

#[derive(Copy, Clone)]
pub struct Quaternion{
    pb: __m128 
}

impl Quaternion {
    pub fn from_vals( a: f32, b: f32, c: f32, d: f32 ) 
        -> Quaternion {
        Quaternion{ 
            pb: unsafe{_mm_set_ps( d, c, b, a )}
        }
    }

    pub fn from_vec( x: &Vec<f32> )
        -> Quaternion {
        Quaternion {
            pb: unsafe{ _mm_loadu_ps( &x[0] ) }
        }
    }

    pub fn to_vec ( self ) -> Vec<f32> {
        let unpacked: (f32, f32, f32, f32);
        unsafe {
            unpacked = std::mem::transmute( self.pb );
        }
        vec![unpacked.0, unpacked.1, unpacked.2, unpacked.3]
    }

    /// Flips sign of i, j, and k components
    #[inline(always)]
    pub fn conjugate( &mut self ) {
        unsafe {
            let mask  = _mm_set_epi32(-2147483648i32,-2147483648i32,-2147483648i32,0); //Sign bits of lower three registers 
            self.pb  = _mm_xor_ps(self.pb, _mm_castsi128_ps(mask)); // flip sign bits       
        }
    }

    #[inline(always)]
    pub fn scalar_add( &mut self, x: f32 ) {
        unsafe {
            let xx = _mm_set_ps( x, x, x, x );
            self.pb = _mm_add_ps( self.pb, xx );
        }
    }

    #[inline(always)]
    pub fn scalar_mul( &mut self, x: f32 ) {
        unsafe {
            let xx = _mm_set_ps( x, x, x, x );
            self.pb = _mm_mul_ps( self.pb, xx );
        }
    }
}

impl Default for Quaternion {
    fn default() -> Quaternion {
        Quaternion {
            pb: unsafe{ _mm_set_ps( 0.0, 0.0, 0.0, 0.0 ) }
        }
    }
}

impl std::ops::AddAssign for Quaternion {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Quaternion) {
        self.pb =  unsafe { _mm_add_ps(self.pb, rhs.pb) };
    }
}

impl std::ops::Add for Quaternion {
    type Output = Quaternion;

    #[inline(always)]
    fn add(self, rhs: Quaternion) -> Quaternion {
        Quaternion{ 
            pb: unsafe { _mm_add_ps(self.pb, rhs.pb) } 
        }
    }
}

impl std::ops::SubAssign for Quaternion {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Quaternion) {
        self.pb =  unsafe { _mm_sub_ps(self.pb, rhs.pb) };
    }
}

impl std::ops::Sub for Quaternion {
    type Output = Quaternion;

    #[inline(always)]
    fn sub(self, rhs: Quaternion) -> Quaternion {
        Quaternion{ 
            pb: unsafe { _mm_sub_ps(self.pb, rhs.pb) } 
        }
    }
}

impl std::ops::Mul for Quaternion {
    type Output = Quaternion;

    #[inline(always)]
    fn mul ( self, rhs: Quaternion) -> Quaternion
    {  
        let a = self.pb;
        let b = rhs.pb;

        let res;
        unsafe {
            let a1123 = _mm_shuffle_ps(a,a,0xE5);
            let a2231 = _mm_shuffle_ps(a,a,0x7A);
            let b1000 = _mm_shuffle_ps(b,b,0x01);
            let b2312 = _mm_shuffle_ps(b,b,0x9E);
            let t1    = _mm_mul_ps(a1123, b1000);
            let t2    = _mm_mul_ps(a2231, b2312);
            let t12   = _mm_add_ps(t1, t2);
            let mask  = _mm_set_epi32(0,0,0,-2147483648i32); //0x80000000
            let t12m  = _mm_xor_ps(t12, _mm_castsi128_ps(mask)); // flip sign bits
            let a3312 = _mm_shuffle_ps(a,a,0x9F);
            let b3231 = _mm_shuffle_ps(b,b,0x7B);
            let a0000 = _mm_shuffle_ps(a,a,0x00);
            let t3    = _mm_mul_ps(a3312, b3231);
            let t0    = _mm_mul_ps(a0000, b);
            let t03   = _mm_sub_ps(t0, t3);
            res       = _mm_add_ps(t03, t12m);
        }

        Quaternion{ pb: res }
    }
}

impl std::ops::MulAssign for Quaternion {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Quaternion) {
        let a = self.pb;
        let b = rhs.pb;

        unsafe {
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
            self.pb   = _mm_add_ps(t03, t12m);
        }
    }
}

impl std::ops::Div for Quaternion {
    type Output = Quaternion;

    #[inline(always)]
    fn div ( self, rhs: Quaternion) -> Quaternion
    {
        let res;
        unsafe {
            let a = self.pb;
            let mut b = _mm_move_ss( rhs.pb, rhs.pb ); //Create a mutable copy of denominator

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
            res = _mm_div_ps( prod, normsq );

        }

        Quaternion{ pb: res }
    }
}

impl std::ops::DivAssign for Quaternion {
    #[inline(always)]
    fn div_assign( &mut self, rhs: Quaternion ) {
        unsafe {
            let a = self.pb;
            let mut b = _mm_move_ss( rhs.pb, rhs.pb ); //Create a mutable copy of denominator

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
            self.pb = _mm_div_ps( prod, normsq );
        }
    }
}

impl std::ops::Neg for Quaternion {
    type Output = Quaternion;

    #[inline(always)]
    fn neg(self) -> Quaternion {
        let res;
        unsafe {
            let mask  = _mm_set_epi32(-2147483648i32,-2147483648i32,-2147483648i32,-2147483648i32); //Sign bits 
            res = _mm_xor_ps( self.pb, _mm_castsi128_ps(mask) );
        }

        Quaternion{ pb: res }
    }
}

/// XXX: Not sure if this is the fastest way to do this.  Can the processor branch on an _m128?
impl std::cmp::PartialEq for Quaternion {
    fn eq( &self, other: &Quaternion) -> bool {

        let unpacked: (f64, f64);
        unsafe {
            let cmp = _mm_cmpneq_ps( self.pb, other.pb );
            unpacked = std::mem::transmute( cmp );
        }
        if unpacked.0 == 0.0 && unpacked.1 == 0.0 {
            return true;
        }
    
        false
    }
}


impl fmt::Display for Quaternion {
    default fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let q = self.to_vec();

        let mut s = String::new();
        s += &format!("({} + {}*I + {}*J + {}*K)", q[0], q[1], q[2], q[3]);
        write!(f, "{}", s)
    }
}

/// XXX: This is a shell class just for benchmarking.  Not sure how much it makes sense
/// to flush this out like the SIMD version
#[derive(Copy, Clone)]
pub struct QuaternionNonSIMD {
    a: f32,
    b: f32,
    c: f32,
    d: f32
}

impl QuaternionNonSIMD {
    pub fn new( a: f32, b: f32, c:f32, d:f32 )
        -> QuaternionNonSIMD {
        QuaternionNonSIMD {
            a: a,
            b: b,
            c: c,
            d: d,
        }
    }
}

impl std::ops::MulAssign for QuaternionNonSIMD {
    fn mul_assign(&mut self, rhs: QuaternionNonSIMD) {
    	self.a = self.a * rhs.a - self.b * rhs.b - self.c * rhs.c - self.d * rhs.d;
	    self.b = self.a * rhs.b + self.b * rhs.a + self.c * rhs.d - self.d * rhs.c;
	    self.c = self.a * rhs.c - self.b * rhs.d + self.c * rhs.a - self.d * rhs.b;
	    self.d = self.a * rhs.d + self.b * rhs.c - self.c * rhs.b + self.d * rhs.a;
    }
}

impl std::ops::DivAssign for QuaternionNonSIMD {
    fn div_assign(&mut self, rhs: QuaternionNonSIMD) {
        let rhs_conj = QuaternionNonSIMD{ a: rhs.a, b: -1.0*rhs.b, 
                                          c: -1.0*rhs.c, d: -1.0*rhs.d };

        *self *= rhs_conj;

        let norm_rhs = rhs.a * rhs.a + rhs.b * rhs.b + rhs.c * rhs.c + rhs.d * rhs.d;

        self.a /= norm_rhs; self.b /= norm_rhs;
        self.c /= norm_rhs; self.d /= norm_rhs;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;
    
    #[test]
    fn basic_assign() {
        let avec = vec![1.0f32, -2.0, 3.0, -4.0];
        let a = Quaternion::from_vals( 1.0, -2.0, 3.0, -4.0 );
        let b = Quaternion::from_vec( &avec );

        assert!(a.to_vec().iter().zip(b.to_vec()).all( |(x,y)| (x-y).abs()<1e-9 ));
        assert!(a.to_vec().iter().zip(avec).all( |(x,y)| (x-y).abs()<1e-9 ));
    }

    #[test]
    fn basic_add() {
        if cfg!(target_arch = "x86_64") {
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

        } else {
            assert!( false );
        }
    }

    
    #[test]
    fn basic_mul() {
        if cfg!(target_arch = "x86_64") {
            let bvec = vec![1.0f32, 2.0, 3.0, 4.0];
            let mut a = Quaternion::from_vals( 2.0, 2.0, 2.0, 2.0 );
            let mut b = Quaternion::from_vec( &bvec );

            a *= b;
            
            let mut res = a.to_vec();
            let mut tv = vec![-16.0f32, 8.0, 4.0, 12.0];
            assert!(res.iter().zip(tv).all( |(x,y)| (x - y).abs() < 1e-9 ));

            b = Quaternion::from_vals( -0.25, 0.125, 0.1, 0.125 );
            a = a * b;

            res = a.to_vec();
            tv = vec![1.1, -4.7, -2.1, -4.7];
            assert!(res.iter().zip(tv).all( |(x,y)| (x - y).abs() < 1e-9 ));
        } else {
            assert!( false );
        }
    }

    #[test]
    fn basic_div() {
        if cfg!(target_arch = "x86_64") {
            let mut a = Quaternion::from_vals( 2.0, 2.0, 2.0, 2.0 );
            let bvec = vec![1.0f32, 2.0, 3.0, 4.0];
            let mut b = Quaternion::from_vec( &bvec );

            a = a / b;
            
            let mut res = a.to_vec();
            let mut tv = vec![2.0f32/3.0, -2.0/15.0, 0.0, -4.0/15.0];
            assert!(res.iter().zip(tv).all( |(x,y)| (x - y).abs() < 1e-9 ));

            b = Quaternion::from_vals( -0.25, 0.125, 0.1, 0.125 );
            a /= b;
            res = a.to_vec();
            tv = vec![-520.0f32/249.0, -184.0/249.0, -40.0/83.0, -8.0/249.0];
            assert!(res.iter().zip(tv).all( |(x,y)| (x - y).abs() < 1e-6 ));
        } else {
            assert!( false );
        }
    }

    #[test]
    fn negate() {
        if cfg!(target_arch = "x86_64") {
            let mut a = Quaternion::from_vals( 1.0, 1.0, 1.0, 1.0 );
            a = -a;

            let tv = vec![-1.0f32, -1.0, -1.0, -1.0];
            assert!( a.to_vec().iter().zip(tv).all( |(x,y)| (x - y).abs() < 1e-9 ));
        } else {
            assert!( false );
        }   
    }

    #[test]
    fn conjugate() {
        if cfg!(target_arch = "x86_64") {
            let mut a = Quaternion::from_vals( 1.0, 1.0, 1.0, 1.0 );
            a.conjugate();

            let tv = vec![1.0f32, -1.0, -1.0, -1.0];
            assert!( a.to_vec().iter().zip(tv).all( |(x,y)| (x - y).abs() < 1e-9 ));
        } else {
            assert!( false );
        }   
    }

    #[bench]
    fn sse_mul(b: &mut Bencher) {
        let mut x = Quaternion::from_vals( 0.5, -0.25, 0.125, -0.1 );
        let y = Quaternion::from_vals( 1.0, 1.25, 0.125, -1.5 );

        b.iter(|| x *= y )
    }

    #[bench]
    fn plain_mul(b: &mut Bencher) {
        let mut x = QuaternionNonSIMD::new( 0.5, -0.25, 0.125, -0.1 );
        let y = QuaternionNonSIMD::new( 1.0, 1.25, 0.125, -1.5 );

        b.iter(|| x *= y )
    }

    #[bench]
    fn sse_div(b: &mut Bencher) {
        let mut x = Quaternion::from_vals( 0.5, -0.25, 0.125, -0.1 );
        let y = Quaternion::from_vals( 1.0, 1.25, 0.125, -1.5 );

        b.iter(|| x /= y )
    }

    #[bench]
    fn plain_div(b: &mut Bencher) {
        let mut x = QuaternionNonSIMD::new( 0.5, -0.25, 0.125, -0.1 );
        let y = QuaternionNonSIMD::new( 1.0, 1.25, 0.125, -1.5 );

        b.iter(|| x /= y )
    }
}
