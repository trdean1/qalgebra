extern crate num_complex;
extern crate test;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use complex::num_complex::Complex;
use std;

#[derive(Copy, Clone)]
pub struct ComplexSIMD64 {
    pb: __m128d
}

impl ComplexSIMD64 {
    pub fn from_vals( a: f64, b: f64 ) -> ComplexSIMD64 {
        ComplexSIMD64 {
            pb: unsafe{ _mm_set_pd( b, a ) }
        }
    }
}

impl std::ops::MulAssign for ComplexSIMD64 {
    #[inline(always)]
    fn mul_assign( &mut self, rhs: ComplexSIMD64) {
        unsafe {
            let mut num1 = _mm_shuffle_pd( self.pb, self.pb, 0 ); //load real part
            let mut num2 = _mm_mul_pd( rhs.pb, num1 ); // y.re * x.re + y.im * x.re
            num1 = _mm_shuffle_pd( self.pb, self.pb, 1 ); //load imaginary part
            num2 = _mm_shuffle_pd( num2, num2, 1 ); //swap values
            let num3 = _mm_mul_pd( num2, num1 );
            self.pb = _mm_addsub_pd( num3, num2 );
        }       
    }
}

#[derive(Copy, Clone)]
pub struct ComplexSIMD32 {
    pb: __m128
}

impl ComplexSIMD32 {
    pub fn from_vals( a: f32, b: f32 ) -> ComplexSIMD32 {
        ComplexSIMD32 {
            pb: unsafe{ _mm_set_ps( 0.0f32, 0.0f32, b, a ) }
        }
    }

    pub fn to_std_complex( self ) -> Complex<f32> {
        let unpacked: (f32, f32, f32, f32);
        unsafe {
            unpacked = std::mem::transmute( self.pb );
        }

        Complex {
            re: unpacked.0,
            im: unpacked.1,
        }
    }
}

impl std::ops::MulAssign for ComplexSIMD32 {
    #[inline(always)]
    fn mul_assign( &mut self, rhs: ComplexSIMD32) {
        let x = self.pb;
        let y = rhs.pb;
        unsafe {
            //load      a     | b       | c  | d
            //load      c     | d       | b  | a
            //mul       ac    | bd      | bc | ad
            //load      ac    | bc      | 0  | 0
            //load      bd    | ad      | 0  | 0
            //addsub    ac-bd | bc + ad | 0  | 0
            let abab = _mm_shuffle_ps( x, x, 0x44 );
            let cddc = _mm_shuffle_ps( y, y, 0x14 );
            let prod = _mm_mul_ps( abab, cddc );
            let p1   = _mm_shuffle_ps( prod, prod, 0x08 );
            let p2   = _mm_shuffle_ps( prod, prod, 0x0D );
            self.pb  = _mm_addsub_ps( p1, p2 );
            
            /*
            asm!("
            movaps ($0), %xmm0
            movaps ($1), %xmm1
            movaps %xmm1, %xmm2
            shufps $$0x44, %xmm0, %xmm2       
            shufps $$0x14, %xmm1, %xmm0        
            mulps  %xmm2, %xmm0
            movaps %xmm0, %xmm1
            shufps $$0x8, %xmm0, %xmm1       
            shufps $$0xd, %xmm0, %xmm0        
            addsubps %xmm0, %xmm1
            movaps %xmm1, ($0)"
            : "+r"(&self.pb)
            : "r"(&rhs.pb)
            : "xmm0","xmm1","xmm2"
            : "volatile" );
            */
        }    
    }
}

pub fn simd_mul_assign64( x: &mut Complex<f64>, y: Complex<f64> ) {
    let unpacked: (f64, f64);
    unsafe {
        let mut num1 = _mm_loaddup_pd( &x.re );
        let mut num2 = _mm_set_pd( y.im, y.re );
        let mut num3 = _mm_mul_pd( num2, num1 );
        num1 = _mm_loaddup_pd( &x.im );
        num2 = _mm_shuffle_pd( num2, num2, 1 );
        num2 = _mm_mul_pd( num2, num1 );
        num3 = _mm_addsub_pd( num3, num2 );
        unpacked = std::mem::transmute( num3 );
    }

    x.im = unpacked.1;
    x.re = unpacked.0;
}

pub fn simd_mul_assign32( x: &mut Complex<f32>, y: Complex<f32> ) {
    let unpacked: (f32, f32, f32, f32);
    let have_nan; 
    unsafe {
        //This should be IEEE compliant as long as no inputs are NAN
        //and we don't overflow
        //
        //load      a     | b     | c  | d
        //load      c     | d     | b  | a
        //mul       ac    | bd    | bc | ad
        //load      ac    | bc    | 0  | 0
        //load      bd    | ad    | 0  | 0
        //addsub    ac-bd | bc+ad | 0  | 0
        let abcd = _mm_set_ps( y.im, y.re, x.im, x.re );
        let cdba = _mm_shuffle_ps( abcd, abcd, 0x1E );
        let prod = _mm_mul_ps( abcd, cdba );
        let p1   = _mm_shuffle_ps( prod, prod, 0x08 );
        let p2   = _mm_shuffle_ps( prod, prod, 0x0D );
        let p3   = _mm_addsub_ps( p1, p2 );
        unpacked = std::mem::transmute( p3 );

        let n = _mm_cmpneq_ps( p3, p3 );
        have_nan = _mm_movemask_ps(n);
    }

    if have_nan == 0 {
        x.im = unpacked.1;
        x.re = unpacked.0;
    } else { //If we have a NAN just fall back to compliant multiply
        *x *= y;
    }
}



#[cfg(test)]
mod tests{
    use super::*;
    use test::Bencher;

    #[test]
    fn mul_test64_std() {
        let mut x = Complex { re: 0.707f64, im: -0.707f64 };
        let y = Complex { re: 0.5f64, im: -0.8124f64 };

        let prod_std = x * y;
        simd_mul_assign64( &mut x, y );

        assert!( (prod_std - x).norm() < 1e-10 );
    }

    #[test]
    fn mul_test32_std() {
        let mut x = Complex { re: 0.1f32, im: -0.2f32 };
        let y = Complex { re: 0.3f32, im: -0.4f32 };

        let mut xm128 = ComplexSIMD32::from_vals(0.1f32, -0.2f32);
        let ym128 = ComplexSIMD32::from_vals(0.3f32, -0.4f32);

        let prod_std = x * y;
        simd_mul_assign32( &mut x, y );
        xm128 *= ym128;
        let x2 = xm128.to_std_complex();

        println!("Expected: {}", x);
        println!("Obtained: {}", x2);

        assert!( (prod_std - x).norm() < 1e-9 );
        assert!( (prod_std - x2).norm() < 1e-9 );
    }

    #[bench]
    fn nonsimd_mul64(b: &mut Bencher) {
        let mut x = Complex { re: 0.707107f64, im: -0.707107f64 };
        let y = Complex { re: 0.5f64, im: -0.866025f64 };

        b.iter(|| x *= y); 
    }

    #[bench]
    fn nonsimd_mul32(b: &mut Bencher) {
        let mut x = Complex { re: 0.707107f32, im: -0.707107f32 };
        let y = Complex { re: 0.5f32, im: -0.866025f32 };

        b.iter(|| x *= y); 
    }

    #[bench]
    fn simd_mul64(b: &mut Bencher) {
        let mut x = ComplexSIMD64::from_vals(0.707107f64, -0.707107f64);
        let y = ComplexSIMD64::from_vals(0.5f64, -0.866025f64);

        b.iter(|| x *= y ); 
    }

    #[bench]
    fn msimd_mul32(b: &mut Bencher) {
        let mut x = ComplexSIMD32::from_vals(0.707107, -0.707107);
        let y = ComplexSIMD32::from_vals(0.5, -0.866025);

        b.iter(|| x *= y ); 
    }

    #[bench]
    fn simd_mul_conv64(b: &mut Bencher) {
        let mut x = Complex { re: 0.707107f64, im: -0.707107f64 };
        let y = Complex { re: 0.5f64, im: -0.866025f64 };

        b.iter(|| simd_mul_assign64( &mut x, y ) ); 
    }

    #[bench]
    fn simd_mul_conv32(b: &mut Bencher) {
        let mut x = Complex { re: 0.707107f32, im: -0.707107f32 };
        let y = Complex { re: 0.5f32, im: -0.866025f32 };

        b.iter(|| simd_mul_assign32( &mut x, y ) ); 
    }
}
