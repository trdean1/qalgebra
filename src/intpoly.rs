#![feature(test)]
extern crate test;

extern crate rug;
use rug::{Assign,Integer};

//pub mod quaternion;
//use quaternion::Quaternion;
use std::fmt;

#[derive(Clone)]
pub struct IntPolynomial {
    coefficients: Vec<u32>
}

impl IntPolynomial {
    /// Create polynomial from vector.  v[0] is degree zero term
    pub fn from_vec(v: &Vec<u32>) -> IntPolynomial {
        IntPolynomial { 
            coefficients : v.clone()
        }
    }

    /// Create a polynomial with an empty coefficient vector that has
    /// space allocated for n coefficients
    pub fn with_capacity( n: usize ) -> IntPolynomial {
        let v = Vec::with_capacity( n );
        IntPolynomial {
            coefficients : v
        }
    }

    /// Create a polynomial with all n zero coefficients
    pub fn zeros( n: usize ) -> IntPolynomial {
        let v = vec![0; n];
        IntPolynomial {
            coefficients : v
        }
    }

    /// Return degree --- length of coefficient array minus one
    pub fn degree( &self ) -> usize {
        self.coefficients.len() - 1
    }

    /// Remove trailing zero terms
    fn trim( &mut self ) {
        while self.coefficients[ self.degree() ] == 0 {
            if self.coefficients.pop().is_none() {
                break;
            }

        }
    }

    /// Multiply all coefficients by c
    pub fn scalar_multiply( &mut self, c: u32 ) {
        for i in 0..self.coefficients.len(){
            self.coefficients[i] *= c;
        }
    }

    /// Multiply by x^n...shift coefficeints up n slots
    pub fn xn_multiply( &mut self, n: usize ) {
        for _ in 0..n {
            self.coefficients.insert( 0, 0 );
        }
    }

    /// Divide by x^n...shift down n slots and return remainder if not zero
    pub fn xn_divide( &mut self, n: usize) -> Option<IntPolynomial> {
        let mut remainder = vec![0; n];
        let mut has_rem = false;
        for i in 0..n {
            remainder[i] = self.coefficients.remove(0);
            if remainder[i] != 0 { has_rem = true; } //Save us iterating again
        }

        if has_rem {
            let mut p = IntPolynomial::from_vec( &remainder );
            p.trim();

            return Some(p);
        }

        None
    }

    /// Evaluate at x using Horner's method
    pub fn horner_eval( &self, x: u32 ) -> u32 {
        let mut res = 0;

        // FMA is slower if we don't have floats or aren't doing SIMD
        //if is_x86_feature_detected!("fma") {
        //    res = self.horner_eval_fma( x );
        //} else {

        for c in self {
            res = res * x + c;
        }

        res
    }

    /// multiply self by rhs returning a new polynomial.  Uses gradeschool multiplication
    /// which is slow unless degree is really small
    pub fn gradeschool_mul( &self, rhs: &IntPolynomial ) -> IntPolynomial {
        let ldegree = self.degree();
        let rdegree = rhs.degree();
        let outdegree = ldegree + rdegree;
        let ref small;
        let ref large;

        if ldegree < rdegree {
            small = self;
            large = rhs;
        } else {
            small = rhs;
            large = self;
        }

        let mut result = IntPolynomial::zeros( outdegree );
        for (i, c) in small.into_iter().enumerate() {
            //XXX Not sure if rust is smart enough to save memory of tmp each loop?
            let mut tmp = large.to_owned();
            tmp.scalar_multiply( c );
            tmp.xn_multiply( i );
            result += tmp;
        }

        result
    }

    /// Multiplies self by rhs creating a new polynomial.  Uses Kronecker substitution.
    /// Calls rug crate (which then call GMP) for arbitrary precision integer multipication
    pub fn kronecker_mul ( &self, rhs: &IntPolynomial ) -> IntPolynomial {
        let bits_per = self.kronecker_coeff_bits( rhs );

        let degree = self.degree() + rhs.degree() + 2;
        let bits_total = bits_per * (degree as u32);

        let product;

        println!("Total bits: {}", bits_total);
        if bits_total <= 64 {
            let packed_product = self.pack_small( bits_per ) * rhs.pack_small( bits_per );

            product = IntPolynomial::unpack_small( packed_product, bits_per )
        } else {
            let packed_self = self.pack_rug( bits_per );
            let packed_rhs = rhs.pack_rug( bits_per );
            let packed_product_incomplete = &packed_self * &packed_rhs;
            let mut packed_product = Integer::new();
            packed_product.assign( packed_product_incomplete );

            product = IntPolynomial::unpack_rug( packed_product, bits_per, Some(degree) );
        }

        return product;
    }

    /// Find the maximum number of bits each coefficient of the product of self and rhs
    /// can contain
    pub fn kronecker_coeff_bits( &self, rhs: &IntPolynomial ) -> u32 {
        let mut largest_c = 0;
        for c in self {
            if c > largest_c {
                largest_c = c;
            }
        }
        for c in rhs {
            if c > largest_c {
                largest_c = c;
            }
        }

        let deg_overhead = if self.degree() > rhs.degree() { 
            fastlog2( self.degree() as u32 + 1 ) + 1
        } else { 
            fastlog2( rhs.degree() as u32 + 1 ) + 1
        };

        2 * (fastlog2( largest_c ) + 1 ) + deg_overhead
    }

    /// Pack a (small) polynomial into a 64-bit int by evaluating at 2^bits
    pub fn pack_small( &self, bits: u32 ) -> u64 {
        let len = self.degree() + 1;
        assert!( bits * len as u32 <= 64 );

        let mut result = 0;
        for (idx, c) in self.into_iter().enumerate() {
            let i = idx as u32;
            let tmp = (c as u64) << (i*bits);
            result |= tmp;
        }

        result
    }

    /// Same as above but uses rug library to pack arbitrary large polynomials
    pub fn pack_rug( &self, bits: u32 ) -> Integer {
        let mut result = Integer::new();

        for i in (0..self.coefficients.len()).rev() {
            result += self[i];
            if i != 0 {
                result <<= bits;
            }
        }

        result
    }

    /// Unpack a 64-bit int into a (small) polynomial by assuming it was evaluated at
    /// 2^bits_per
    pub fn unpack_small( v: u64, bits_per: u32 ) -> IntPolynomial {
        let max_coeffs = 64 / bits_per;
        let mut coeffs = Vec::<u32>::with_capacity( max_coeffs as usize );
        let mut vv = v;

        let mask = u64::pow( 2, bits_per ) - 1;
        for _ in 0 .. max_coeffs {
            coeffs.push( (vv & mask) as u32 );
            vv >>= bits_per; 
            if vv == 0 {
                break;
            }
        }

        IntPolynomial {
            coefficients: coeffs
        }
    }

    /// Same as above but using rug
    /// If we know the degree (we should if we are multiplying) then we can pass it in
    /// to save time
    pub fn unpack_rug( v: Integer, bits_per: u32, degree_maybe: Option<usize> ) -> IntPolynomial {
        let mut coeffs = Vec::<u32>::new();
        if degree_maybe.is_some() {
            coeffs.reserve( degree_maybe.unwrap() + 1 );
        }

        let mut vv = Integer::new();
        vv += v;

        let mask = u32::pow( 2, bits_per ) - 1;

        while vv > 0 {
            let low = vv.to_u32_wrapping() & mask;
            coeffs.push( low );
            vv >>= bits_per;
        }

        IntPolynomial {
            coefficients: coeffs
        }
    }
}

/// An O(log n) algorithm to give floor(log v) for a 32-bit number
fn fastlog2( v: u32 ) -> u32 {
    let mut res = 0;
    let mut tmp = v;
    let b = [0x2, 0xC, 0xF0, 0xFF00, 0xFFFF0000];
    let s = [1, 2, 4, 8, 16];

    for i in (0..5).rev() {
        if (tmp & b[i]) != 0 {
            tmp >>= s[i];
            res |= s[i];
        }
    }

    res
}

//////////////////////////////////////////////////////////////
/// Operator Overloads
//////////////////////////////////////////////////////////////

impl fmt::Display for IntPolynomial { 
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut s = String::new();

        for (idx,c) in self.coefficients.iter().enumerate() {
            if idx == 0 {
                s += &format!("{}", c);
            } else if idx >= 1 {
                s += &format!("{} X", c);
            }
            if idx != 0 && idx >= 2 {
                s += &format!("^{}", (idx as i64) );
            }
            if (idx as i64) < (self.coefficients.len() as i64) - 1 {
                s += &format!(" + ");
            }
        }
        write!(f, "{}", s)
    }
} 

impl std::ops::Index<usize> for IntPolynomial {
    type Output = u32;

    fn index(&self, i: usize) -> &u32 {
        &self.coefficients[i]
    }
}

impl std::ops::IndexMut<usize> for IntPolynomial {
    fn index_mut( &mut self, i: usize) -> &mut u32 {
        &mut self.coefficients[i]
    }
}

impl std::ops::AddAssign for IntPolynomial {
    fn add_assign( &mut self, rhs: IntPolynomial ) {
        for (idx, e) in rhs.into_iter().enumerate() {
            if idx < self.coefficients.len() {
                self[idx] += e;
            } else {
                self.coefficients.push( e );
            }
        }

        self.trim();
    }
}

impl std::ops::Add for IntPolynomial {
    type Output = IntPolynomial;

    fn add ( self, rhs: IntPolynomial ) -> IntPolynomial {
        let maxdeg = if self.degree() > rhs.degree() { 
            self.degree() + 1
        } else {
            rhs.degree() + 1
        };

        let mut result = IntPolynomial::zeros( maxdeg );

        for i in 0 .. maxdeg {
            if i < self.degree() + 1 {
                result[i] += self[i];
            } 
            if i < rhs.degree() + 1 {
                result[i] += rhs[i];
            }
        }
        result.trim();
        result
    }
}

impl std::cmp::PartialEq for IntPolynomial {
    fn eq( &self, other: &IntPolynomial) -> bool {
        if self.degree() != other.degree() {
            return false;
        }

        for idx in 0 .. (self.degree()+1) {
            if self[idx] != other[idx] {
                return false;
            }
        }
        true
        
        //Can also do this in one line, but this is less efficient
        //self.into_iter().zip(other).all( |(x,y)| x == y )
    }
}

//////////////////////////////////////////////////////////////
/// Iterator Stuff
//////////////////////////////////////////////////////////////

impl IntoIterator for IntPolynomial {
    type Item = u32;
    type IntoIter = IntPolynomialIterator;

    fn into_iter(self) -> Self::IntoIter {
        IntPolynomialIterator {
            polynomial: self,
            index: 0,
        }
    }
}

impl<'a> IntoIterator for &'a IntPolynomial {
    type Item = u32;
    type IntoIter = RefIntPolynomialIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        RefIntPolynomialIterator {
            polynomial: self,
            index: 0,
        }
    }
}

pub struct IntPolynomialIterator {
    polynomial: IntPolynomial,
    index: usize,
}

pub struct RefIntPolynomialIterator<'a> {
    polynomial: &'a IntPolynomial,
    index: usize,
}

impl Iterator for IntPolynomialIterator {
    type Item = u32;

    fn next( &mut self ) -> Option<u32> {
        if self.index < self.polynomial.coefficients.len() {
            let result = self.polynomial.coefficients[self.index];
            self.index += 1;
            return Some(result);
        } else {
            return None;
        }
    }
}

impl<'a> Iterator for RefIntPolynomialIterator<'a> {
    type Item = u32;

    fn next( &mut self ) -> Option<u32> {
        if self.index < self.polynomial.coefficients.len() {
            let result = self.polynomial.coefficients[self.index];
            self.index += 1;
            return Some(result);
        } else {
            return None;
        }
    }
}

impl ExactSizeIterator for IntPolynomialIterator {
    fn len(&self) -> usize {
        self.polynomial.degree() + 1
    }
}

impl<'a> ExactSizeIterator for RefIntPolynomialIterator<'a> {
    fn len(&self) -> usize {
        self.polynomial.degree() + 1
    }
}

//////////////////////////////////////////////////////////////
/// Tests
//////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;
    
    #[test]
    fn poly_add() {
        let mut p = IntPolynomial::from_vec( &vec![1, 2, 3, 4] );
        let q = IntPolynomial::from_vec( &vec![4, 3, 2, 1, 2] );
        let t1 = IntPolynomial::from_vec( &vec![5, 5, 5, 5, 2] );
        let t2 = IntPolynomial::from_vec( &vec![9, 8, 7, 6, 4] );

        p += q.clone();
        assert!( p == t1 );

        p = p + q;
        assert!( p == t2 );
    }

    #[test]
    fn scalar_ops() {
        let mut p = IntPolynomial::from_vec( &vec![0, 1, 2, 3] );
        let t1 = IntPolynomial::from_vec( &vec![1, 2, 3] );
        let t2 = IntPolynomial::from_vec( &vec![2, 4, 6] );
        let t3 = IntPolynomial::from_vec( &vec![0, 0, 2, 4, 6] );

        assert!( p.xn_divide( 1 ).is_none() );
        assert!( p == t1 );

        p.scalar_multiply( 2 );
        assert!( p == t2 );

        p.xn_multiply( 2 );
        assert!( p == t3 );
    }

    #[test]
    fn gradeschool_mul() {
        let p = IntPolynomial::from_vec( &vec![1, 2, 3] );
        let q = IntPolynomial::from_vec( &vec![2, 3] );
        let t = IntPolynomial::from_vec( &vec![2, 7, 12, 9] );

        let r = p.gradeschool_mul( &q );

        assert!( r == t );
    }

    #[test]
    fn kronecker_sub_mul_small() {
        let p = IntPolynomial::from_vec( &vec![1, 2, 3] );
        let q = IntPolynomial::from_vec( &vec![2, 3] );
        let t = IntPolynomial::from_vec( &vec![2, 7, 12, 9] );
        
        let r = p.kronecker_mul( &q ); 

        assert!( r == t);
    }

    #[test]
    fn kronecker_rug() {
        let p = IntPolynomial::from_vec( &vec![1, 2, 3, 4, 5, 6] );
        let q = IntPolynomial::from_vec( &vec![2, 3, 1234] );
        let t = IntPolynomial::from_vec( &vec![2, 7, 1246, 2485, 3724, 4963, 6188, 7404] );
        
        let r = p.kronecker_mul( &q ); 

        println!("{}", r);

        assert!( r == t);
    }

    #[test]
    fn horner_eval() {
        let p = IntPolynomial::from_vec( &vec![1, 2, 3, 4, 5, 6] );
        let p1 = p.horner_eval( 1 );
        let p2 = p.horner_eval( 2 );
        let p3 = p.horner_eval( 10 );

        assert!( p1 == 21 );
        assert!( p2 == 120 );
        assert!( p3 == 123456 );
    }

    //////////////////////////////////////////////////////////////
    /// Benchmarks 
    //////////////////////////////////////////////////////////////

    #[bench]
    fn bench_gradeschool( b: &mut Bencher ) {
        let p = IntPolynomial::from_vec( &vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] );
        let q = IntPolynomial::from_vec( &vec![2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 0, 10, 11, 12] );

        b.iter( || {
            p.gradeschool_mul( &q ) 
        });
    }

    #[bench]
    fn bench_kronecker( b: &mut Bencher ) {
        let p = IntPolynomial::from_vec( &vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] );
        let q = IntPolynomial::from_vec( &vec![2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 0, 10, 11, 12] );

        b.iter( || {
            p.kronecker_mul( &q ) 
        });
    }
}
