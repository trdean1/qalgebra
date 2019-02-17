#![feature(specialization)]
#![feature(test)]
#![feature(asm)]
extern crate test;
extern crate num_traits;

//extern crate rug;
//use rug::{Assign,Integer};

pub mod karatsuba;
pub mod quaternion;
pub mod complex;
//use quaternion::Quaternion;
use std::fmt;

use num_traits::{Num,NumAssign,NumAssignOps};
use num_traits::sign::Signed;

pub trait Numeral: Copy + Clone + 
                   Num + NumAssign + NumAssignOps + 
                   std::fmt::Display { }

impl<T> Numeral for T where T: Copy + Clone + 
                               Num + NumAssign + NumAssignOps + 
                               std::fmt::Display { }

pub trait AlmostEq {
    fn almost_eq( &self, other: &Self) -> bool;
}

#[derive(Clone)]
pub struct Polynomial<T: Numeral> {
    coefficients: Vec<T>
}

impl<T> Polynomial<T> where T: Numeral {
    /// Create polynomial from vector.  v[0] is degree zero term
    pub fn from_vec(v: &Vec<T>) -> Self {
        Self {
            coefficients : v.clone()
        }
    }


    /// Create a polynomial with an empty coefficient vector that has
    /// space allocated for n coefficients
    pub fn with_capacity( n: usize ) -> Self {
        let v = Vec::with_capacity( n );
        Self {
            coefficients : v
        }
    }

    
    /// Create a polynomial with all n zero coefficients
    pub fn zeros( n: usize ) -> Self {
        let v = vec![T::zero(); n];
        Self {
            coefficients : v
        }
    }


    /// Return degree --- length of coefficient array minus one
    pub fn degree( &self ) -> usize {
        if self.coefficients.len() == 0 {
            return 0;
        }
        self.coefficients.len() - 1
    }

    /// A 'smart' attempt at a deep copy.  Will not allocate memory
    /// if self has enough memory to copy
    pub fn copy_from( &mut self, other: &Self ) {
        self.coefficients.reserve( other.degree() + 1 );
        self.coefficients.clear();

        for c in other {
            self.coefficients.push( c );
        }
    }

    /// Remove trailing zero terms
    /// XXX: Again assuming default is 0
    fn trim( &mut self ) {
        if self.coefficients.len() == 0 { return; }

        while self.coefficients[ self.degree() ] == T::zero() {
            self.coefficients.pop();
            if self.coefficients.len() == 0 {
                break;
            }

        }
    }

    //////////////////////////////////////////////////////////////
    // Arithmetic Stuff
    //////////////////////////////////////////////////////////////

    /// Multiply all coefficients by c
    pub fn scalar_multiply( &mut self, c: T ) {
        for i in 0..self.coefficients.len(){
            self.coefficients[i] *= c;
        }
    }

    /// Multiply by x^n...shift coefficeints up n slots
    pub fn xn_multiply( &mut self, n: usize ) {
        for _ in 0..n {
            self.coefficients.insert( 0, T::zero() );
        }
    }

    /// Divide by x^n...shift down n slots and return remainder if not zero
    pub fn xn_divide( &mut self, n: usize) -> Option<Self> {
        let mut remainder = vec![T::zero(); n];
        let mut has_rem = false;
        for i in 0..n {
            remainder[i] = self.coefficients.remove(0);
            if remainder[i] != T::zero() { has_rem = true; } //Save us iterating again
        }

        if has_rem {
            let mut p = Self::<T>::from_vec( &remainder );
            p.trim();

            return Some(p);
        }

        None
    }
    
    /// multiply self by rhs returning a new polynomial.  Uses gradeschool multiplication
    /// which is slow unless degree is really small
    pub fn gradeschool_mul( &self, rhs: &Self ) -> Self {
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

        let mut result = Self::<T>::zeros( outdegree );
        let mut tmp = Self::<T>::with_capacity( large.degree()+1 );
        for (i, c) in small.into_iter().enumerate() {
            if c != T::zero() {
                tmp.copy_from( large );
                tmp.scalar_multiply( c );
                tmp.xn_multiply( i );
                result += tmp.clone(); //XXX: Hopefully this doesn't do anything memory heavy?
            }
        }

        result
    }

    //pub fn karatsuba_recurse( &self, rhs Polynomial<T> ) -> Polynomial<T> {
        //If degree_self < base_case || degree_rhs < base_case: //Perhaps some logic for l+r?
        //  return gradeschool_mul
        //
        //m = floor(max_degree/2)
        //high_self, low_self = split at m
        //high_rhs, low_rhs = split at m
        //z0 = karatsuba_recurse( low_self, low_rhs )
        //z1 = karatsuba_recurse( high_self, high_rhs )
        //z2 = karatsuba_recurse( (low_self + high_self), (low_rhs + high_rhs) )
        //
        //return (z2 * x^(2m) + (z1 - z2 - z0) * x^m + z0 
    //}
    

}



//////////////////////////////////////////////////////////////
// Evaluate 
//////////////////////////////////////////////////////////////

trait Eval<T>
{
    fn eval( &self, T ) -> T;

    fn eval_zero( &self ) -> T;

    fn eval_infty( &self ) -> T;
}

impl<T> Eval<T> for Polynomial<T> where T: Numeral 
{
    default fn eval(&self, x: T) -> T { 
        let mut res = T::zero();

        if x == res { return self.eval_zero(); }

        for c in self {
            res *= x;
            res += c;
        }

        res       
    }

    /// Save time by just returning the zero coefficient if we want to evaluate at zero
    fn eval_zero( &self ) -> T {
        return self.coefficients[0];
    }

    /// Return largest coefficient
    fn eval_infty( &self ) -> T {
        return self[self.degree()];
    }
}

/// For floats, overload to use FMA intrinsic
impl Eval<f32> for Polynomial<f32>
{
    default fn eval(&self, x: f32) -> f32 { 
        let mut res = 0f32;

        if x == res { return self.eval_zero(); }

        for c in self {
            res = res.mul_add( x, c );
        }

        res       
    }
}

impl Eval<f64> for Polynomial<f64>
{
    default fn eval(&self, x: f64) -> f64 { 
        let mut res = 0f64;

        if x == res { return self.eval_zero(); }

        for c in self {
            res = res.mul_add( x, c );
        }

        res       
    }
}

//////////////////////////////////////////////////////////////
// fmt 
//////////////////////////////////////////////////////////////

impl<T> fmt::Display for Polynomial<T> where T: Numeral { 
    default fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
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

impl<T> fmt::Display for Polynomial<T> where T: Numeral + Signed + PartialOrd{ 
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut s = String::new();

        for (idx,c) in self.coefficients.iter().enumerate() {
            let mut v = T::zero();
            if *c < T::zero() {
                v -= *c;
            } else {
                v = *c;
            }
            if idx == 0 {
                s += &format!("{}", c);
            } else if idx >= 1 {
                s += &format!("{} X", v);
            }
            if idx != 0 && idx >= 2 {
                s += &format!("^{}", (idx as i64) );
            }
            if (idx as i64) < (self.coefficients.len() as i64) - 1 {
                if self.coefficients[idx+1] >= T::zero() {
                    s += &format!(" + ");
                } else {
                    s += &format!(" - ");
                }
            }
        }
        write!(f, "{}", s)
    }
} 

//////////////////////////////////////////////////////////////
// std::op Overloads
//////////////////////////////////////////////////////////////

impl<T> std::ops::Index<usize> for Polynomial<T> where T: Numeral {
    type Output = T;

    fn index(&self, i: usize) -> &T {
        &self.coefficients[i]
    }
}

impl<T> std::ops::IndexMut<usize> for Polynomial<T> where T: Numeral {
    fn index_mut( &mut self, i: usize) -> &mut T {
        &mut self.coefficients[i]
    }
}

impl<T> std::ops::AddAssign for Polynomial<T> where T: Numeral {
    fn add_assign( &mut self, rhs: Self ) {
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

impl<T> std::ops::Add for Polynomial<T> where T: Numeral {
    type Output = Self;

    fn add ( self, rhs: Self ) -> Self::Output {
        let maxdeg = if self.degree() > rhs.degree() { 
            self.degree() + 1
        } else {
            rhs.degree() + 1
        };

        let mut result = Self::<T>::zeros( maxdeg );

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

impl<T> std::ops::SubAssign for Polynomial<T> where T: Numeral {
    fn sub_assign( &mut self, rhs: Self ) {
        for (idx, e) in rhs.into_iter().enumerate() {
            if idx < self.coefficients.len() {
                self[idx] -= e;
            } else {
                self.coefficients.push( e );
            }
        }

        self.trim();
    }
}

impl<T> std::ops::Sub for Polynomial<T> where T: Numeral {
    type Output = Self;

    fn sub ( self, rhs: Self ) -> Self::Output {
        let maxdeg = self.degree().max( rhs.degree() ) + 1;

        let mut result = Self::zeros( maxdeg );

        for i in 0 .. maxdeg {
            if i < self.degree() + 1 {
                result[i] += self[i];
            } 
            if i < rhs.degree() + 1 {
                result[i] -= rhs[i];
            }
        }
        result.trim();
        result
    }
}

impl<T> std::cmp::PartialEq for Polynomial<T> where T: Numeral {
    fn eq( &self, other: &Self) -> bool {
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
// Include almost_eq function for types that have AlmostEq trait
//////////////////////////////////////////////////////////////

impl<T> AlmostEq for Polynomial<T> where T: Numeral + AlmostEq {
    fn almost_eq( &self, other: &Self ) -> bool {
         if self.degree() != other.degree() {
            return false;
        }

        for idx in 0 .. (self.degree()+1) {
            if !self[idx].almost_eq( &other[idx] ) {
                return false;
            }
        }
        true       
    }
}

impl AlmostEq for Polynomial<f32>  {
    fn almost_eq( &self, other: &Self ) -> bool {
         if self.degree() != other.degree() {
            return false;
        }

        for idx in 0 .. (self.degree()+1) {
            if ( self[idx] - other[idx] ).abs() > 1e-9 {
                return false;
            }
        }
        true       
    }
}

impl AlmostEq for Polynomial<f64>  {
    fn almost_eq( &self, other: &Self ) -> bool {
         if self.degree() != other.degree() {
            return false;
        }

        for idx in 0 .. (self.degree()+1) {
            if ( self[idx] - other[idx] ).abs() > 1e-9 {
                return false;
            }
        }
        true       
    }
}

//////////////////////////////////////////////////////////////
// std::op Overloads for references
//////////////////////////////////////////////////////////////
impl<'a, T> std::ops::AddAssign<&'a Polynomial<T>> for Polynomial<T> where T: Numeral {
    fn add_assign( &mut self, rhs: &'a Polynomial<T> ) {
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

impl<'a, T> std::ops::SubAssign<&'a Polynomial<T>> for Polynomial<T> where T: Numeral {
    fn sub_assign( &mut self, rhs: &'a Polynomial<T> ) {
        for (idx, e) in rhs.into_iter().enumerate() {
            if idx < self.coefficients.len() {
                self[idx] -= e;
            } else {
                self.coefficients.push( e );
            }
        }

        self.trim();
    }
}

//////////////////////////////////////////////////////////////
// Iterator Stuff
//////////////////////////////////////////////////////////////
impl<T> IntoIterator for Polynomial<T> where T: Numeral {
    type Item = T;
    type IntoIter = PolynomialIterator<T>;

    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter {
            polynomial: self,
            index: 0,
        }
    }
}

impl<'a, T> IntoIterator for &'a Polynomial<T> where T: 'a + Numeral {
    type Item = T;
    type IntoIter = RefPolynomialIterator<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter {
            polynomial: self,
            index: 0,
        }
    }
}

pub struct PolynomialIterator<T: Numeral>  {
    polynomial: Polynomial<T>,
    index: usize,
}

pub struct RefPolynomialIterator<'a, T: 'a + Numeral> {
    polynomial: &'a Polynomial<T>,
    index: usize,
}

impl<T> Iterator for PolynomialIterator<T> where T: Numeral {
    type Item = T;

    fn next( &mut self ) -> Option<T> {
        if self.index < self.polynomial.coefficients.len() {
            let result = self.polynomial.coefficients[self.index];
            self.index += 1;
            return Some(result);
        } else {
            return None;
        }
    }
}

impl<'a, T> Iterator for RefPolynomialIterator<'a, T> where T: 'a + Numeral {
    type Item = T;

    fn next( &mut self ) -> Option<T> {
        if self.index < self.polynomial.coefficients.len() {
            let result = self.polynomial.coefficients[self.index];
            self.index += 1;
            return Some(result);
        } else {
            return None;
        }
    }
}

impl<T> ExactSizeIterator for PolynomialIterator<T> where T: Numeral {
    fn len(&self) -> usize {
        self.polynomial.degree() + 1
    }
}

impl<'a, T> ExactSizeIterator for RefPolynomialIterator<'a, T> where T: 'a + Numeral {
    fn len(&self) -> usize {
        self.polynomial.degree() + 1
    }
}
//////////////////////////////////////////////////////////////
// Tests
//////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn poly_add() {
        let mut p = Polynomial::<i32>::from_vec( &vec![1, 2, 3, 4] );
        let q = Polynomial::from_vec( &vec![4i32, 3, 2, 1, 2] );
        let t1 = Polynomial::from_vec( &vec![5i32, 5, 5, 5, 2] );
        let t2 = Polynomial::from_vec( &vec![9i32, 8, 7, 6, 4] );

        p += q.clone();
        assert!( p == t1 );

        p = p + q;
        assert!( p == t2 );
    }

    #[test]
    fn poly_sub() {
        let mut p = Polynomial::<i32>::from_vec( &vec![1, 2, 3, 4] );
        let q = Polynomial::from_vec( &vec![4i32, 3, 2, -1, -2] );
        let t1 = Polynomial::from_vec( &vec![-3i32, -1, 1, 5, -2] );
        let t2 = Polynomial::from_vec( &vec![-7i32, -4, -1, 6] );

        p -= q.clone();

        assert!( p == t1 );

        println!("P \t= {}", p );
        println!("Q \t= {}", q );
        p = p - q;

        println!("P - Q \t= {}", p );
        assert!( p == t2 );
    }

    #[test]
    fn scalar_ops() {
        let mut p = Polynomial::from_vec( &vec![0i32, 1, 2, 3] );
        let t1 = Polynomial::from_vec( &vec![1i32, 2, 3] );
        let t2 = Polynomial::from_vec( &vec![2i32, 4, 6] );
        let t3 = Polynomial::from_vec( &vec![0i32, 0, 2, 4, 6] );

        assert!( p.xn_divide( 1 ).is_none() );
        assert!( p == t1 );

        p.scalar_multiply( 2 );
        assert!( p == t2 );

        p.xn_multiply( 2 );
        assert!( p == t3 );
    }

    #[test]
    fn eval() {
        let p = Polynomial::from_vec( &vec![1i32, 2, 3, 4, 5, 6] );
        let p1 = p.eval( 1 );
        let p2 = p.eval( 2 );
        let p3 = p.eval( 10 );

        assert!( p1 == 21 );
        assert!( p2 == 120 );
        assert!( p3 == 123456 );
    } 

    #[test]
    fn eval_f32() {
        let p = Polynomial::from_vec( &vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0] );
        let p1 = p.eval( 1.0 );
        let p2 = p.eval( 2.0 );
        let p3 = p.eval( 10.0 );

        assert!( p1 == 21.0 );
        assert!( p2 == 120.0 );
        assert!( p3 == 123456.0 );
    } 

    #[test]
    fn gradeschool_mul() {
        let p = Polynomial::from_vec( &vec![1i32, 2, 3] );
        let q = Polynomial::from_vec( &vec![2i32, 3] );
        let t = Polynomial::from_vec( &vec![2i32, 7, 12, 9] );

        let r = p.gradeschool_mul( &q );

        assert!( r == t );
    }

    #[test]
    fn gradschool_zero_index() {
        let p1 = Polynomial::from_vec( &vec![14,0,-8,1] );
        let p2 = Polynomial::from_vec( &vec![0,-18,2,-18] );
        let t = Polynomial::from_vec( &vec![0, -252, 28, -108, -34, 146, -18] );

        let r = p1.gradeschool_mul( &p2 );

        assert!( r == t );
    }
}
