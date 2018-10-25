#![feature(specialization)]
#![feature(test)]
extern crate test;

//extern crate rug;
//use rug::{Assign,Integer};

pub mod karatsuba;
pub mod quaternion;
//use quaternion::Quaternion;
use std::fmt;
use std::ops::*;

pub trait Numeral: Copy + Clone + Default + PartialEq +  
                   Add + AddAssign + Mul + MulAssign + SubAssign +
                   Mul<Output=Self> +
                   std::fmt::Display { }
impl<T> Numeral for T where T: Copy + Clone + Default + PartialEq + 
                               Add + AddAssign + Mul + MulAssign + SubAssign +
                               Mul<Output=T> + 
                               std::fmt::Display { }

#[derive(Clone)]
pub struct Polynomial<T: Numeral> {
    coefficients: Vec<T>
}

impl<T> Polynomial<T> where T: Numeral {
    /// Create polynomial from vector.  v[0] is degree zero term
    pub fn from_vec(v: &Vec<T>) -> Polynomial<T> {
        Polynomial { 
            coefficients : v.clone()
        }
    }


    /// Create a polynomial with an empty coefficient vector that has
    /// space allocated for n coefficients
    pub fn with_capacity( n: usize ) -> Polynomial<T> {
        let v = Vec::with_capacity( n );
        Polynomial {
            coefficients : v
        }
    }

    
    /// Create a polynomial with all n zero coefficients
    /// XXX: This assumes T::default is the additive identity.  Is this smart?
    pub fn zeros( n: usize ) -> Polynomial<T> {
        let v = vec![T::default(); n];
        Polynomial {
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
    pub fn copy_from( &mut self, other: &Polynomial<T> ) {
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

        while self.coefficients[ self.degree() ] == T::default() {
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
            self.coefficients.insert( 0, T::default() );
        }
    }

    /// Divide by x^n...shift down n slots and return remainder if not zero
    pub fn xn_divide( &mut self, n: usize) -> Option<Polynomial<T>> {
        let mut remainder = vec![T::default(); n];
        let mut has_rem = false;
        for i in 0..n {
            remainder[i] = self.coefficients.remove(0);
            if remainder[i] != T::default() { has_rem = true; } //Save us iterating again
        }

        if has_rem {
            let mut p = Polynomial::<T>::from_vec( &remainder );
            p.trim();

            return Some(p);
        }

        None
    }
    
    /// multiply self by rhs returning a new polynomial.  Uses gradeschool multiplication
    /// which is slow unless degree is really small
    pub fn gradeschool_mul( &self, rhs: &Polynomial<T> ) -> Polynomial<T> {
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

        let mut result = Polynomial::<T>::zeros( outdegree );
        let mut tmp = Polynomial::<T>::with_capacity( large.degree()+1 );
        for (i, c) in small.into_iter().enumerate() {
            if c != T::default() {
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
        let mut res = T::default();

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

impl<T> fmt::Display for Polynomial<T> where T: Numeral{ 
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

impl<T> fmt::Display for Polynomial<T> where T: Numeral + Neg + PartialOrd{ 
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut s = String::new();

        for (idx,c) in self.coefficients.iter().enumerate() {
            let mut v = T::default();
            if *c < T::default() {
                v -= *c;
            } else {
                v = *c;
            }
            if idx == 0 {
                s += &format!("{}", v);
            } else if idx >= 1 {
                s += &format!("{} X", v);
            }
            if idx != 0 && idx >= 2 {
                s += &format!("^{}", (idx as i64) );
            }
            if (idx as i64) < (self.coefficients.len() as i64) - 1 {
                if self.coefficients[idx+1] >= T::default() {
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
    fn add_assign( &mut self, rhs: Polynomial<T> ) {
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
    type Output = Polynomial<T>;

    fn add ( self, rhs: Polynomial<T> ) -> Polynomial<T> {
        let maxdeg = if self.degree() > rhs.degree() { 
            self.degree() + 1
        } else {
            rhs.degree() + 1
        };

        let mut result = Polynomial::<T>::zeros( maxdeg );

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

impl<T> std::ops::SubAssign for Polynomial<T> where T: Numeral + Neg {
    fn sub_assign( &mut self, rhs: Polynomial<T> ) {
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

impl<T> std::ops::Sub for Polynomial<T> where T: Numeral + Neg{
    type Output = Polynomial<T>;

    fn sub ( self, rhs: Polynomial<T> ) -> Polynomial<T> {
        let maxdeg = if self.degree() > rhs.degree() { 
            self.degree() + 1
        } else {
            rhs.degree() + 1
        };

        let mut result = Polynomial::<T>::zeros( maxdeg );

        for i in 0 .. maxdeg {
            if i < self.degree() + 1 {
                result[i] -= self[i];
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
    fn eq( &self, other: &Polynomial<T>) -> bool {
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

impl<'a, T> std::ops::SubAssign<&'a Polynomial<T>> for Polynomial<T> where T: Numeral + Neg {
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
        PolynomialIterator {
            polynomial: self,
            index: 0,
        }
    }
}

impl<'a, T> IntoIterator for &'a Polynomial<T> where T: 'a + Numeral {
    type Item = T;
    type IntoIter = RefPolynomialIterator<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        RefPolynomialIterator {
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
