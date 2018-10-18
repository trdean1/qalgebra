#![feature(test)]
extern crate test;

pub mod quaternion;
use quaternion::Quaternion;
use std::fmt;

#[derive(Clone)]
pub struct IntPolynomial {
    coefficients: Vec<i32>
}

impl IntPolynomial {
    pub fn from_vec(v: &Vec<i32>) -> IntPolynomial {
        IntPolynomial { 
            coefficients : v.clone()
        }
    }

    pub fn with_capacity( n: usize ) -> IntPolynomial {
        let v = Vec::with_capacity( n );
        IntPolynomial {
            coefficients : v
        }
    }

    pub fn zeros( n: usize ) -> IntPolynomial {
        let v = vec![0; n];
        IntPolynomial {
            coefficients : v
        }
    }

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

    //Multiply all coefficients by c
    pub fn scalar_multiply( &mut self, c: i32 ) {
        for i in 0..self.coefficients.len(){
            self.coefficients[i] *= c;
        }
    }

    //Multiply by x^n...shift coefficeints up n slots
    pub fn xn_multiply( &mut self, n: usize ) {
        for _ in 0..n {
            self.coefficients.insert( 0, 0 );
        }
    }

    //Divide by x^n...shift down n slots and return remainder if not zero
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

    pub fn gradeschool_mul( self, rhs: IntPolynomial ) -> IntPolynomial {
        let ldegree = self.degree();
        let rdegree = rhs.degree();
        let outdegree = ldegree + rdegree;
        let ref small;
        let ref large;

        if ldegree < rdegree {
            small = &self;
            large = &rhs;
        } else {
            small = &rhs;
            large = &self;
        }

        let mut result = IntPolynomial::zeros( outdegree );
        for (i, c) in small.into_iter().enumerate() {
            //XXX Not sure if rust is smart enough to save memory of temp each loop?
            let mut tmp = large.to_owned();
            tmp.scalar_multiply( c );
            tmp.xn_multiply( i );
            result += tmp;
        }

        result
    }
}

impl fmt::Display for IntPolynomial { 
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut s = String::new();

        for (idx,c) in self.coefficients.iter().enumerate() {
            if idx == 0 {
                s += &format!("{}", c);
            } else if idx >= 1 {
                s += &format!("{} X", c.abs());
            }
            if idx != 0 && idx >= 2 {
                s += &format!("^{}", (idx as i64) );
            }
            if (idx as i64) < (self.coefficients.len() as i64) - 1 {
                if self[idx+1] >= 0 {
                    s += &format!(" + ");
                } else {
                    s += &format!(" - ");
                }
            }
        }
        write!(f, "{}", s)
    }
} 

impl std::ops::Index<usize> for IntPolynomial {
    type Output = i32;

    fn index(&self, i: usize) -> &i32 {
        &self.coefficients[i]
    }
}

impl std::ops::IndexMut<usize> for IntPolynomial {
    fn index_mut( &mut self, i: usize) -> &mut i32 {
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
    type Item = i32;
    type IntoIter = IntPolynomialIterator;

    fn into_iter(self) -> Self::IntoIter {
        IntPolynomialIterator {
            polynomial: self,
            index: 0,
        }
    }
}

impl<'a> IntoIterator for &'a IntPolynomial {
    type Item = i32;
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
    type Item = i32;

    fn next( &mut self ) -> Option<i32> {
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
    type Item = i32;

    fn next( &mut self ) -> Option<i32> {
        if self.index < self.polynomial.coefficients.len() {
            let result = self.polynomial.coefficients[self.index];
            self.index += 1;
            return Some(result);
        } else {
            return None;
        }
    }
}

//////////////////////////////////////////////////////////////
/// Tests
//////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

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

        let r = p.gradeschool_mul( q );

        assert!( r == t );
    }
}
