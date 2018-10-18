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
                s += &format!("^{}", (idx as i64) - 1);
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

pub struct IntPolynomialIterator {
    polynomial: IntPolynomial,
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
}
