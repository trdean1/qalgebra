use super::*;

pub enum KPError {
    UnequalDegree,
    NotPowerTwo,
}

pub struct KaratsubaPlan<T: Numeral + Neg> {
    a: Polynomial<T>,
    b: Polynomial<T>,
    layers: Vec<KaratsubaLayer<T>>,
}

pub struct KaratsubaLayer<T: Numeral + Neg> {
    chunks: Vec<KaratsubaChunk<T>>,
}

impl<T> KaratsubaLayer<T> where T: Numeral + Neg {
    /// Create a polynomial with an empty coefficient vector that has
    /// space allocated for n coefficients
    pub fn with_capacity( num_chunks: usize, chunk_size: usize ) -> KaratsubaLayer<T> {
        let chunks = Vec::<KaratsubaChunk<T>>::with_capacity( num_chunks );
        let mut layer = KaratsubaLayer { chunks };

        for i in 0 .. num_chunks {
            layer.chunks.push( KaratsubaChunk::<T>::with_capacity( chunk_size ) );
        }

        layer
    }
}

pub struct KaratsubaChunk<T: Numeral + Neg> {
    upper: Polynomial<T>,
    lower: Polynomial<T>,
    sum: Polynomial<T>,
}

impl<T> KaratsubaChunk<T> where T: Numeral + Neg {
    /// Create a polynomial with an empty coefficient vector that has
    /// space allocated for n coefficients
    pub fn with_capacity( n: usize ) -> KaratsubaChunk<T> {
        KaratsubaChunk {
            upper: Polynomial::<T>::with_capacity( n ),
            lower: Polynomial::<T>::with_capacity( n ),
            sum:   Polynomial::<T>::with_capacity( n ),
        }
    }
}


impl<T> KaratsubaPlan<T> where T: Numeral + Neg {
    pub fn new_plan( a: Polynomial<T>, b: Polynomial<T> ) -> Result<KaratsubaPlan<T>, KPError> {
        //For now, start with the simple case where a and b have the same degree which
        //is one minus a power of two.
        if a.degree() != b.degree() {
            return Err(KPError::UnequalDegree);
        }
        
        let n = a.degree() + 1;
        let is_pow_two = ( n & (n - 1) ) == 0;
        if !is_pow_two {
            return Err(KPError::NotPowerTwo);
        }

        let num_layers = fastlog2( n as u32 ) as usize; 

        let mut layers = Vec::with_capacity( num_layers );

        for i in 0 .. num_layers {
            let num_chunks = 3usize.pow(i as u32 + 1);
            let chunk_size = n >> (i+1);
            layers.push( KaratsubaLayer::<T>::with_capacity( num_chunks, chunk_size ) );
        }

        Ok( KaratsubaPlan{ a, b, layers } )
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
