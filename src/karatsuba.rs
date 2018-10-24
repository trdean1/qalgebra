use super::*;

pub enum KaratsubaError {
    DegreeMismatch,
    NotPowerTwo,
}

pub struct KaratsubaPlan<T: Numeral + Neg> {
    n: usize,
    a: Polynomial<T>,
    b: Polynomial<T>,
    layers: Vec<KaratsubaLayer<T>>,
}

pub struct KaratsubaLayer<T: Numeral + Neg> {
    chunks_a: Vec<KaratsubaChunk<T>>,
    chunks_b: Vec<KaratsubaChunk<T>>,
    results: Vec<KaratsubaChunk<T>>,
}

pub struct KaratsubaChunk<T: Numeral + Neg> {
    pub upper: Polynomial<T>,
    pub lower: Polynomial<T>,
    pub sum: Polynomial<T>,
    n: usize,
}

impl<T> KaratsubaLayer<T> where T: Numeral + Neg {
    /// Create a polynomial with an empty coefficient vector that has
    /// space allocated for n coefficients
    pub fn with_capacity( num_chunks: usize, chunk_size: usize ) -> KaratsubaLayer<T> {
        let chunks_a = Vec::<KaratsubaChunk<T>>::with_capacity( num_chunks );
        let chunks_b = Vec::<KaratsubaChunk<T>>::with_capacity( num_chunks );
        let results  = Vec::<KaratsubaChunk<T>>::with_capacity( num_chunks );
        let mut layer = KaratsubaLayer { chunks_a, chunks_b, results };

        for _ in 0 .. num_chunks {
            layer.chunks_a.push( KaratsubaChunk::<T>::with_capacity( chunk_size ) );
            layer.chunks_b.push( KaratsubaChunk::<T>::with_capacity( chunk_size ) );
            layer.results .push( KaratsubaChunk::<T>::with_capacity( 3 * chunk_size / 2 ) );
        }

        layer
    }
}

impl<T> KaratsubaChunk<T> where T: Numeral + Neg {
    /// Create a polynomial with an empty coefficient vector that has
    /// space allocated for n coefficients
    pub fn with_capacity( n: usize ) -> KaratsubaChunk<T> {
        KaratsubaChunk {
            upper: Polynomial::<T>::zeros( n ),
            lower: Polynomial::<T>::zeros( n ),
            sum:   Polynomial::<T>::zeros( n ),
            n: n,
        }
    }

    pub fn len( &self ) -> usize {
        self.n
    }
}


impl<T> KaratsubaPlan<T> where T: Numeral + Neg {
    pub fn new_plan( n: usize ) -> Result<KaratsubaPlan<T>, KaratsubaError> {
        //For now, start with the simple case where a and b have the same degree which
        //is one minus a power of two.
        
        //let n = a.degree() + 1;
        //let is_pow_two = ( n & (n - 1) ) == 0;
        if !n.is_power_of_two() {
            return Err(KaratsubaError::NotPowerTwo);
        }

        let num_layers = fastlog2( n as u32 ) as usize + 1; 

        let mut layers = Vec::with_capacity( num_layers );

        for i in 0 .. num_layers {
            let num_chunks = 3usize.pow(i as u32);
            let chunk_size = n >> (i);
            layers.push( KaratsubaLayer::<T>::with_capacity( num_chunks, chunk_size ) );
        }

        let a = Polynomial::<T>::with_capacity( n );
        let b = Polynomial::<T>::with_capacity( n );

        Ok( KaratsubaPlan{ n, a, b, layers } )
    }

    pub fn execute_plan( &mut self, a: &Polynomial<T>, b: &Polynomial<T> ) 
        -> Result<Polynomial<T>, KaratsubaError> {
        //For now, start with the simple case where a and b have the same degree which
        //is one minus a power of two.
        if (self.n != a.degree() + 1) || (self.n != b.degree() + 1) {
            return Err( KaratsubaError::DegreeMismatch );
        }

        self.a.copy_from( a );
        self.b.copy_from( b );

        //Forward pass.  We're turning the recursive function into an iterative one.
        //The forward pass fills out the data structure with either the values of the a and b
        //coefficients or the sum of the upper and lower half of each
        for i in 0..self.layers.len() {
            if i == 0 {
                KaratsubaPlan::split_and_sum_to( a, &mut self.layers[0].chunks_a[0] );
                KaratsubaPlan::split_and_sum_to( b, &mut self.layers[0].chunks_b[0] );
            }  else {
                let num_chunks_last = self.layers[i-1].chunks_a.len();

                //Not sure this is the most readable way to do this.  I have to split to form
                //two refs from the vector 'layers'.  This requires a deref from a raw pointer
                //because the borrow checker can't tell whether or not slices overlap.  The other
                //option is to make an unsafe block to pull out two elements rather than using the
                //std call to split a slice.  current[0] is the current layer (the one at
                //layers[i]). last[0] is one layer up.  The slice 'last' only has one element.
                let (last, current) = self.layers.split_at_mut( i );

                for j in 0 .. num_chunks_last {
                    KaratsubaPlan::split_and_sum_to( &last[0].chunks_a[j].upper, &mut current[0].chunks_a[3*j  ] );
                    KaratsubaPlan::split_and_sum_to( &last[0].chunks_a[j].lower, &mut current[0].chunks_a[3*j+1] );
                    KaratsubaPlan::split_and_sum_to( &last[0].chunks_a[j].sum  , &mut current[0].chunks_a[3*j+2] );

                    KaratsubaPlan::split_and_sum_to( &last[0].chunks_b[j].upper, &mut current[0].chunks_b[3*j  ] );
                    KaratsubaPlan::split_and_sum_to( &last[0].chunks_b[j].lower ,&mut current[0].chunks_b[3*j+1] );
                    KaratsubaPlan::split_and_sum_to( &last[0].chunks_b[j].sum  , &mut current[0].chunks_b[3*j+2] );
                }
            } 
        }

        //Reverse pass. Start at the bottom and recombine. Each chunk becomes
        // (upper) * x^n + (sum - lower - upper) * x^n/2 + (lower)
        for i in (1 .. self.layers.len()).rev() {
            let num_layers = self.layers.len();
            let num_chunks_parent = self.layers[i-1].chunks_a.len();
            //Same strategy as in forward pass.  Parent holds the result of the sums in current.
            let (parent, current) = self.layers.split_at_mut( i );

            for j in 0 .. num_chunks_parent {
                if i == num_layers - 1 {
                    KaratsubaPlan::combine_chunk_base( &current[0].chunks_a[3*j], 
                                                       &current[0].chunks_b[3*j], 
                                                       &mut parent[0].results[j].upper );
                    KaratsubaPlan::combine_chunk_base( &current[0].chunks_a[3*j+1], 
                                                       &current[0].chunks_b[3*j+1], 
                                                       &mut parent[0].results[j].lower );
                    KaratsubaPlan::combine_chunk_base( &current[0].chunks_a[3*j+2], 
                                                       &current[0].chunks_b[3*j+2], 
                                                       &mut parent[0].results[j].sum );
                } else {
                    KaratsubaPlan::combine_chunk_to( &current[0].results[3*j], &mut parent[0].results[j].upper );
                    KaratsubaPlan::combine_chunk_to( &current[0].results[3*j+1], &mut parent[0].results[j].lower );
                    KaratsubaPlan::combine_chunk_to( &current[0].results[3*j+2], &mut parent[0].results[j].sum );
                }
            }
        }

        Ok(Polynomial::<T>::with_capacity(1))
    }

    /// Copies out lower and upper half and computes sum.  
    fn split_and_sum_to( poly: &Polynomial<T>, chunk: &mut KaratsubaChunk<T> ) {
        let n = poly.degree() + 1;
        let n2 = n / 2;
        for idx in 0 .. n  {
            if idx < n2 {
                chunk.lower[idx] = poly[idx];
                chunk.sum[idx] = poly[idx];
            } else {
                chunk.upper[idx - n2] = poly[idx];
                chunk.sum[idx - n2] += poly[idx];
            }
        }
    }

    fn combine_chunk_to( chunk: &karatsuba::KaratsubaChunk<T>, result: &mut Polynomial<T> ) {
        let n = 2 * chunk.len();
        assert!(n != 2 );

        *result += &chunk.upper;
        result.xn_multiply( n / 2 );

        *result += &chunk.sum;
        *result -= &chunk.lower;
        *result -= &chunk.upper;
        result.xn_multiply( n / 2 );

        *result += &chunk.lower;
    }

    /// For now assume the base case is just n=1.  We will actually probably want
    /// the base case to be a bit higher than n=1 and so we will have to call gradeschool
    /// multiplication on each polynomial.  I'm not sure the best way to do this and be memory
    /// efficient.
    fn combine_chunk_base( chunk_a: &KaratsubaChunk<T>, 
                           chunk_b: &KaratsubaChunk<T>, 
                           result: &mut Polynomial<T> ) {
        assert!( chunk_a.len() == 1 );

        result[0] += chunk_a.upper[0] * chunk_b.upper[0];
        result.xn_multiply( 1 );

        result[0] += chunk_a.sum[0] * chunk_b.sum[0];
        result[0] -= chunk_a.lower[0] * chunk_b.lower[0];
        result[0] -= chunk_a.upper[0] * chunk_b.lower[0];
        result.xn_multiply( 1 );

        result[0] += chunk_a.lower[0] * chunk_b.lower[0];
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

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn init() {
        let plan = KaratsubaPlan::<i32>::new_plan( 16 );
    }

    #[test]
    fn split() {
        let polynomial = Polynomial::from_vec( &vec![4, 3, 2, 1] );
        let mut chunk = KaratsubaChunk::with_capacity( 2 );

        let u = Polynomial::from_vec( &vec![2,1] );
        let l = Polynomial::from_vec( &vec![4,3] );
        let s = Polynomial::from_vec( &vec![6,4] );

        KaratsubaPlan::split_and_sum_to( &polynomial, &mut chunk );

        assert!( chunk.upper == u );
        assert!( chunk.lower == l );
        assert!( chunk.sum   == s );
    }
}
