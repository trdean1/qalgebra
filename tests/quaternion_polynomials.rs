extern crate qalgebra;
extern crate rand;

//use rand::prelude::*;

use qalgebra::Polynomial;
use qalgebra::quaternion::Quaternion;
//use qalgebra::karatsuba::KaratsubaPlan;

#[test]
fn basic() {
    //let mut rng = thread_rng();

    let v1 = vec![ Quaternion::from_vals(  1.0, 1.0, 1.0, 1.0 ),
                   Quaternion::from_vals( -1.0, 1.0, 1.0, 1.0 ) ];
    let v2 = vec![ Quaternion::from_vals( -1.0, 1.0, 1.0, 1.0 ),
                   Quaternion::from_vals(  1.0, 1.0, 1.0, 1.0 ) ];
    let v3 = vec![ Quaternion::from_vals(  0.0, 2.0, 2.0, 2.0 ),
                   Quaternion::from_vals(  0.0, 2.0, 2.0, 2.0 ) ];

    let p1 = Polynomial::from_vec( &v1 );
    let p2 = Polynomial::from_vec( &v2 );
    let p3 = Polynomial::from_vec( &v3 );

    assert!( p3 == p1 + p2 );
}
