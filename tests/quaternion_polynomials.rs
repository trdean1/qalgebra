/*
extern crate qalgebra;
extern crate rand;

use rand::prelude::*;

use qalgebra::Polynomial;
use qalgebra::quaternion::Quaternion;
use qalgebra::karatsuba::KaratsubaPlan;
use qalgebra::AlmostEq;

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

#[test]
fn mul_basic() {
    let v1 = vec![ Quaternion::from_vals(  1.0, 1.0, 1.0, 1.0 ),
                   Quaternion::from_vals( -1.0, 1.0, 1.0, 1.0 ) ];
    let v2 = vec![ Quaternion::from_vals( -1.0, 0.0, 0.0, 0.0 ),
                   Quaternion::from_vals(  1.0, 0.0, 0.0, 0.0 ) ];
    let v3 = vec![ Quaternion::from_vals( -1.0, -1.0, -1.0, -1.0 ),
                   Quaternion::from_vals(  2.0, 0.0, 0.0, 0.0 ),
                   Quaternion::from_vals( -1.0, 1.0, 1.0, 1.0 ) ];

    let p1 = Polynomial::from_vec( &v1 );
    let p2 = Polynomial::from_vec( &v2 );
    let p3 = Polynomial::from_vec( &v3 );

    let p4 = p1.gradeschool_mul( &p2 );

    println!("{}", p4);
    assert!( p4.almost_eq( &p3 ) );
}

#[test]
fn degree4_mul() {
    let mut rng = thread_rng();

    let mut p1 = Polynomial::<Quaternion>::zeros( 4 );
    let mut p2 = Polynomial::<Quaternion>::zeros( 4 );

    let mut tmp1 = vec![0.0f32, 0.0f32, 0.0f32, 0.0f32];
    let mut tmp2 = vec![0.0f32, 0.0f32, 0.0f32, 0.0f32];

    for i in 0 .. 20 {
        println!("Test {}", i );

        for j in 0 .. 4 {
            for k in 0 .. 4 {
                tmp1[k] = rng.gen_range( -1.0f32, 1.0f32 );
                tmp2[k] = rng.gen_range( -1.0f32, 1.0f32 );
            }
            let qtmp1 = Quaternion::from_vec( &tmp1 );
            let qtmp2 = Quaternion::from_vec( &tmp2 );
            p1[j] = qtmp1;
            p2[j] = qtmp2;
        }

        println!("p1 = {}", p1 );
        println!("p2 = {}", p2 );

        let plan_maybe = KaratsubaPlan::<Quaternion>::new_plan( 4 );

        let mut ka = Polynomial::<Quaternion>::with_capacity( 6 );

        if let Ok(mut plan) = plan_maybe {
            assert!( plan.execute_plan( &mut ka, &p1, &p2 ).is_ok() );
            let gs = p1.gradeschool_mul( &p2 ); 

            println!("Gradeschool: {}", gs );
            println!("Karatsuba: {}", ka );
            println!("Diff: {}", gs.clone() - ka.clone() );
            assert!( ka.almost_eq( &gs ) );
        } else {
            assert!(false);
        }
        println!("\n\n");
    }
}
*/
