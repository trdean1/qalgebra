extern crate qalgebra;
extern crate rand;

use rand::prelude::*;

use qalgebra::Polynomial;
use qalgebra::karatsuba::KaratsubaPlan;

#[test]
fn degree4_mul() {
    let mut rng = thread_rng();

    let mut p1 = Polynomial::zeros( 4 );
    let mut p2 = Polynomial::zeros( 4 );

    for i in 0 .. 100 {
        println!("Test {}", i );
        for i in 0 .. 4 {
            p1[i] = rng.gen_range( -20, 20 );
            p2[i] = rng.gen_range( -20, 20 );
        }
        println!("p1 = {}", p1 );
        println!("p2 = {}", p2 );

        let plan_maybe = KaratsubaPlan::<i32>::new_plan( 4 );

        let mut ka = Polynomial::with_capacity( 6 );

        if let Ok(mut plan) = plan_maybe {
            assert!( plan.execute_plan( &mut ka, &p1, &p2 ).is_ok() );
            let gs = p1.gradeschool_mul( &p2 ); 

            println!("Gradeschool: {}", gs );
            println!("Karatsuba: {}", ka );
            assert!( ka == gs );
        } else {
            assert!(false);
        }
        println!("\n\n");
    }
}

#[test]
fn degree8_mul() {
    let mut rng = thread_rng();

    let mut p1 = Polynomial::zeros( 8 );
    let mut p2 = Polynomial::zeros( 8 );

    for i in 0 .. 100 {
        println!("Test {}", i );
        for i in 0 .. 8 {
            p1[i] = rng.gen_range( -20, 20 );
            p2[i] = rng.gen_range( -20, 20 );
        }
        println!("p1 = {}", p1 );
        println!("p2 = {}", p2 );

        let plan_maybe = KaratsubaPlan::<i32>::new_plan( 8 );

        let mut ka = Polynomial::with_capacity( 12 );

        if let Ok(mut plan) = plan_maybe {
            assert!( plan.execute_plan( &mut ka, &p1, &p2 ).is_ok() );
            let gs = p1.gradeschool_mul( &p2 ); 

            println!("Gradeschool: {}", gs );
            println!("Karatsuba: {}", ka );
            assert!( ka == gs );
        } else {
            assert!(false);
        }
        println!("\n\n");
    }
}
