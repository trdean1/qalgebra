# qalgebra
Algorithms for performing arithmetic on polynomials of quaternions

Start of what is hopefully a useful project.

I plan to use SIMD implementations to implement several ways of multiplying polynomials of quaternions 
with floating point coefficients (grade-school, Toom-Cook, FFT).

I will template the multiplication algorithms so they should be useful over any algebra, you just need to 
define scalar arithmatic and possibly a way to take an FFT (or at least compute roots of unity).  

So far I have implemented algorithms for scalar quaternion arithmetic using SSE instructions.
