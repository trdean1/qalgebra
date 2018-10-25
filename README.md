# qalgebra
Algorithms for performing arithmetic on special polynomials of quaternions.
These polynomials have quaternion coefficients but are variables of complex
roots of unity (i.e. z=e^{-i \theta}, and the j, and k components are both
zero).

Start of what is hopefully a useful project.

I plan to use SIMD implementations to implement several ways of multiplying polynomials of quaternions 
with floating point coefficients (grade-school, Toom-Cook, FFT).

I will template the multiplication algorithms so they should be useful over any algebra, you just need to 
define scalar arithmatic and possibly a way to take an FFT (or at least compute roots of unity).  
