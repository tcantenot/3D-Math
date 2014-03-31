////////////////////////////////////////////////////////////////////////////////
// From : http://orbit.dtu.dk/files/57573287/onb_frisvad_jgt2012.pdf
//
// Finding an orthonormal basis from a unit 3D vector.
////////////////////////////////////////////////////////////////////////////////
#include <cmath>
#include <iostream>

#define SSE_INSTR

#ifdef SSE_INSTR
#include <xmmintrin.h>
#endif // SSE_INSTR

// Very simple 3d vector class
struct Vec3f
{
    Vec3f(float xx = 0.f, float yy = 0.f, float zz = 0.f):
        x(xx), y(yy), z(zz)
    {

    }

    Vec3f operator*(float n) const
    {
        return Vec3f(*this) *= n;
    }

    Vec3f & operator*=(float n)
    {
        x *= n;
        y *= n;
        z *= n;
        return *this;
    }

    Vec3f operator-(Vec3f const & v) const
    {
        return Vec3f(*this) -= v;
    }

    Vec3f & operator-=(Vec3f const & v)
    {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return *this;
    }

    friend std::ostream & operator<<(std::ostream & os, Vec3f const & v)
    {
        return os << "(" << v.x << ", " << v.y << ", " << v.z << ")";
    }


    float x;
    float y;
    float z;
};

#define NEAR_ONE 0.9999999f

// Dot product
float dot(Vec3f const & lhs, Vec3f const & rhs)
{
    return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

// Cross product
Vec3f cross(Vec3f const & lhs, Vec3f const & rhs)
{
    return Vec3f(lhs.y * rhs.z - lhs.z * rhs.y,
                 lhs.z * rhs.x - lhs.x * rhs.z,
                 lhs.x * rhs.y - lhs.y * rhs.x);
}



// Using the C++ standard library sqrt function
inline float inv_sqrt(float x)
{
    return static_cast<float>(1.0 / sqrt(x));
}

// John Carmack's fast inverse square root
// See http://en.wikipedia.org/wiki/Fast_inverse_square_root
inline float fast_inv_sqrt(float x)
{
    float xhalf = 0.5f * x;
    int i = *(int *) &x;
    i = 0x5f3759df - (i >> 1);
    x = *(float *) &i;
    x = x * (1.5f - xhalf * x * x); // (repeat to improve precision)
    x = x * (1.5f - xhalf * x * x); // (repeat to improve precision)
    return x;
}

#ifdef SSE_INSTR
// Streaming SIMD Extension (SSE) scalar rsqrt function
inline float SSE_rsqrt(float x)
{
    // The following compiles to movss , rsqrtss , movss
    _mm_store_ss(&x, _mm_rsqrt_ss(_mm_load_ss(&x)));
    return x;
}

// Include a Newton iteration with the SSE rsqrt to improve precision
inline float SSE_rsqrt_1N(float x)
{
    float y = SSE_rsqrt(x);
    return y * (1.5f - 0.5f * x * y * y);
}
#endif // SSE_INSTR


#define rsqrt fast_inv_sqrt

// Naive orthonormal basis generation
void naive(Vec3f const & n, Vec3f & b1, Vec3f & b2)
{
    // If n is near the x-axis, use the y-axis. Otherwise use the x-axis.
    b1 = (n.x > NEAR_ONE) ? Vec3f(0.f, 1.f, 0.f) : Vec3f(1.f, 0.f, 0.f);
    b1 -= n * dot(b1, n); // Make b1 orthogonal to n
    b1 *= rsqrt(dot(b1, b1)); // Normalize b1
    b2 = cross(n, b1); // Construct b2 using a cross product
}

// Hughes Moeller's orthonormal basis generation
void hughes_moeller(Vec3f const & n, Vec3f & b1, Vec3f & b2)
{
    // Choose a vector orthogonal to n as the direction of b2.
    if(std::abs(n.x) > std::abs(n.z)) b2 = Vec3f(-n.y, n.x, 0.f);
    else b2 = Vec3f(0.f, -n.z, n.y);
    b2 *= rsqrt(dot(b2, b2)); // Normalize b2
    b1 = cross(b2, n); // Construct b1 using a cross product
}

// Fastest way (according to the paper (link above))
// No sqrt and no normalization needed !!!
void frisvad(Vec3f const & n, Vec3f & b1, Vec3f & b2)
{
    if(n.z < -NEAR_ONE) // Handle the singularity
    {
        b1 = Vec3f(0.f, -1.f, 0.f);
        b2 = Vec3f(-1.f, 0.f, 0.f);
        return;
    }

    float const a = 1.f / (1.f + n.z);
    float const b = -n.x * n.y * a;
    b1 = Vec3f(1.f - n.x * n.x * a, b, -n.x);
    b2 = Vec3f(b, 1.f - n.y * n.y * a, -n.y);
}

int main()
{
    float n = 468854;
    std::cout << inv_sqrt(n) << std::endl;
    std::cout << fast_inv_sqrt(n) << std::endl;

    #ifdef SSE_INSTR
    std::cout << SSE_rsqrt(n) << std::endl;
    std::cout << SSE_rsqrt_1N(n) << std::endl;
    #endif // SSE_INSTR

    std::cout << std::endl;

    Vec3f v(1.f, 0.f, 0.f);

    {
        Vec3f b1, b2;
        naive(v, b1, b2);

        std::cout << "b1: " << b1 << std::endl;
        std::cout << "b2: " << b2 << std::endl;
    }

    std::cout << std::endl;

    {
        Vec3f b1, b2;
        hughes_moeller(v, b1, b2);

        std::cout << "b1: " << b1 << std::endl;
        std::cout << "b2: " << b2 << std::endl;
    }

    std::cout << std::endl;

    {
        Vec3f b1, b2;
        frisvad(v, b1, b2);

        std::cout << "b1: " << b1 << std::endl;
        std::cout << "b2: " << b2 << std::endl;
    }

    return 0;
}
