#include "interpolate.h"
#include <glm/geometric.hpp>

float triangleArea(const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2)
{
    glm::vec3 a = -v0 + v1;
    glm::vec3 b = -v0 + v2;
    return 0.5f * glm::length(glm::cross(a, b));
}

// TODO Standard feature
// Given three triangle vertices and a point on the triangle, compute the corresponding barycentric coordinates of the point.
// and return a vec3 with the barycentric coordinates (alpha, beta, gamma).
// - v0;     Triangle vertex 0
// - v1;     Triangle vertex 1
// - v2;     Triangle vertex 2
// - p;      Point on triangle
// - return; Corresponding barycentric coordinates for point p.
// This method is unit-tested, so do not change the function signature.
glm::vec3 computeBarycentricCoord(const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2, const glm::vec3& p)
{
    float A = triangleArea(v0, v1, v2);
    float alpha = triangleArea(p, v1, v2) / A;
    float beta = triangleArea(p, v0, v2) / A;
    float gamma = 1 - alpha - beta;            // triangleArea(p, v0, v1) / A;
    return glm::vec3(alpha, beta, gamma);
}

// TODO Standard feature
// Linearly interpolate three normals using barycentric coordinates.
// - n0;     Triangle normal 0
// - n1;     Triangle normal 1
// - n2;     Triangle normal 2
// - bc;     Barycentric coordinate
// - return; The smoothly interpolated normal.
// This method is unit-tested, so do not change the function signature.
glm::vec3 interpolateNormal(const glm::vec3& n0, const glm::vec3& n1, const glm::vec3& n2, const glm::vec3 bc)
{
    float nx = bc.x * n0.x + bc.y * n1.x + bc.z * n2.x;
    float ny = bc.x * n0.y + bc.y * n1.y + bc.z * n2.y;
    float nz = bc.x * n0.z + bc.y * n1.z + bc.z * n2.z;
    return glm::vec3(nx, ny, nz);
}

// TODO Standard feature
// Linearly interpolate three texture coordinates using barycentric coordinates.
// - n0;     Triangle texture coordinate 0
// - n1;     Triangle texture coordinate 1
// - n2;     Triangle texture coordinate 2
// - bc;     Barycentric coordinate
// - return; The smoothly interpolated texturre coordinate.
// This method is unit-tested, so do not change the function signature.
glm::vec2 interpolateTexCoord(const glm::vec2& t0, const glm::vec2& t1, const glm::vec2& t2, const glm::vec3 bc)
{
    float tx = bc.x * t0.x + bc.y * t1.x + bc.z * t2.x;
    float ty = bc.x * t0.y + bc.y * t1.y + bc.z * t2.y;
    return glm::vec2(tx, ty);
}
