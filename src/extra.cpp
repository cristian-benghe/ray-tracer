#include "extra.h"
#include "bvh.h"
#include "light.h"
#include "recursive.h"
#include "shading.h"
#include <framework/trackball.h>
std::vector<Ray> sampledRays(glm::vec3 pixelOrigin, glm::vec3 pixelDirection, float apertureSize, int numRays, const Trackball& camera, float focusDistance)
{
    std::vector<Ray> rays;
    // Calculate a vector perpendicular to the camera direction
    glm::vec3 cameraUp = camera.up(); 
    glm::vec3 cameraRight = -camera.left();

    for (int i = 0; i < numRays; ++i) {
        float offset_x = (2.0 * rand() / RAND_MAX - 1.0) * apertureSize;
        float offset_y = (2.0 * rand() / RAND_MAX - 1.0) * apertureSize;

        glm::vec3 dirr = glm::normalize(pixelDirection);

        glm::vec3 pointt = dirr * focusDistance + pixelOrigin;

        glm::vec3 origin = pixelOrigin + offset_x * cameraRight + offset_y * cameraUp;
        rays.push_back(Ray { origin, glm::normalize(pointt - origin), std::numeric_limits<float>::max() });
    }
    return rays;
}

// TODO; Extra feature
// Given the same input as for `renderImage()`, instead render an image with your own implementation
// of Depth of Field. Here, you generate camera rays s.t. a focus point and a thin lens camera model
// are in play, allowing objects to be in and out of focus.
// This method is not unit-tested, but we do expect to find it **exactly here**, and we'd rather
// not go on a hunting expedition for your implementation, so please keep it here!
void renderImageWithDepthOfField(const Scene& scene, const BVHInterface& bvh, const Features& features, const Trackball& camera, Screen& screen)
{   
    if (!features.extra.enableDepthOfField) {
        return;
    }
    float depth = features.extra.depth; //focalDistance
    glm::vec2 position = (glm::vec2(0) + 0.5f) / glm::vec2(screen.resolution()) * 2.f - 1.f;
    glm::vec3 dir = camera.generateRay(position).direction;
    glm::vec3 focalPoint = camera.position() + depth * dir; //point that is in focus
    glm::vec3 directionToFocus = glm::normalize(focalPoint - camera.position());
    float focalDistance = glm::length(focalPoint - camera.position());
    glm::vec3 lensPosition = camera.position() + focalDistance * directionToFocus;

#ifdef NDEBUG // Enable multi threading in Release mode
#pragma omp parallel for schedule(guided)
#endif
    for (int y = 0; y < screen.resolution().y; y++) {
        for (int x = 0; x != screen.resolution().x; x++) {
            // Assemble useful objects on a per-pixel basis; e.g. a per-thread sampler
            // Note; we seed the sampler for consistenct behavior across frames
            RenderState state = {
                .scene = scene,
                .features = features,
                .bvh = bvh,
                .sampler = { static_cast<uint32_t>(screen.resolution().y * x + y) }
            };
            glm::vec2 position = (glm::vec2(x,y) + 0.5f) / glm::vec2(screen.resolution()) * 2.f - 1.f;
             
            auto rays = sampledRays(camera.generateRay(position).origin,
                camera.generateRay(position).direction,
                0.10f, 8, camera, features.extra.depth);

            auto L = renderRays(state, rays);
            screen.setPixel(x, y, L);
        }
    }
}

// TODO; Extra feature
// Given the same input as for `renderImage()`, instead render an image with your own implementation
// of motion blur. Here, you integrate over a time domain, and not just the pixel's image domain,
// to give objects the appearance of "fast movement".
// This method is not unit-tested, but we do expect to find it **exactly here**, and we'd rather
// not go on a hunting expedition for your implementation, so please keep it here!
void renderImageWithMotionBlur(const Scene& scene, const BVHInterface& bvh, const Features& features, const Trackball& camera, Screen& screen)
{
    if (!features.extra.enableMotionBlur) {
        return;
    }

}

// TODO; Extra feature
// Given a rendered image, compute and apply a bloom post-processing effect to increase bright areas.
// This method is not unit-tested, but we do expect to find it **exactly here**, and we'd rather
// not go on a hunting expedition for your implementation, so please keep it here!
void postprocessImageWithBloom(const Scene& scene, const Features& features, const Trackball& camera, Screen& image)
{
    if (!features.extra.enableBloomEffect) {
        return;
    }

    // ...
}

// TODO; Extra feature
// Given a camera ray (or reflected camera ray) and an intersection, evaluates the contribution of a set of
// glossy reflective rays, recursively evaluating renderRay(..., depth + 1) along each ray, and adding the
// results times material.ks to the current intersection's hit color.
// - state;    the active scene, feature config, bvh, and sampler
// - ray;      camera ray
// - hitInfo;  intersection object
// - hitColor; current color at the current intersection, which this function modifies
// - rayDepth; current recursive ray depth
// This method is not unit-tested, but we do expect to find it **exactly here**, and we'd rather
// not go on a hunting expedition for your implementation, so please keep it here!
void renderRayGlossyComponent(RenderState& state, Ray ray, const HitInfo& hitInfo, glm::vec3& hitColor, int rayDepth)
{
    // Generate an initial specular ray, and base secondary glossies on this ray
    auto numSamples = state.features.extra.numGlossySamples;

    Ray reflectedRay = generateReflectionRay(ray, hitInfo);

    glm::vec3 u, v;
    glm::vec3 r = ray.direction;

    u = glm::cross(glm::vec3(0, 1, 0), r);
    if (r == glm::vec3(0,1,0))
        u = glm::cross(glm::vec3(1, 0, 0), r);
    v = glm::cross(r, u);

    u = glm::normalize(u);
    v = glm::normalize(v);
    float radius = 1.0f; 
    
    glm::vec3 computedGlossyColor = glm::vec3(0);

    for (int i = 0; i < numSamples; ++i) {
        float theta = (static_cast<float>(rand()) / RAND_MAX) * 2 * glm::pi<float>();

        float randomSmallerRadius = static_cast<double>(rand()) / RAND_MAX * radius;

        glm::vec3 r_prime_direction = glm::normalize(reflectedRay.direction + hitInfo.material.shininess / 64.0f * 
            (u * (randomSmallerRadius * cos(theta)) + v * (randomSmallerRadius * sin(theta))));

        Ray perturbedRay;
        perturbedRay.origin = reflectedRay.origin;
        perturbedRay.direction = r_prime_direction;
        perturbedRay.t = std::numeric_limits<float>::max();
        computedGlossyColor += renderRay(state, perturbedRay, rayDepth + 1);
    }
    
    hitColor += computedGlossyColor / (float) numSamples * hitInfo.material.ks;
}

// TODO; Extra feature
// Given a camera ray (or reflected camera ray) that does not intersect the scene, evaluates the contribution
// along the ray, originating from an environment map. You will have to add support for environment textures
// to the Scene object, and provide a scene with the right data to supply this.
// - state; the active scene, feature config, bvh, and sampler
// - ray;   ray object
// This method is not unit-tested, but we do expect to find it **exactly here**, and we'd rather
// not go on a hunting expedition for your implementation, so please keep it here!
glm::vec3 sampleEnvironmentMap(RenderState& state, Ray ray)
{
    if (state.features.extra.enableEnvironmentMap) {
        // Part of your implementation should go here
        return glm::vec3(0.f);
    } else {
        return glm::vec3(0.f);
    }
}


// TODO: Extra feature
// As an alternative to `splitPrimitivesByMedian`, use a SAH+binning splitting criterion. Refer to
// the `Data Structures` lecture for details on this metric.
// - aabb;       the axis-aligned bounding box around the given triangle set
// - axis;       0, 1, or 2, determining on which axis (x, y, or z) the split must happen
// - primitives; the modifiable range of triangles that requires splitting
// - return;     the split position of the modified range of triangles
// This method is unit-tested, so do not change the function signature.
size_t splitPrimitivesBySAHBin(const AxisAlignedBox& aabb, uint32_t axis, std::span<BVH::Primitive> primitives)
{
    using Primitive = BVH::Primitive;

    return 0; // This is clearly not the solution
}