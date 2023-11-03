#include "extra.h"
#include "bvh.h"
#include "light.h"
#include "recursive.h"
#include "shading.h"
#include "draw.h"
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
    //glm::vec3 directionToFocus = glm::normalize(focalPoint - camera.position());
    float focalDistance = glm::length(focalPoint - camera.position());
    //glm::vec3 lensPosition = camera.position() + focalDistance * directionToFocus;

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

Screen onlyBrightPixels(const Screen& image) 
{
    Screen onlyBright = image;
    
    int x = image.resolution().x;
    int y = image.resolution().y;

    for (int i = 0; i < x; i++) {
        for (int j = 0; i < y; i++) {

            glm::vec3 currentPixel = image.pixels()[image.indexAt(i, j)];

            // convert to grayscale and comapre with brightness threshold
            if (currentPixel.x * 0.299 + currentPixel.y * 0.587 + currentPixel.z * 0.114 <= 0.6) {
                onlyBright.setPixel(i, j, glm::vec3(0.0f));
            }
        }
    }

    return onlyBright;
}

int factorial(int n) 
{
    if (n == 1 || n == 0) return 1;
    else return n * factorial(n - 1);
}

std::vector<float> calculateGaussianKernel(int size)
{
    std::vector<float> kernel;
    float coefficientSum = 0.0f;
    int nFactorial = factorial(size);

    for (int i = 0; i <= size; i++) {
        
        int nkFactorial = factorial(size - i);
        int kFactorial = factorial(i);

        float coefficient = nFactorial / (kFactorial * nkFactorial);

        coefficientSum += coefficient;
        kernel.push_back(coefficient);
    }

    for (int i = 0; i <= size; i++) {
        kernel[i] /= coefficientSum;
    }

    return kernel;
}

void applyGaussianFilter(Screen& image, std::vector<float> kernel, bool isHorizontal) {

    int x = image.resolution().x;
    int y = image.resolution().y;

    int halfExtent = static_cast<int>(std::floor(kernel.size() / 2));

    for (int i = 0; i < x; i++) {
        for (int j = 0; j < y; j++) {

            glm::vec3 newColor = glm::vec3(0);

            for (int k = 0; k < kernel.size(); k++) {

                int coordI = i;
                int coordJ = j;
                if (isHorizontal) coordI = i - halfExtent + k;
                else coordJ = j - halfExtent + k;

                if (coordI >= 0 && coordI < x && coordJ >= 0 && coordJ < y) {
                    newColor += image.pixels()[image.indexAt(coordI, coordJ)] * kernel[k];
                }
            }
            
            image.setPixel(i, j, newColor);
        }
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
    image = onlyBrightPixels(image);
    std::vector<float> kernel = calculateGaussianKernel(16);
    applyGaussianFilter(image, kernel, true);
    applyGaussianFilter(image, kernel, false);
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
    using Node = BVH::Node;
    AxisAlignedBox aabb0;
    AxisAlignedBox aabb1;
    float lowest = std::numeric_limits<float>::max();
    float surfaceArea;
    size_t k = 0;

    // Calculate the surface area of the parent node.
    float x = aabb.upper.x - aabb.lower.x;
    float y = aabb.upper.y - aabb.lower.y;
    float z = aabb.upper.z - aabb.lower.z;
    float aabbArea = 2 * (x * y + x * z + y * z);

    // Sort the primitives.
    std::vector<Primitive> p;
    p.assign(primitives.begin(), primitives.end());
    if (axis == 0)
    std::sort(p.begin(), p.end(), [](Primitive p0, Primitive p1) {
        return computePrimitiveCentroid(p0).x < computePrimitiveCentroid(p1).x;
        });

    else if (axis == 1)
    std::sort(p.begin(), p.end(), [](Primitive p0, Primitive p1) {
        return computePrimitiveCentroid(p0).y < computePrimitiveCentroid(p1).y;
    });

    else if (axis == 2)
    std::sort(p.begin(), p.end(), [](Primitive p0, Primitive p1) {
        return computePrimitiveCentroid(p0).z < computePrimitiveCentroid(p1).z;
    });

    // Define the bins or interval of splitting planes.
    const int size = std::ceil(primitives.size() / 8.0f);

    // Repeat the following process for every bin.
    for (int i = 1; i < p.size() - 1; i += size)
    {
        // Compute the probability of hitting A and B multiplied by the amount of primitives (cost of intersection of A and B's elements).
        std::vector<Primitive> v0({ p.begin(), p.begin() + i });
        std::vector<Primitive> v1({ p.begin() + i, p.end() });
        aabb0 = computeSpanAABB(v0);
        x = aabb0.upper.x - aabb0.lower.x;
        y = aabb0.upper.y - aabb0.lower.y;
        z = aabb0.upper.z - aabb0.lower.z;
        surfaceArea = (x * y + x * z + y * z) * v0.size();

        aabb1 = computeSpanAABB(v1);
        x = aabb1.upper.x - aabb1.lower.x;
        y = aabb1.upper.y - aabb1.lower.y;
        z = aabb1.upper.z - aabb1.lower.z;
        surfaceArea += (x * y + x * z + y * z) * v1.size();
        surfaceArea *= 2;
        surfaceArea /= aabbArea;

        // If the cost is lower than what we have calculated before, replace this is as the current lowest cost with its corresponding splitting index.
        if (surfaceArea < lowest)
        {
            lowest = surfaceArea;
            k = i;
        }
    }

    for (int i = 0; i < primitives.size(); i++)
        primitives[i] = p[i];
    
    // Return the index of the first element in the second subrange (splitting index).
    return k;
}

// Helper method to count amount of non-leaf nodes in the BVH (including the dummy node).
uint32_t countNodes(const BVHInterface& bvh)
{
    // If root node is a leaf return 0.
    uint32_t count = 0;
    using Node = BVHInterface::Node;
    Node node = bvh.nodes()[0];
    if (node.isLeaf())
        return 0;

    // Go through every left child of a node, when a leaf is hit check the last seen right child (on top of the stack).
    std::vector<Node> n;
    while (true)
    {
        if (node.isLeaf())
        {
            if (n.empty())
                break;

            node = n.back();
            n.pop_back();
            
        }

        else
        {
            count++;
            n.push_back(bvh.nodes()[node.rightChild()]);
            node = bvh.nodes()[node.leftChild()];
        }
    }

    return count;
}

// Return a vector with all the primitives under a given node.
std::vector<BVHInterface::Primitive> findPrimitives(const BVHInterface& bvh, BVHInterface::Node node)
{
    using Node = BVHInterface::Node;
    using Primitive = BVHInterface::Primitive;
    std::vector<Primitive> primitives;
    std::vector<Node> nodes;
    nodes.push_back(node);

    // Go through every left child and store the corresponding right child on the stack.
    // When a leaf is hit, save the leaf in a separate vector, and traverse through the last saved right child (on top of the stack).
    while (!nodes.empty())
    {
        if (node.isLeaf())
        {
            for (int i = 0; i < node.primitiveCount(); i++)
            {
                primitives.push_back(bvh.primitives()[node.primitiveOffset() + i]);
            }

            node = nodes.back();
            nodes.pop_back();
        }

        else
        {
            nodes.push_back(bvh.nodes()[node.rightChild()]);
            node = bvh.nodes()[node.leftChild()];
        }
    }

    return primitives;
}

// Debug for SAH+Binning. Show for the selected node the AABB of itself and its children,
// and show the triangles under those children by matching their colors to their node's AABB color.
// Every triangle has a slightly different accent of their designated color by using a randomizer.
void test(Sampler& sampler, const BVHInterface& bvh, int nodeIndex)
{
    if (nodeIndex >= 0)
    {
        using Node = BVHInterface::Node;
        using Primitive = BVHInterface::Primitive;
        Node node = bvh.nodes()[nodeIndex];
        if (!node.isLeaf()) {
            drawAABB(node.aabb, DrawMode::Wireframe, glm::vec3(1, 0, 0));
            Node leftChild = bvh.nodes()[node.leftChild()];
            drawAABB(leftChild.aabb, DrawMode::Wireframe, glm::vec3(0, 1, 0));
            std::vector<Primitive> primitivesLeftChild = findPrimitives(bvh, leftChild);
            for (const Primitive& primitive : primitivesLeftChild) {
                drawTriangle(primitive.v0, primitive.v1, primitive.v2, { .kd = sampler.next_1d() * glm::vec3(0, 1, 0) });
            }
            Node rightChild = bvh.nodes()[node.rightChild()];
            drawAABB(rightChild.aabb, DrawMode::Wireframe, glm::vec3(0, 0, 1));
            std::vector<Primitive> primitivesRightChild = findPrimitives(bvh, rightChild);
            for (const Primitive& primitive : primitivesRightChild) {
                drawTriangle(primitive.v0, primitive.v1, primitive.v2, { .kd = sampler.next_1d() * glm::vec3(0, 0, 1) });
            }
        }
    }
}