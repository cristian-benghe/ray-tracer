#include "bvh.h"
#include "draw.h"
#include "extra.h"
#include "interpolate.h"
#include "intersect.h"
#include "render.h"
#include "scene.h"
#include "texture.h"
#include <algorithm>
#include <bit>
#include <chrono>
#include <framework/opengl_includes.h>
#include <iostream>

bool isInsideAABB(const AxisAlignedBox& aabb, const glm::vec3& origin)
{
    if (glm::all(glm::lessThanEqual(origin, aabb.upper)) && glm::all(glm::greaterThanEqual(origin, aabb.lower)))
        return true;
    return false;
}

// Helper method to fill in hitInfo object. This can be safely ignored (or extended).
// Note: many of the functions in this helper tie in to standard/extra features you will have
// to implement separately, see interpolate.h/.cpp for these parts of the project
void updateHitInfo(RenderState& state, const BVHInterface::Primitive& primitive, const Ray& ray, HitInfo& hitInfo)
{
    const auto& [v0, v1, v2] = std::tie(primitive.v0, primitive.v1, primitive.v2);
    const auto& mesh = state.scene.meshes[primitive.meshID];
    const auto n = glm::normalize(glm::cross(v1.position - v0.position, v2.position - v0.position));
    const auto p = ray.origin + ray.t * ray.direction;

    // First, fill in default data, unrelated to separate features
    hitInfo.material = mesh.material;
    hitInfo.normal = n;
    hitInfo.barycentricCoord = computeBarycentricCoord(v0.position, v1.position, v2.position, p);

    // Next, if `features.enableNormalMapping` is true, generate smoothly interpolated vertex normals
    if (state.features.enableNormalInterp) {
        hitInfo.normal = interpolateNormal(v0.normal, v1.normal, v2.normal, hitInfo.barycentricCoord);
    }

    // Next, if `features.enableTextureMapping` is true, generate smoothly interpolated vertex uvs
    if (state.features.enableTextureMapping) {
        hitInfo.texCoord = interpolateTexCoord(v0.texCoord, v1.texCoord, v2.texCoord, hitInfo.barycentricCoord);
    }

    // Finally, catch flipped normals
    if (glm::dot(ray.direction, n) > 0.0f) {
        hitInfo.normal = -hitInfo.normal;
    }
}

// BVH constructor; can be safely ignored. You should not have to touch this
// NOTE: this constructor is tested, so do not change the function signature.
BVH::BVH(const Scene& scene, const Features& features)
{
    // Store start of bvh build for timing
    using clock = std::chrono::high_resolution_clock;
    const auto start = clock::now();

    // Count the total nr. of triangles in the scene
    size_t numTriangles = 0;
    for (const auto& mesh : scene.meshes)
        numTriangles += mesh.triangles.size();

    // Given the input scene, gather all triangles over which to build the BVH as a list of Primitives
    std::vector<Primitive> primitives;
    primitives.reserve(numTriangles);
    for (uint32_t meshID = 0; meshID < scene.meshes.size(); meshID++) {
        const auto& mesh = scene.meshes[meshID];
        for (const auto& triangle : mesh.triangles) {
            primitives.push_back(Primitive {
                .meshID = meshID,
                .v0 = mesh.vertices[triangle.x],
                .v1 = mesh.vertices[triangle.y],
                .v2 = mesh.vertices[triangle.z] });
        }
    }

    // Tell underlying vectors how large they should approximately be
    m_primitives.reserve(numTriangles);
    m_nodes.reserve(numTriangles + 1);

    // Recursively build BVH structure; this is where your implementation comes in
    m_nodes.emplace_back(); // Create root node
    m_nodes.emplace_back(); // Create dummy node s.t. children are allocated on the same cache line
    buildRecursive(scene, features, primitives, RootIndex);

    // Fill in boilerplate data
    buildNumLevels();
    buildNumLeaves();

    // Output end of bvh build for timing
    const auto end = clock::now();
    std::cout << "BVH construction time: " << std::chrono::duration<double, std::milli>(end - start).count() << "ms" << std::endl;
    printf("%u\n", m_primitives.size());
}

// BVH helper method; allocates a new node and returns its index
// You should not have to touch this
uint32_t BVH::nextNodeIdx()
{
    const auto idx = static_cast<uint32_t>(m_nodes.size());
    m_nodes.emplace_back();
    return idx;
}

// TODO: Standard feature
// Given a BVH triangle, compute an axis-aligned bounding box around the primitive
// - primitive; a single triangle to be stored in the BVH
// - return;    an axis-aligned bounding box around the triangle
// This method is unit-tested, so do not change the function signature.
AxisAlignedBox computePrimitiveAABB(const BVHInterface::Primitive primitive)
{

    glm::vec3 v0 = primitive.v0.position;
    glm::vec3 v1 = primitive.v1.position;
    glm::vec3 v2 = primitive.v2.position;
    float lowerX = glm::min(v0.x, v1.x);
    lowerX = glm::min(lowerX, v2.x);
    float lowerY = glm::min(v0.y, v1.y);
    lowerY = glm::min(lowerY, v2.y);
    float lowerZ = glm::min(v0.z, v1.z);
    lowerZ = glm::min(lowerZ, v2.z);
    float upperX = glm::max(v0.x, v1.x);
    upperX = glm::max(upperX, v2.x);
    float upperY = glm::max(v0.y, v1.y);
    upperY = glm::max(upperY, v2.y);
    float upperZ = glm::max(v0.z, v1.z);
    upperZ = glm::max(upperZ, v2.z);
    return { .lower = glm::vec3(lowerX, lowerY, lowerZ), .upper = glm::vec3(upperX, upperY, upperZ) };
}

// TODO: Standard feature
// Given a range of BVH triangles, compute an axis-aligned bounding box around the range.
// - primitive; a contiguous range of triangles to be stored in the BVH
// - return; a single axis-aligned bounding box around the entire set of triangles
// This method is unit-tested, so do not change the function signature.
AxisAlignedBox computeSpanAABB(std::span<const BVHInterface::Primitive> primitives)
{
    glm::vec3 lowest(std::numeric_limits<float>::max());
    glm::vec3 highest(std::numeric_limits<float>::lowest());

    for (const BVH::Primitive& primitive : primitives) {
        lowest = glm::min(lowest, computePrimitiveAABB(primitive).lower);
        highest = glm::max(highest, computePrimitiveAABB(primitive).upper);
    }

    return { .lower = lowest, .upper = highest };
}

// TODO: Standard feature
// Given a BVH triangle, compute the geometric centroid of the triangle
// - primitive; a single triangle to be stored in the BVH
// - return; the geometric centroid of the triangle's vertices
// This method is unit-tested, so do not change the function signature.
glm::vec3 computePrimitiveCentroid(const BVHInterface::Primitive primitive)
{
    glm::vec3 v0 = primitive.v0.position;
    glm::vec3 v1 = primitive.v1.position;
    glm::vec3 v2 = primitive.v2.position;
    return (v0 + v1 + v2) / 3.0f;
}

// TODO: Standard feature
// Given an axis-aligned bounding box, compute the longest axis; x = 0, y = 1, z = 2.
// - aabb;   the input axis-aligned bounding box
// - return; 0 for the x-axis, 1 for the y-axis, 2 for the z-axis
// if several axes are equal in length, simply return the first of these
// This method is unit-tested, so do not change the function signature.
uint32_t computeAABBLongestAxis(const AxisAlignedBox& aabb)
{
    glm::vec3 diff = aabb.upper - aabb.lower;
    if (diff.x >= diff.y) {
        if (diff.x >= diff.z) {
            return 0;
        }

        return 2;
    }

    else if (diff.y >= diff.z) {
        return 1;
    }

    return 2;
}

// TODO: Standard feature
// Given a range of BVH triangles, sort these along a specified axis based on their geometric centroid.
// Then, find and return the split index in the range, such that the subrange containing the first element
// of the list is at least as big as the other, and both differ at most by one element in size.
// Hint: you should probably reuse `computePrimitiveCentroid()`
// - aabb;       the axis-aligned bounding box around the given triangle range
// - axis;       0, 1, or 2, determining on which axis (x, y, or z) the split must happen
// - primitives; the modifiable range of triangles that requires sorting/splitting along an axis
// - return;     the split position of the modified range of triangles
// This method is unit-tested, so do not change the function signature.
size_t splitPrimitivesByMedian(const AxisAlignedBox& aabb, uint32_t axis, std::span<BVHInterface::Primitive> primitives)
{
    using Primitive = BVHInterface::Primitive;

    std::vector<Primitive> centroids;
    centroids.assign(primitives.begin(), primitives.end());
    if (axis == 0)
        std::sort(centroids.begin(), centroids.end(), [](Primitive p0, Primitive p1) {
            return computePrimitiveCentroid(p0).x < computePrimitiveCentroid(p1).x;
        });
    else if (axis == 1)
        std::sort(centroids.begin(), centroids.end(), [](Primitive p0, Primitive p1) {
            return computePrimitiveCentroid(p0).y < computePrimitiveCentroid(p1).y;
        });
    else if (axis == 2)
        std::sort(centroids.begin(), centroids.end(), [](Primitive p0, Primitive p1) {
            return computePrimitiveCentroid(p0).z < computePrimitiveCentroid(p1).z;
        });

    for (int i = 0; i < primitives.size(); i++) {
        primitives[i] = centroids[i];
    }

    return std::round(centroids.size() / 2.0f) - 1;
}

bool sd(const AxisAlignedBox& box, Ray& ray)
{
    float xmin = box.lower.x;
    float xmax = box.upper.x;
    float ymin = box.lower.y;
    float ymax = box.upper.y;
    float zmin = box.lower.z;
    float zmax = box.upper.z;
    if (xmin == xmax || ymin == ymax || zmin == zmax || glm::length(ray.direction) == 0)
        return false;
    float txmin = (xmin - ray.origin.x) / ray.direction.x;
    float txmax = (xmax - ray.origin.x) / ray.direction.x;
    float tymin = (ymin - ray.origin.y) / ray.direction.y;
    float tymax = (ymax - ray.origin.y) / ray.direction.y;
    float tzmin = (zmin - ray.origin.z) / ray.direction.z;
    float tzmax = (zmax - ray.origin.z) / ray.direction.z;
    float txin = std::min(txmin, txmax);
    float txout = std::max(txmin, txmax);
    float tyin = std::min(tymin, tymax);
    float tyout = std::max(tymin, tymax);
    float tzin = std::min(tzmin, tzmax);
    float tzout = std::max(tzmin, tzmax);
    float tin = std::max(txin, tyin);
    tin = std::max(tin, tzin);
    float tout = std::min(txout, tyout);
    tout = std::min(tout, tzout);
    if (tout <= tin || tout <= 0)
        return false;

    if (tin <= 0) {
        if (tout < ray.t)
            ray.t = tout;
        else
            return false;
    }

    else {
        if (tin < ray.t)
            ray.t = tin;
        else
            return false;
    }

    return true;
}


// TODO: Standard feature
// Hierarchy traversal routine; called by the BVH's intersect(),
// you must implement this method and implement it carefully!
//
// If `features.enableAccelStructure` is not enabled, the method should just iterate the BVH's
// underlying primitives (or the scene's geometry). The default imlpementation already does this.
// You will have to implement the part which actually traverses the BVH for a faster intersect,
// given that `features.enableAccelStructure` is enabled.
//
// This method returns `true` if geometry was hit, and `false` otherwise. On first/closest hit, the
// distance `t` in the `ray` object is updated, and information is updated in the `hitInfo` object.
//
// - state;    the active scene, and a user-specified feature config object, encapsulated
// - bvh;      the actual bvh which should be traversed for faster intersection
// - ray;      the ray intersecting the scene's geometry
// - hitInfo;  the return object, with info regarding the hit geometry
// - return;   boolean, if geometry was hit or not
//
// This method is unit-tested, so do not change the function signature.
bool intersectRayWithBVH(RenderState& state, const BVHInterface& bvh, Ray& ray, HitInfo& hitInfo)
{
    // Relevant data in the constructed BVH
    std::span<const BVHInterface::Node> nodes = bvh.nodes();
    std::span<const BVHInterface::Primitive> primitives = bvh.primitives();

    // Return value
    bool is_hit = false;

    if (state.features.enableAccelStructure) {
        // TODO: implement here your (probably stack-based) BVH traversal.
        //
        // Some hints (refer to bvh_interface.h either way). BVH nodes are packed, so the
        // data is not easily extracted. Helper methods are available, however:
        // - For a given node, you can test if the node is a leaf with `node.isLeaf()`.
        // - If the node is not a leaf, you can obtain the left/right children with `node.leftChild()` etc.
        // - If the node is a leaf, you can obtain the offset to and nr. of primitives in the bvh's list
        //   of underlying primitives with `node.primitiveOffset()` and `node.primitiveCount()`
        //
        // In short, you will have to step down the bvh, node by node, and intersect your ray
        // with the node's AABB. If this intersection passes, you should:
        // - if the node is a leaf, intersect with the leaf's primitives
        // - if the node is not a leaf, test the left and right children as well!
        //
        // Note that it is entirely possible for a ray to hit a leaf node, but not its primitives,
        // and it is likewise possible for a ray to hit both children of a node.
        std::vector<BVHInterface::Node> tNodes;
        tNodes.reserve(256);
        BVHInterface::Node node = nodes[0];
        BVHInterface::Primitive p;
        tNodes.push_back(node);
        float t;
        while (!tNodes.empty()) {
            t = ray.t;
            //drawAABB(node.aabb, DrawMode::Wireframe);
            if (intersectRayWithShape(node.aabb, ray) || isInsideAABB(node.aabb, ray.origin)) {
                //drawSphere(ray.t * ray.direction + ray.origin, 0.01f, glm::vec3(1));
                ray.t = t;
                if (node.isLeaf()) {
                    for (int j = 0; j < node.primitiveCount(); j++) {
                        p = primitives[node.primitiveOffset() + j];
                        if (intersectRayWithTriangle(p.v0.position, p.v1.position, p.v2.position, ray, hitInfo)) {
                            //drawTriangle(p.v0, p.v1, p.v2, { .kd = glm::vec3(1, 0, 0) });
                            updateHitInfo(state, p, ray, hitInfo);
                            is_hit = true;
                        }
                    }

                    node = tNodes.back();
                    tNodes.pop_back();
                }

                else {
                    tNodes.push_back(nodes[node.rightChild()]);
                    node = nodes[node.leftChild()];
                }
            }

            else {
                node = tNodes.back();
                tNodes.pop_back();
            }
        }

    } else {
        // Naive implementation; simply iterates over all primitives
        for (const auto& prim : primitives) {
            const auto& [v0, v1, v2] = std::tie(prim.v0, prim.v1, prim.v2);
            if (intersectRayWithTriangle(v0.position, v1.position, v2.position, ray, hitInfo)) {
                updateHitInfo(state, prim, ray, hitInfo);
                is_hit = true;
            }
        }
    }

    // Intersect with spheres.
    for (const auto& sphere : state.scene.spheres)
        is_hit |= intersectRayWithShape(sphere, ray, hitInfo);

    return is_hit;
}

// TODO: Standard feature
// Leaf construction routine; you should reuse this in in `buildRecursive()`
// Given an axis-aligned bounding box, and a range of triangles, generate a valid leaf object
// and store the triangles in the `m_primitives` vector.
// You are free to modify this function's signature, as long as the constructor builds a BVH
// - scene;      the active scene
// - features;   the user-specified features object
// - aabb;       the axis-aligned bounding box around the primitives beneath this leaf
// - primitives; the range of triangles to be stored for this leaf
BVH::Node BVH::buildLeafData(const Scene& scene, const Features& features, const AxisAlignedBox& aabb, std::span<Primitive> primitives)
{
    Node node;
    // TODO fill in the leaf's data; refer to `bvh_interface.h` for details

    node.aabb = aabb;
    uint32_t data0 = 1u << 31;
    data0 += m_primitives.size();
    node.data = { data0, static_cast<uint32_t>(primitives.size()) };

    // Copy the current set of primitives to the back of the primitives vector
    std::copy(primitives.begin(), primitives.end(), std::back_inserter(m_primitives));

    return node;
}

// TODO: Standard feature
// Node construction routine; you should reuse this in in `buildRecursive()`
// Given an axis-aligned bounding box, and left/right child indices, generate a valid node object.
// You are free to modify this function's signature, as long as the constructor builds a BVH
// - scene;           the active scene
// - features;        the user-specified features object
// - aabb;            the axis-aligned bounding box around the primitives beneath this node
// - leftChildIndex;  the index of the node's left child in `m_nodes`
// - rightChildIndex; the index of the node's right child in `m_nodes`
BVH::Node BVH::buildNodeData(const Scene& scene, const Features& features, const AxisAlignedBox& aabb, uint32_t leftChildIndex, uint32_t rightChildIndex)
{
    Node node;
    // TODO fill in the node's data; refer to `bvh_interface.h` for details

    node.aabb = aabb;
    node.data = { leftChildIndex, rightChildIndex };

    return node;
}

// TODO: Standard feature
// Hierarchy construction routine; called by the BVH's constructor,
// you must implement this method and implement it carefully!
//
// You should implement the other BVH standard features first, and this feature last, as you can reuse
// most of the other methods to assemble this part. There are detailed instructions inside the
// method which we recommend you follow.
//
// Arguments:
// - scene;      the active scene
// - features;   the user-specified features object
// - primitives; a range of triangles to be stored in the BVH
// - nodeIndex;  index of the node you are currently working on, this is already allocated
//
// You are free to modify this function's signature, as long as the constructor builds a BVH
void BVH::buildRecursive(const Scene& scene, const Features& features, std::span<Primitive> primitives, uint32_t nodeIndex)
{
    // WARNING: always use nodeIndex to index into the m_nodes array. never hold a reference/pointer,
    // because a push/emplace (in ANY recursive calls) might grow vectors, invalidating the pointers.

    // Compute the AABB of the current node.
    // AxisAlignedBox aabb = computeSpanAABB(primitives);

    // As a starting point, we provide an implementation which creates a single leaf, and stores
    // all triangles inside it. You should remove or comment this, and work on your own recursive
    // construction algorithm that implements the following steps. Make sure to reuse the methods
    // you have previously implemented to simplify this process.
    //
    // 1. Determine if the node should be a leaf, when the nr. of triangles is less or equal to 4
    //    (hint; use the `LeafSize` constant)
    // 2. If it is a leaf, fill in the leaf's data, and store its range of triangles in `m_primitives`
    // 3. If it is a node:
    //    3a. Split the range of triangles along the longest axis into left and right subspans,
    //        using either median or SAH-Binning based on the `Features` object
    //    3b. Allocate left/right child nodes
    //        (hint: use `nextNodeIdx()`)
    //    3c. Fill in the current node's data; aabb, left/right child indices
    //    3d. Recursively build left/right child nodes over their respective triangles
    //        (hint; use `std::span::subspan()` to split into left/right ranges)

    // Just configure the current node as a giant leaf for now
    // m_nodes[nodeIndex] = buildLeafData(scene, features, aabb, primitives);

    AxisAlignedBox aabb = computeSpanAABB(primitives);
    if (primitives.size() <= LeafSize) {
        m_nodes[nodeIndex] = buildLeafData(scene, features, aabb, primitives);
    } else {
        size_t index;
        if (features.extra.enableBvhSahBinning)
            index = splitPrimitivesBySAHBin(aabb, computeAABBLongestAxis(aabb), primitives);
        else
            index = splitPrimitivesByMedian(aabb, computeAABBLongestAxis(aabb), primitives);

        uint32_t i0 = nextNodeIdx();
        uint32_t i1 = nextNodeIdx();
        m_nodes[nodeIndex] = buildNodeData(scene, features, aabb, i0, i1);
        std::span<Primitive> leftSubspan = primitives.subspan(0, index + 1);
        std::span<Primitive> rightSubspan = primitives.subspan(index + 1);
        buildRecursive(scene, features, leftSubspan, i0);
        buildRecursive(scene, features, rightSubspan, i1);
    }
}

// TODO: Standard feature, or part of it
// Compute the nr. of levels in your hierarchy after construction; useful for `debugDrawLevel()`
// You are free to modify this function's signature, as long as the constructor builds a BVH
void BVH::buildNumLevels()
{
    m_numLevels = 0;
    Node node = nodes()[0];
    while (!node.isLeaf()) {
        m_numLevels++;
        node = nodes()[node.leftChild()];
    }

    m_numLevels++;
}

// Compute the nr. of leaves in your hierarchy after construction; useful for `debugDrawLeaf()`
// You are free to modify this function's signature, as long as the constructor builds a BVH
void BVH::buildNumLeaves()
{
    m_numLeaves = 0;
    std::vector<Node> n;
    n.push_back(nodes()[0]);
    Node node;
    while (n.size() > 0) {
        node = n.back();
        n.pop_back();
        if (node.isLeaf()) {
            m_numLeaves++;
        }

        else {
            n.push_back(nodes()[node.rightChild()]);
            n.push_back(nodes()[node.leftChild()]);
        }
    }
}

// Draw the bounding boxes of the nodes at the selected level. Use this function to visualize nodes
// for debugging. You may wish to implement `buildNumLevels()` first. We suggest drawing the AABB
// of all nodes on the selected level.
// You are free to modify this function's signature.
void BVH::debugDrawLevel(int level)
{
    // Example showing how to draw an AABB as a (white) wireframe box.
    // Hint: use draw functions (see `draw.h`) to draw the contained boxes with different
    // colors, transparencies, etc.
    std::vector<Node> n;
    uint32_t l = 0;
    n.push_back(nodes()[0]);
    while (l < level) {
        const uint32_t f = n.size();
        for (int i = 0; i < f; i++) {
            if (!n[0].isLeaf()) {
                n.push_back(nodes()[n[0].leftChild()]);
                n.push_back(nodes()[n[0].rightChild()]);
            }

            n.erase(n.begin());
        }

        l++;
    }

    for (const Node& node : n) {
        drawAABB(node.aabb, DrawMode::Wireframe, glm::vec3(1, 1, 1), 1.0f);
    }
}

// Draw data of the leaf at the selected index. Use this function to visualize leaf nodes
// for debugging. You may wish to implement `buildNumLeaves()` first. We suggest drawing the AABB
// of the selected leaf, and then its underlying primitives with different colors.
// - leafIndex; index of the selected leaf.
//              (Hint: not the index of the i-th node, but of the i-th leaf!)
// You are free to modify this function's signature.
void BVH::debugDrawLeaf(int leafIndex)
{
    // Example showing how to draw an AABB as a (white) wireframe box.
    // Hint: use drawTriangle (see `draw.h`) to draw the contained primitives
    // AxisAlignedBox aabb { .lower = glm::vec3(0.0f), .upper = glm::vec3(1.0f, 1.05f, 1.05f) };
    // drawAABB(aabb, DrawMode::Wireframe, glm::vec3(0.05f, 1.0f, 0.05f), 0.1f);
    if (leafIndex > 0) {
        std::vector<Node> n;
        std::vector<Node> leafs;
        n.push_back(nodes()[0]);
        Node node;
        while (n.size() > 0) {
            node = n.back();
            n.pop_back();
            if (node.isLeaf())
                leafs.push_back(node);
            else {
                n.push_back(nodes()[node.rightChild()]);
                n.push_back(nodes()[node.leftChild()]);
            }
        }

        drawAABB(leafs[leafIndex - 1].aabb, DrawMode::Wireframe);
        Primitive primitive;
        std::array<Material, 4> m = { Material { .kd = glm::vec3(1, 0, 0) },
            Material { .kd = glm::vec3(0, 1, 0) },
            Material { .kd = glm::vec3(0, 0, 1) },
            Material { .kd = glm::vec3(.3, .6, .9) } };
        for (int i = 0; i < leafs[leafIndex - 1].primitiveCount(); i++) {
            primitive = primitives()[leafs[leafIndex - 1].primitiveOffset() + i];
            drawTriangle(primitive.v0, primitive.v1, primitive.v2, m[i]);
        }
    }
}

void drawBVHIntersection(const BVHInterface& bvh, Ray& ray, int index, bool showParent, bool showLeftChild, bool showRightChild)
{
    ray.t = std::numeric_limits<float>::max();
    std::vector<BVHInterface::Node> tNodes;
    tNodes.reserve(256);
    BVHInterface::Node node = bvh.nodes()[0];
    BVHInterface::Primitive p;
    HitInfo hitInfo;
    tNodes.push_back(node);
    float t;
    int i = 0;
    while (!tNodes.empty()) {
        ray.t += 0.1f;
        t = ray.t;
        if (i == index)
            drawSphere(ray.t * ray.direction + ray.origin, 0.01f, glm::vec3(0, 1, 0));
        if (i == index && !node.isLeaf() && showParent)
            drawAABB(node.aabb, DrawMode::Wireframe);
        if (i == index && !node.isLeaf() && showLeftChild)
            drawAABB(bvh.nodes()[node.leftChild()].aabb, DrawMode::Wireframe, {0, 1, 0});
        if (i == index && !node.isLeaf() && showRightChild)
            drawAABB(bvh.nodes()[node.rightChild()].aabb, DrawMode::Wireframe, {0, 0, 1});
        if (i == index && node.isLeaf() && showParent)
            drawAABB(node.aabb, DrawMode::Wireframe, glm::vec3(1, 0, 0));

        if (intersectRayWithShape(node.aabb, ray) || isInsideAABB(node.aabb, ray.origin)) {
            if (i == index && showParent)
                drawSphere(ray.t * ray.direction + ray.origin, 0.01f, glm::vec3(1, 0, 0));
            ray.t = t;
            if (node.isLeaf()) {
                for (int j = 0; j < node.primitiveCount(); j++) {
                    p = bvh.primitives()[node.primitiveOffset() + j];
                    drawTriangle(p.v0, p.v1, p.v2, { .kd = glm::vec3(1, 0, 0) });
                    if (intersectRayWithTriangle(p.v0.position, p.v1.position, p.v2.position, ray, hitInfo)) {

                    }
                }

                node = tNodes.back();
                tNodes.pop_back();
            }

            else {
                tNodes.push_back(bvh.nodes()[node.rightChild()]);
                node = bvh.nodes()[node.leftChild()];
            }

        } else {
            node = tNodes.back();
            tNodes.pop_back();
        }

        i++;
    }
}
