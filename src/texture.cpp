#include "texture.h"
#include "render.h"
#include <framework/image.h>

// TODO: Standard feature
// Given an image, and relevant texture coordinates, sample the texture s.t.
// the nearest texel to the coordinates is acquired from the image.
// - image;    the image object to sample from.
// - texCoord; sample coordinates, generally in [0, 1]
// - return;   the nearest corresponding texel
// This method is unit-tested, so do not change the function signature.
// TODO: implement this function.
// Note: the pixels are stored in a 1D array, row-major order. You can convert from (i, j) to
//       an index using the method seen in the lecture.
// Note: the center of the first pixel should be at coordinates (0.5, 0.5)
// Given texcoords, return the corresponding pixel of the image
// The pixel are stored in a 1D array of row major order
// you can convert from position (i,j) to an index using the method seen in the lecture
// Note, the center of the first pixel is at image coordinates (0.5, 0.5)
glm::vec3 sampleTextureNearest(const Image& image, const glm::vec2& texCoord)
{
    int i = std::floor(texCoord.x * (image.width));
    int j = std::floor(texCoord.y * (image.height));

    i = std::clamp(i, 0, image.width - 1);
    j = std::clamp(j, 0, image.height - 1);

    int index = j * image.width + i;
    return image.pixels[index];
}

glm::vec2 calculateBoundary(const float p)
{
    float a, b;
    if (std::abs(p - std::floor(p)) > std::abs(p - std::ceil(p))) {
        a = std::floor(p) + 0.5;
        b = std::ceil(p) + 0.5;
    }
    else if (std::abs(p - std::ceil(p)) > std::abs(p - std::floor(p))) {
        a = std::floor(p) - 0.5;
        b = std::ceil(p) - 0.5;
    }

    return glm::vec2{a, b};
}

glm::vec4 boundaryPoints(const float x, const float y) 
{
    glm::vec2 xAxis = calculateBoundary(x);
    glm::vec2 yAxis = calculateBoundary(y);
    return glm::vec4 { xAxis.x, xAxis.y, yAxis.x, yAxis.y };
}

// TODO: Standard feature
// Given an image, and relevant texture coordinates, sample the texture s.t.
// a bilinearly interpolated texel is acquired from the image.
// - image;    the image object to sample from.
// - texCoord; sample coordinates, generally in [0, 1]
// - return;   the filter of the corresponding texels
// This method is unit-tested, so do not change the function signature.
// TODO: implement this function.
// Note: the pixels are stored in a 1D array, row-major order. You can convert from (i, j) to
//       an index using the method seen in the lecture.
// Note: the center of the first pixel should be at coordinates (0.5, 0.5)
// Given texcoords, return the corresponding pixel of the image
// The pixel are stored in a 1D array of row major order
// you can convert from position (i,j) to an index using the method seen in the lecture
// Note, the center of the first pixel is at image coordinates (0.5, 0.5)
glm::vec3 sampleTextureBilinear(const Image& image, const glm::vec2& texCoord)
{
    float i = texCoord.x * (image.width);
    float j = texCoord.y * (image.height);

    //i = std::max(0.0f, std::min(i, static_cast<float>(image.width - 1)));
    //j = std::max(0.0f, std::min(j, static_cast<float>(image.height - 1)));

    i = std::clamp(i, 0.0f, static_cast<float>(image.width - 1));
    j = std::clamp(j, 0.0f, static_cast<float>(image.height - 1));

    glm::vec4 c = boundaryPoints(i, j);
    // lower x boundary     - c.x
    // higher x boundary    - c.y
    // lower y boundary     - c.z
    // higher y boundary    - c.w
    // lower, lower is (0, 0) -> upper left corner
    
    float alpha = i - c.x;
    float beta = c.w - j;

    // int index = j * image.width + i;
    int indexBL = std::floor(c.w) * image.width + std::floor(c.x);
    glm::vec3 bottomLeft = image.pixels[indexBL];
    int indexBR = std::floor(c.w) * image.width + std::floor(c.y);
    glm::vec3 bottomRight = image.pixels[indexBR];
    int indexUL = std::floor(c.z) * image.width + std::floor(c.x);
    glm::vec3 upperLeft = image.pixels[indexUL];
    int indexUR = std::floor(c.z) * image.width + std::floor(c.y);
    glm::vec3 upperRight = image.pixels[indexUR];

    glm::vec3 finalColor = (1 - alpha) * (1 - beta) * bottomLeft + 
        alpha * (1 - beta) * bottomRight + 
        (1 - alpha) * beta * upperLeft + 
        alpha * beta * upperRight;
    
    return finalColor;
}