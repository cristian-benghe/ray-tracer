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
    glm::vec2 texCoordSafe = glm::clamp(texCoord, glm::vec2(0.0f), glm::vec2(1.0f) - glm::vec2(std::numeric_limits<float>::epsilon()));

    int i = static_cast<int>(std::floor(image.width * texCoordSafe.x));
    int j = static_cast<int>(std::floor(image.height * (1.0f - texCoordSafe.y)));

    int index = j * image.width + i;
    return image.pixels[index];
}

glm::vec2 calculateBoundary(const float p, const int maxBoundary)
{
    float lowerBound = 0.0f;
    float upperBound = 0.0f;
    if (std::abs(p - std::floor(p)) > std::abs(p - std::ceil(p))) {
        lowerBound = std::floor(p) + 0.5;
        upperBound = std::ceil(p) + 0.5;
    } else if (std::abs(p - std::ceil(p)) > std::abs(p - std::floor(p))) {
        lowerBound = std::floor(p) - 0.5;
        upperBound = std::ceil(p) - 0.5;
    }

    glm::vec2 bothBoundaries = glm::vec2 { lowerBound, upperBound };
    //glm::vec2 bothBoundariesSafe = glm::clamp(bothBoundaries, glm::vec2(0.0f), glm::vec2(maxBoundary) - glm::vec2(std::numeric_limits<float>::epsilon()));

    return bothBoundaries;
}

glm::vec4 boundaryPoints(const float i, const float j, const int imageWidth, const int imageHeight)
{
    glm::vec2 iAxis = calculateBoundary(i, imageWidth);
    glm::vec2 jAxis = calculateBoundary(j, imageHeight);
    return glm::vec4 { iAxis.x, iAxis.y, jAxis.x, jAxis.y };
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
    glm::vec2 texCoordSafe = glm::clamp(texCoord, glm::vec2(0.0f), glm::vec2(1.0f) - glm::vec2(std::numeric_limits<float>::epsilon()));

    float i = image.width * texCoordSafe.x;
    float j = image.height * (1.0f - texCoordSafe.y);

    glm::vec4 c = boundaryPoints(i, j, image.width, image.height);
    // lower x boundary     - c.x
    // higher x boundary    - c.y
    // lower y boundary     - c.z
    // higher y boundary    - c.w
    // lower, lower is (0, 0) -> upper left corner

    float alpha = i - c.x;
    float beta = c.w - j;

    // int index = j * image.width + i;
    int indexBL = static_cast<int>(std::floor(c.w)) * image.width + static_cast<int>(std::floor(c.x));
    glm::vec3 bottomLeft = image.pixels[indexBL];
    int indexBR = static_cast<int>(std::floor(c.w)) * image.width + static_cast<int>(std::floor(c.y));
    glm::vec3 bottomRight = image.pixels[indexBR];
    int indexUL = static_cast<int>(std::floor(c.z)) * image.width + static_cast<int>(std::floor(c.x));
    glm::vec3 upperLeft = image.pixels[indexUL];
    int indexUR = static_cast<int>(std::floor(c.z)) * image.width + static_cast<int>(std::floor(c.y));
    glm::vec3 upperRight = image.pixels[indexUR];

    glm::vec3 finalColor = 
        (1 - alpha) * (1 - beta) * bottomLeft + 
        alpha * (1 - beta) * bottomRight + 
        (1 - alpha) * beta * upperLeft + 
        alpha * beta * upperRight;

    return finalColor;
}