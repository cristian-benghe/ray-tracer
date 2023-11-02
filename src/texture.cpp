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

    float shiftI = i - std::floor(i);
    float shiftJ = j - std::floor(j);
    
    float gradientI;
    float gradientJ;
    
    if (shiftI > 0.5f)
        gradientI = shiftI - 0.5f;
    else 
        gradientI = shiftI + 0.5f;
    
    if (shiftJ > 0.5f) 
        gradientJ = shiftJ - 0.5f;
    else 
        gradientJ = shiftJ + 0.5f;

    int centralI = 0;
    int centralJ = 0;
    if (shiftI <= 0.5f && shiftJ <= 0.5f) {
        centralI = std::floor(i);
        centralJ = std::floor(j);
    }
    if (shiftI > 0.5f && shiftJ < 0.5f) {
        centralI = std::floor(i) + 1;
        centralJ = std::floor(j);
    }
    if (shiftI < 0.5f && shiftJ > 0.5f) {
        centralI = std::floor(i);
        centralJ = std::floor(j) + 1;
    }
    if (shiftI > 0.5f && shiftJ > 0.5f) {
        centralI = std::floor(i) + 1;
        centralJ = std::floor(j) + 1;
    }

    glm::vec2 upperLeft = { glm::clamp(centralI - 1, 0, image.width - 1), glm::clamp(centralJ - 1, 0, image.height - 1) };
    glm::vec2 upperRight = { glm::clamp(centralI, 0, image.width - 1), glm::clamp(centralJ - 1, 0, image.height - 1) };
    glm::vec2 lowerLeft = { glm::clamp(centralI - 1, 0, image.width - 1), glm::clamp(centralJ, 0, image.height - 1) };
    glm::vec2 lowerRight = { glm::clamp(centralI, 0, image.width - 1), glm::clamp(centralJ, 0, image.height - 1) };

    glm::vec3 colorUL = image.pixels[upperLeft.y * image.width + upperLeft.x];
    glm::vec3 colorUR = image.pixels[upperRight.y * image.width + upperRight.x];
    glm::vec3 colorLL = image.pixels[lowerLeft.y * image.width + lowerLeft.x];
    glm::vec3 colorLR = image.pixels[lowerRight.y * image.width + lowerRight.x];

    glm::vec3 interpolated = glm::mix(
        glm::mix(colorUL, colorUR, gradientI),
        glm::mix(colorLL, colorLR, gradientI),
        gradientJ );
    
    return interpolated;
}