import cv2
import numpy as np

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15

def alignImages(im1, im2):
    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors
    sift = cv2.SIFT_create(MAX_FEATURES)
    keypoints1, descriptors1 = sift.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = sift.detectAndCompute(im2Gray, None)

    # Match features
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, _ = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography to warp image
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg, h


if __name__ == '__main__':
    # Read reference image
    refFilename = "form.jpg"
    print("Reading reference image :", refFilename)
    imReference = cv2.imread(refFilename)

    # Read image to be aligned
    imFilename = "scanned-form.jpg"
    print("Reading image to align :", imFilename)
    im = cv2.imread(imFilename)

    # Align images
    print("Aligning images...")
    imReg, h = alignImages(im, imReference)

    # Write aligned image to disk
    outFilename = "aligned.jpg"
    print("Saving aligned image :", outFilename)
    cv2.imwrite(outFilename, imReg)

    # Print estimated homography
    print("Estimated homography :\n", h)