import cv2


class ImageMatcher:
    '''
    A class for matching features between two images using the SIFT algorithm.
    '''

    def __init__(self, img1_path, img2_path, save_path):
        '''
        Initializes the ImageMatcher with paths to two images.

        Parameters:
        img1_path (str): Path to the first image.
        img2_path (str): Path to the second image.
        '''
        self.img1_path = img1_path
        self.img2_path = img2_path
        self.save_path = save_path
        self.img1 = None
        self.img2 = None
        self.sift = None
        self.keypoints_1 = None
        self.descriptors_1 = None
        self.keypoints_2 = None
        self.descriptors_2 = None
        self.matches = None
        self.good_matches = None
        self.matched_img = None

    def load_images(self):
        '''
        Loads the images in grayscale mode.
        '''
        self.img1 = cv2.imread(self.img1_path, cv2.IMREAD_GRAYSCALE)
        self.img2 = cv2.imread(self.img2_path, cv2.IMREAD_GRAYSCALE)

    def detect_features(self):
        '''
        Detects keypoints and computes descriptors using SIFT.
        '''
        self.sift = cv2.SIFT_create()
        self.keypoints_1, self.descriptors_1 = self.sift.detectAndCompute(
            self.img1, None)
        self.keypoints_2, self.descriptors_2 = self.sift.detectAndCompute(
            self.img2, None)

    def match_keypoints(self):
        '''
        Matches keypoints using a FLANN matcher with ratio test.
        '''
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        self.matches = flann.knnMatch(self.descriptors_1,
                                      self.descriptors_2,
                                      k=2)

        self.good_matches = []
        for m, n in self.matches:
            if m.distance < 0.5 * n.distance:
                self.good_matches.append(m)

    def draw_and_save_matches(self):
        '''
        Draws the good matches and saves the result to disk.
        '''
        self.matched_img = cv2.drawMatches(
            self.img1,
            self.keypoints_1,
            self.img2,
            self.keypoints_2,
            self.good_matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite(self.save_path, self.matched_img)

    def process_images(self):
        '''
        Processes the images from loading to saving the matched result.
        '''
        self.load_images()
        self.detect_features()
        self.match_keypoints()
        self.draw_and_save_matches()
