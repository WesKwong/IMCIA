from imagematcher import ImageMatcher

if __name__ == '__main__':
    img1_path = './data/img1.jpg'
    img2_path = './data/img2.jpg'
    save_path = './data/matched.jpg'

    matcher = ImageMatcher(img1_path, img2_path, save_path)
    matcher.load_images()
    matcher.detect_features()
    matcher.match_keypoints()
    matcher.draw_and_save_matches()
