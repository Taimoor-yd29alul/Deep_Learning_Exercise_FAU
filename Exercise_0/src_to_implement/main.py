from pattern import Checker, Circle, Spectrum
from generator import ImageGenerator


def main():
    checker_pattern = Checker(200, 20)
    circle_pattern = Circle(200, 20, (100, 100))
    spectrum_pattern = Spectrum(200)

    image_generator = ImageGenerator('./exercise_data/', './Labels.json', 50, [50, 50, 3],
                                     True, False, False)

    checker_pattern.draw()
    circle_pattern.draw()
    spectrum_pattern.draw()

    checker_pattern.show()
    circle_pattern.show()
    spectrum_pattern.show()
    image_generator.show()


if __name__ == "__main__":
    main()
