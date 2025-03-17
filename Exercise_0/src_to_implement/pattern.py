import numpy as np
import matplotlib.pyplot as plt


#Hints:
#init () is the constructor of the class. Following functions from the cheat sheet might be
#useful: np.tile(), np.arange(), np.zeros(), np.ones(), np.concatenate() and np.expand dims()


class Checker:
    def __init__(self, resolution, tile_size):
        self._resolution = resolution  # no of pixels in each dimension
        self._tile_size = tile_size  # no of pixels for an individual tile
        self.output = None

    def draw(self):
        denominator = 2 * self._tile_size

        if self._resolution % denominator != 0:
            raise ValueError("Resolution is not evenly divisible by the tile_size")

        total_tiles = self._resolution // self._tile_size
        repetition_along_dimension = total_tiles // 2

        black_tile = np.zeros((self._tile_size, self._tile_size))
        white_tile = np.ones((self._tile_size, self._tile_size))

        black_white_tile = np.concatenate((black_tile, white_tile), axis=1)
        white_black_tile = np.concatenate((white_tile, black_tile), axis=1)

        black_white_pattern_row = np.tile(black_white_tile, (1, repetition_along_dimension))
        white_black_pattern_row = np.tile(white_black_tile, (1, repetition_along_dimension))

        vertical_stacked_pattern = np.concatenate((black_white_pattern_row, white_black_pattern_row), axis=0)

        checkerboard_pattern = np.tile(vertical_stacked_pattern, (repetition_along_dimension, 1))
        self.output = checkerboard_pattern
        return self.output.copy()

    def show(self):
        plt.axis('off')
        plt.imshow(self.output, cmap='gray')
        plt.show()


class Circle:
    def __init__(self, resolution, radius, position):
        self._resolution = resolution
        self._radius = radius
        self._position = position
        self.output = None

    def draw(self):
        x = np.arange(0, self._resolution)
        y = np.arange(0, self._resolution)

        x_coord, y_coord = np.meshgrid(x, y)

        x_position_squared = (x_coord - self._position[0]) ** 2
        y_position_squared = (y_coord - self._position[1]) ** 2

        distance_from_circle_center = np.sqrt(x_position_squared + y_position_squared)

        self.output = distance_from_circle_center
        self.output[self.output <= self._radius] = 1
        self.output[self.output > self._radius] = 0
        return self.output.copy()

    def show(self):
        plt.axis('off')
        plt.imshow(self.output, cmap='gray')
        plt.show()


class Spectrum:
    def __init__(self, resolution):
        self._resolution = resolution
        self.output = None

    def draw(self):
        red_gradient = np.linspace(0, 1, self._resolution)
        green_gradient = np.linspace(0, 1, self._resolution)
        blue_gradient = np.linspace(1, 0, self._resolution)

        tiled_red_gradient = np.tile(red_gradient, (self._resolution, 1))
        tiled_green_gradient = np.transpose(
            np.tile(green_gradient, (self._resolution, 1)))  # green color increases from top to bottom
        tiled_blue_gradient = np.tile(blue_gradient, (self._resolution, 1))

        red_gradient_3D_array = np.expand_dims(tiled_red_gradient, axis=2)
        green_gradient_3D_array = np.expand_dims(tiled_green_gradient, axis=2)
        blue_gradient_3D_array = np.expand_dims(tiled_blue_gradient, axis=2)

        self.output = np.concatenate((red_gradient_3D_array, green_gradient_3D_array, blue_gradient_3D_array), axis=2)
        return self.output.copy()

    def show(self):
        if self.output is not None:
            plt.imshow(self.output)
            plt.axis('off')  # Optional: turns off axis labels
            plt.show()

# checker = Checker(200, 20)
# checker.draw()
# checker.show()
#
# circle = Circle(200, 20, (100, 100))
# circle.draw()
# circle.show()
#
# spectrum = Spectrum(200)
# spectrum.draw()
# spectrum.show()
