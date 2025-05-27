import numpy as np
import matplotlib.pyplot as plt


class Checker:
    def __init__(self, resolution: int, tile_size: int):
        self.resolution = resolution
        self.tile_size = tile_size
        self.output = None

    def draw(self):
        if (self.resolution % (2 * self.tile_size) != 0):
            print('resolution can\'t divide the 2*tile_size')
            return np.zeros((self.resolution, self.resolustion))
        x, y = np.indices((self.resolution, self.resolution))
        x_tile = x // self.tile_size
        y_tile = y // self.tile_size
        self.output = (x_tile + y_tile) % 2;
        return self.output.copy()

    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.axis('off')
        plt.tight_layout()  # 调整布局，使棋盘充满整个画布
        plt.show()


class Circle:
    def __init__(self, resolution: int, radius: int, position: tuple):
        self.resolution = resolution;
        self.radius = radius;
        self.position = position;
        self.output = None;

    def draw(self):
        if (self.position[0] > self.resolution | self.position[1] > self.resolution):
            print('beyond size!')
            self.output = np.zeros((self.resolution, self.resolution))
        else:
            self.output = np.zeros((self.resolution, self.resolution))
            # x, y = np.meshgrid(np.arange(self.resolution), np.arange(self.resolution))
            y ,x = np.indices((self.resolution, self.resolution))
            distance = np.sqrt((x - self.position[0]) ** 2 + (y - self.position[1]) ** 2)
            self.output[distance <= self.radius] = 1

        return self.output.copy()

    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.axis('off')
        plt.show()


class Spectrum:
    def __init__(self, resolution):
        self.resolution = resolution
        self.output = None

    def draw(self):
        red_channel = np.linspace(0, 1, self.resolution)
        green_channel = np.linspace(0, 1, self.resolution)
        blue_channel = np.linspace(0, 1, self.resolution)

        # red, green = np.meshgrid(red_channel, green_channel)
        #
        spectrum = np.zeros((self.resolution, self.resolution, 3))
        # spectrum[:, :, 0] = red
        # spectrum[:, :, 1] = green


        # print(spectrum)
        spectrum[:, :, 0] = red_channel[np.newaxis, :]
        # print('----------red')
        # print(spectrum)
        spectrum[:, :, 1] = green_channel[:, np.newaxis]
        # # print('---------green')
        # print(spectrum)
        spectrum[:, :, 2] = np.abs(blue_channel[np.newaxis, :] - 1)
        # print('------blue')
        # print(spectrum)

        self.output = spectrum;

        return self.output.copy()

    def show(self):
        plt.imshow(self.output)
        plt.axis('off')
        plt.show()


# # Main script
if __name__ == "__main__":
    # Test Checker class
    # resolution = 20
    # tile_size = 2
    # checker = Checker(resolution, tile_size)
    # checker.draw()
    # checker.show()
    # circle = Circle(resolution=100, radius=25, position=(50, 40))
    # circle.draw()
    # circle.show()
    spectrum = Spectrum(4)
    spectrum.draw()
    spectrum.show()

