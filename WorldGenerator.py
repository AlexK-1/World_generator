from random import randint, random
import numpy as np
import cv2
from math import sqrt


def noise_generation(size: int, step: int, additional: int = 0, generation_type: str = "standard"):
    """
    Generates perlin noise.
    :param size: The size of the noise output in height and width in pixels.
    :param step: The number of squares in one row (pixelation level).
    :param additional: The number that will be added to each element of the output array.
    :param generation_type: Generation type ("standard" or "gradient").
    :return: array of noise
    """

    def f(x: float):
        if x < 0:
            return 0
        if x > 100:
            return 100
        else:
            return x

    noise_ = []
    line = -1
    index = -1
    for i in range(step):
        line += 1
        column = -1
        for n in range(step):
            index += 1
            column += 1
            if generation_type == "standard":
                noise_.append(f((random() * 100) + additional))
            elif generation_type == "gradient":
                r2 = 100 - abs(((line - (step / 2)) / (step / 2)) * 100)
                r = randint(int(r2 - 150), int(r2 + 150)) + additional
                r = float(str(r) + "." + str(randint(0, 100)))
                r = f(r)

                noise_.append(f(r))
    noise_ = np.array(noise_)
    noise_ = np.reshape(noise_, [step, step])
    noise_ = noise_.astype(np.uint8)

    dim = (size, size)
    noise_ = cv2.resize(noise_, dim, interpolation=cv2.INTER_AREA)
    noise_ = noise_.astype(np.uint8)
    return noise_


def perlin_noise(size: int, smooth: int = 50, steps: list = [4, 16, 32, 64, 128], generation_type: str = "standard"):
    """
    Generates perlin noise.
    :param size: The size of the output noise image in width and height in pixels.
    :param smooth: The level of smoothness of each noise from which the world will be composed
        (the smaller the number, the more "square" the world will be) (standard 50).
    :param steps: The "step" parameters of each perlin noise layer.
    :param generation_type: Generation type ("standard" or "gradient").

    :raise ValueError: If the smooth argument is less than 1.
    :raise ValueErrorㅤ: The value of the noise value is not in the power of 2.

    :return: Array of perlin noise images.
    """

    if smooth < 1:
        raise ValueError("The value of the smooth argument was introduced, which is less than 1.")

    global perlin, length
    perlin = []
    length = 0

    def make_layer(step: int):
        global perlin, length
        noise_list = noise_generation(size, step, generation_type=generation_type)
        # for i in range(50):
        if smooth > 0:
            noise_list = cv2.blur(noise_list, (smooth, smooth))

        list_ = np.reshape(noise_list, [size * size]).tolist()
        for index in range(size ** 2):
            if len(perlin) < size * size:
                perlin.append(str(list_[index]))
            else:
                perlin[index] = perlin[index] + " " + str(list_[index])
        length += 1

    for step in steps:
        make_layer(step)

    list_ = perlin.copy()
    for index in range(size ** 2):
        join_list = list(map(float, list_[index].split()))
        list_[index] = sum(join_list) / length
    list_ = np.array(list_)
    list_ = np.reshape(list_, [size, size])
    list_ = list_.astype(np.uint8)
    perlin = list_.copy()
    return np.array(perlin)


class World:
    """
    A class for generating a color map image.

    Perlin noise is used for generation.

    You can use the built-in standard_generation function with an already configured generation algorithm
    or create a generation algorithm yourself.

    To create a map, 3 perlin noise is generated: heights (landscape), temperatures, rivers and mountains.
    To create perlin noise, several noises with varying degrees of pixelation are used.
    To create a single noise, the make_layer function is used. To combine several noises into one,
    the join_layers function is used. After creating three perlin noises, the noises are combined into a single
    color image using the coloring_map function.
    """

    def __init__(self, size: int = 512, smooth: int = 50, height: int = 0, temperature: int = 0,
                 no_rivers: bool = False, no_mountains: bool = False, no_glaciers: bool = False,
                 no_deserts: bool = False):
        """
        :param size: The size of the world output in height and width in pixels.
        :param smooth: The level of smoothness of each noise from which the world will be composed
            (the smaller the number, the more "square" the world will be) (standard 50).
        :param height: The height of the world (the lower the world, the more water there is in it)
            (-7 is the ocean with islands, 0 is standard, the mainland with lakes and mountains).
        :param temperature: Temperature is a measure (the lower the temperature, the more glaciers and fewer deserts;
            the higher the temperature, the fewer glaciers and more deserts)
            (-30 - ice age, -10 -few deserts, 0 - standard, 20 - few glaciers, 70 - endless desert).
        :param no_rivers: Removes rivers from world generation.
        :param no_mountains: Removes mountains from world generation.
        :param no_glaciers: Removes glaciers from world generation.
        :param no_deserts: Removes deserts from world generation.
        """

        self.size = size
        self.smooth = smooth
        self.height = height
        self.temperature = temperature

        self.no_rivers = no_rivers
        self.no_mountains = no_mountains
        self.no_glaciers = no_glaciers
        self.no_deserts = no_deserts

        self.height_noise = []
        self.temperature_noise = []
        self.mountains_rivers_noise = []
        self.color_noise = []

        self.length = 0
        self.i = 0
        self.n = 0

    def __str__(self):
        return str(self.color_noise)

    def __repr__(self):
        return str(self.color_noise)

    def __bool__(self):
        return len(self.color_noise) > 0

    def add_layer_to_list(self, obj: list, layer_type: str):
        list_ = np.reshape(obj, [self.size * self.size]).tolist()
        for index in range(self.size ** 2):
            if layer_type == "height" or layer_type == "h":
                if len(self.height_noise) < self.size * self.size:
                    self.height_noise.append(str(list_[index]))
                else:
                    self.height_noise[index] = self.height_noise[index] + " " + str(list_[index])
            elif layer_type == "temperature" or layer_type == "t":
                if len(self.temperature_noise) < self.size * self.size:
                    self.temperature_noise.append(list_[index])
                else:
                    self.temperature_noise[index] = str(self.temperature_noise[index]) + " " + str(list_[index])
            elif layer_type == "mountains rivers" or layer_type == "m r":
                if len(self.mountains_rivers_noise) < self.size * self.size:
                    self.mountains_rivers_noise.append(list_[index])
                else:
                    self.mountains_rivers_noise[index] = str(self.mountains_rivers_noise[index]) + " " + str(
                        list_[index])
            else:
                if len(self.height_noise) < self.size * self.size:
                    self.height_noise.append(str(list_[index]))
                else:
                    self.height_noise[index] = self.height_noise[index] + " " + str(list_[index])
        self.length += 1

    def make_layer(self, step: int, layer_type: str = "height"):
        """
        Creating and adding noise to the list. It is necessary to create perlin noise.
        :param step: The number of squares in one row (pixelation level).
        :param layer_type: Layer type: height/h, temperature/t, mountains rivers/m r.
        :return: an image (gray) of a single layer of Perlin noise.
        """

        if layer_type == "height" or layer_type == "h":
            noise_list = noise_generation(self.size, step, self.height, "standard")
        elif layer_type == "mountains rivers" or layer_type == "m r":
            if (not self.no_rivers) or (not self.no_mountains):
                noise_list = noise_generation(self.size, step, 0, "standard")
            else:
                noise_list = np.zeros([self.size, self.size])
        elif layer_type == "temperature" or layer_type == "t":
            if (not self.no_deserts) or (not self.no_glaciers):
                noise_list = noise_generation(self.size, step, self.temperature, "gradient")
            else:
                noise_list = np.zeros([self.size, self.size])
        else:
            noise_list = noise_generation(self.size, step)

        if self.smooth > 0:
            noise_list = cv2.blur(noise_list, (self.smooth, self.smooth))
        self.add_layer_to_list(noise_list, layer_type)
        return noise_list

    def join_layers(self, layers_type: str = "height"):
        """
        Combine layers into perlin noise.
        :param layers_type: Perlin Noise type: height/h, temperature/t, mountains rivers/m r.
        """
        if layers_type == "height" or layers_type == "h":
            list_ = self.height_noise.copy()
        elif layers_type == "mountains rivers" or layers_type == "m r":
            list_ = self.mountains_rivers_noise.copy()
        elif layers_type == "temperature" or layers_type == "t":
            list_ = self.temperature_noise.copy()
        else:
            list_ = self.height_noise.copy()

        if ((layers_type == "mountains rivers" or layers_type == "m r") and (
                (not self.no_mountains) or (not self.no_rivers))) or (
                (layers_type == "temperature" or layers_type == "t") and (
                (not self.no_glaciers) or (not self.no_deserts))) or (layers_type == "height" or layers_type == "h"):
            for index in range(self.size ** 2):
                join_list = list(map(float, list_[index].split()))
                list_[index] = sum(join_list) / self.length
            list_ = np.array(list_)
            list_ = np.reshape(list_, [self.size, self.size])
        else:
            list_ = np.zeros([self.size, self.size])
        list_ = list_.astype(np.uint8)

        if layers_type == "height" or layers_type == "h":
            self.height_noise = list_.copy()
        elif layers_type == "mountains" or layers_type == "m r":
            self.mountains_rivers_noise = list_.copy()
        elif layers_type == "temperature" or layers_type == "t":
            self.temperature_noise = list_.copy()
        else:
            self.height_noise = list_.copy()

    def coloring_map(self):
        """
        Combines three perlin noises (heights, temperatures, mountains and rivers) into a single color image.
        :return: Array images (RGB) of the generated world.
        """
        self.height_noise = cv2.cvtColor(self.height_noise, cv2.COLOR_GRAY2RGB)
        self.temperature_noise = cv2.cvtColor(self.temperature_noise, cv2.COLOR_GRAY2RGB)
        self.mountains_rivers_noise = cv2.cvtColor(self.mountains_rivers_noise, cv2.COLOR_GRAY2RGB)

        color_noise = np.copy(self.height_noise)

        for self.i in range(self.size):
            for self.n in range(self.size):
                height_save = self.height_noise[self.i][self.n][1]
                temp_save = self.temperature_noise[self.i][self.n][1]
                mo_ri_save = self.mountains_rivers_noise[self.i][self.n][1]
                if height_save < 50:
                    color_noise[self.i][self.n] = [192, 110, 2]
                else:
                    if height_save < 53:
                        color_noise[self.i][self.n] = [0, 193, 236]
                    else:
                        if height_save < 57:
                            color_noise[self.i][self.n] = [0, 193, 3]
                        else:
                            color_noise[self.i][self.n] = [3, 165, 0]

                if not self.no_mountains:
                    mo_2 = ((mo_ri_save / 100) ** 3) * 10000
                    if mo_2 > 65:
                        color_noise[self.i][self.n] = [115, 115, 115]
                        if mo_2 > 70:
                            color_noise[self.i][self.n] = [255, 255, 255]

                if not self.no_rivers:
                    if 14 < mo_ri_save < 16:
                        color_noise[self.i][self.n] = [192, 110, 2]

                if not self.no_glaciers:
                    if temp_save < 15.2:
                        if temp_save < 13.2:
                            color_noise[self.i][self.n] = [234, 229, 215]
                        else:
                            color_noise[self.i][self.n] = [234, 220, 169]
                if not self.no_deserts:
                    if temp_save > 23:
                        if height_save > 57:
                            color_noise[self.i][self.n] = [0, 174, 218]
                        elif height_save < 50:
                            if temp_save > 26:
                                color_noise[self.i][self.n] = [0, 174, 218]
                        else:
                            color_noise[self.i][self.n] = [0, 193, 236]
        color_noise = cv2.cvtColor(color_noise, cv2.COLOR_BGR2RGB)
        self.color_noise = color_noise
        return color_noise

    def standard_generation(self):
        """
        Generates the world, saving you from having to create the generation algorithm yourself.
        :return: Array images (RGB) of the generated world.
        """
        self.make_layer(4, "h")
        self.make_layer(16, "h")
        self.make_layer(32, "h")
        self.make_layer(64, "h")
        self.make_layer(128, "h")

        self.join_layers("h")

        self.make_layer(32, "t")
        self.make_layer(64, "t")
        self.make_layer(128, "t")

        self.join_layers("t")

        self.make_layer(16, "m r")
        self.make_layer(32, "m r")
        self.make_layer(64, "m r")
        self.make_layer(128, "m r")

        self.join_layers("m r")

        return self.coloring_map()
