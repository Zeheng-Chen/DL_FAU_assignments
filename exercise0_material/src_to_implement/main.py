from pattern import Checker,Circle,Spectrum

def main():
    checker = Checker(1000,100)
    checker.draw()
    checker.show()


    circle = Circle(resolution=800, radius=200, position=(400, 400))
    circle.draw()
    circle.show()

    spectrum = Spectrum(resolution=800)
    spectrum.draw()
    spectrum.show()

if __name__ == "__main__":
    main()
