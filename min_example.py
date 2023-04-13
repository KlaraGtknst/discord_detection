from src.SAX import SAX
from src.HOT_SAX import HOTSAX
import numpy as np

if __name__ == "__main__":

    # functions to test code
    def try_SAX():
        # Instantiate the class
        sax = SAX(symbols=4, paa_length=3)

        # example
        dat = np.array([-2, 0, 2, 0, -1])
        sax.fit(X=dat)
        symbols = sax.transform(X=dat, return_as='symbol')

        print('symbols: ', symbols)
        print('reconstructed values from symbols: ', sax.reconstruct(S=symbols, reconstruct_from='symbol'))

        symbols = sax.transform(X=dat, return_as='idx')
        print('integers: ', symbols)

    def try_HOT_SAX():
        # Instantiate the class
        sax = SAX(symbols=4, paa_length=3)
        hot_sax = HOTSAX(sax=sax, window_size=2, number_of_discords=1)

        # example
        dat = np.array([-2, 0, 2, 0, -1, 3, 0, 0, 2, -1, 1, 2])
        #sax.fit(X=dat)
        Y = hot_sax.compare_pairwise(X=dat)

        print('solution with window size 2:', Y)

        hot_sax = HOTSAX(sax=sax, window_size=1, number_of_discords=1)
        Y = hot_sax.compare_pairwise(X=dat)

        print('solution with window size 1:', Y)


    # test code (calling functions)
    #try_SAX()
    #try_HOT_SAX()

    # Instantiate the class
    sax = SAX(symbols=4, paa_length=3)
    hot_sax = HOTSAX(sax=sax, window_size=2, number_of_discords=3)
    dat = np.array([-2, 0, 2, 0, -1, 3, 0, 0, 2, -1, 1, 2])
    #sax.fit(X=dat)
    hot_sax.identify_discord(dat)

