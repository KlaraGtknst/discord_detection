from src.SAX import SAX
from src.HOT_SAX import HOTSAX
import numpy as np

if __name__ == "__main__":
    # Instantiate the class
    sax = SAX(symbols=4, paa_length=3)
    #print(sax.paa_length)

    # example
    dat = np.array([-2, 0, 2, 0, -1])

    sax.fit(X=dat)
    symbols = sax.transform(X=dat, return_as='symbol')
    #print(symbols)

    #print(sax.reconstruct(S=symbols, reconstruct_from='symbol'))


if __name__ == "__main__":
    # Instantiate the class
    sax = SAX(symbols=4, paa_length=3)
    hot_sax = HOTSAX(sax=sax)

    # example
    dat = np.array([-2, 0, 2, 0, -1, 3, 0, 0, 2, -1, 1, 2])
    sax.fit(X=dat)

    #symbols = sax.transform(X=dat, return_as='idx')
    #print(3, symbols)
    Y = hot_sax.compare_pairwise(window_size=2, X=dat)

    print('solution:', Y)



if __name__ == "__main__":
    # TODO: get array to matrix  it window size einträgen im inneren, so das die verleichbar
    dat = np.array([-2, 0, 2, 0, -1])
    # 3 = num of splits
    dat = np.array_split(dat, 3)
    #print(np.expand_dims(dat, axis=(3,2)))
    #print(dat)
