import pickle, numpy,pylab

posDir = 'specPosteriors'

chains, probs, parnames = pickle.load(open('%s/ABCD_HB5_DHBW_Fe_OIII3_4_posterior.pickle'%posDir, 'rb'))
print(probs[:,-1].max())

pylab.plot(probs[:,:].T)
pylab.show()

boo = probs[:,-1]>-60720


i = 0
for name in parnames:
    dat = chains[:, :, i]

    print(name)

    print(dat[:,-1][boo].mean())
    print(dat[:,-1][boo].std())
    #print(dat[:, -100:].mean())
    #print(dat[:, -100:].std())

    #pylab.plot(dat[:,:].T, '-', alpha=0.2)
    #pylab.ylabel(name)
    #pylab.show()
    i+=1