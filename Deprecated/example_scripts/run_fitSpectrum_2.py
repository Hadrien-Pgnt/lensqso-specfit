import numpy
#import agn_spec_features
import pickle
import initParams_fitSpectra
##Emission Lines--
## Variable widths.

import fitSpectrum



## num lines

### if variable width, num widths = 4
### if variable amplitude, num amplitudes = 4
### gauss hermite polynomials.


##agn_pars_dictionary

##global line parameters


###(possibly) variable line parameters
###Iron parameters



ofiName = 'specPosteriors/ABCD_HB5_DHBW_Fe_OIII3_reinit_1'
#specClass_loaded.runMC(1)

specClass_loaded = pickle.load(open('%s.pickle'%ofiName,'rb'))


#print(specClass_loaded.outChains.shape)
'''specClass_loaded.startChains = specClass_loaded.outChains[:,-1,:]



#print(specClass_loaded.outChains[::5,-1,:].shape)


#specClass_loaded.writeFluxRatios(inputChains=specClass_loaded.outChains[::5,-1,:])

bestFit =  specClass_loaded.outChains[:,-1,:][specClass_loaded.probabilities[:,-1]
                                              == specClass_loaded.probabilities[:,-1].max()][0]

specs,specSums,specNames = specClass_loaded.getChainComponents(useChains = numpy.array([bestFit]))
print(specs.shape)

specClass_loaded.plotVals(specs[0],useDat = True, labels = specNames, ylim = [-6,40],xlim = [4600,5150])

specClass_loaded.writeFluxRatios(inputChains=specClass_loaded.outChains[::5,-1,:])

'''

specClass_loaded.ofiName = ofiName.replace('_1','_2')



specClass_loaded.runMC(1e4)

#print(specClass_loaded.outChains.shape)

pickle.dump(specClass_loaded,open('%s.pickle'%specClass_loaded.ofiName,'wb'))

'''


bestFit =  specClass_loaded.outChains[:,-1,:][specClass_loaded.probabilities[:,-1]
                                               == specClass_loaded.probabilities[:,-1].max()][0]


specs,specSums = specClass_loaded.getChainComponents(useChains = numpy.array([bestFit]))
print(specs.shape)

specClass_loaded.plotVals(specs[0], useDat = True)

'''
'''chains,probs,parnames = pickle.load(open(specClass_loaded.ofiName+'.pickle','rb'))



#print(chains.shape)


specClass_loaded.outChains = chains
specClass_loaded.probabilities= probs


pickle.dump(specClass_loaded, open('%s_classInit.pickle'%ofiName,'wb'))


bestFit =  chains[:,-1,:][probs[:,-1] == probs[:,-1].max()][0]
'''

'''bestFit =  specClass_loaded.outChains[:,-1,:][specClass_loaded.probabilities[:,-1]
                                              == specClass_loaded.probabilities[:,-1].max()][0]


specs,specSums = specClass_loaded.getChainComponents(useChains = numpy.array([bestFit]))
print(specs.shape)

specClass_loaded.plotVals(specs[0],useDat = True)
'''


#specClass_loaded.plotVals(lamRest, bestFit,dats,sigs,agnSpecs)


#specClass_loaded.writeFluxRatios(inputChains=chains[::5,-1,:])




#specClass_loaded.writeFluxRatios(chains[::5,-1,:],agnSpecs,oFiName,probs[:,-1].max(),list(agnCompsInitDict.keys()))
#outchains,outprobs = specClass_loaded.runMC(niter)
#specClass_loaded.writeMC()




#chains,probs,parnames = pickle.load(open(oFiName,'rb'))
#startChains = chains[:,-1,:]
#oFiName = 'specPosteriors/0608_ABCD_HB5_dHBW_OIII1.pickle'

#mcInfo = [startChains, parnames, nwalkers, niter,priors,dats,sigs,agnSpecs]


#chains, probs = agnSpectrumClass.runMC(mcInfo)


#pickle.dump([chains,probs,parnames], open(oFiName,'wb'))


#oFiName = 'specPosteriors/0608_ABCD_HB5_OIII1.pickle'

#chains,probs,parnames = pickle.load(open(oFiName,'rb'))
#oFiName = 'specPosteriors/0608_AOnly_HB5_OIII3_Hbnarrow_2.pickle'

#chains = chains[::5,-2,:]
#specs = agnSpectrumClass.getChainComponents(chains,agnSpecs)

#chains,probs,parnames = pickle.load(open(oFiName,'rb'))

'''print('logp best')

print(probs[:,-1].max())

bestFit =  chains[:,-1,:][probs[:,-1] == probs[:,-1].max()]


print(parnames)
print(bestFit)

i=0
for name in parnames:
    print(name)
    print(bestFit[0][i])
    i+=1





#oFiName = 'specPosteriors/0608_AOnly_HB5_OIII3_Hbnarrow_2.pickle'
#nwalkers, #niter, #nparams
#chains = chains[::5,-1,:]

#print(chains.shape)

#specs,sums = agnSpectrumClass.getChainComponents(bestFit,agnSpecs)
#specs,sums = agnSpectrumClass.getChainComponents(chains[::5,-1,:],agnSpecs)

agnSpectrumClass.writeFluxRatios(chains[::5,-1,:],agnSpecs,oFiName,probs[:,-1].max(),list(agnCompsInitDict.keys()))

#print(specs.shape)
#print('sums')
#print(sums)
#print('tparnames')
#print(tparnames)
#print(banana)
#print('specs')
#print(specs.shape)
#averageFluxRatios, stdFluxRatios, averageFluxes,stdFluxes = agnSpectrumClass.calcFluxRatios(chains[::5,-1,:],agnSpecs)
#print(averageFluxRatios,stdFluxRatios)
#print(banana)

#print(banana)

#print(parnames)
#print(parnames.shape)

#print(lamRest.shape)
import pylab
#parnames = ['total']+parnames
print(specA.keys())
tparnames = ['total'] + list(specA.keys())


j=0
for image in agnSpecs.keys():
    pylab.errorbar(lamRest, dats[j],yerr = sigs[j], fmt = 'k.')
    i = 0

    for lineName in tparnames:


        pylab.plot(lamRest, specs[0,j,i,:].T, label = lineName)
        i+=1
    j+=1





    pylab.legend()
    pylab.show()
'''




'''print(specs)
print(specs.shape)
import pylab
pylab.plot(lamRest,specs[:,0,2,:].T)
pylab.show()
'''

'''totalFluxes = numpy.sum(specs[:,0,2,:],axis = -1)
print(totalFluxes.shape)
print(totalFluxes.mean())
print(totalFluxes.std())


totalFluxes = numpy.sum(specs[:,1,2,:],axis = -1)
print(totalFluxes.shape)
print(totalFluxes.mean())
print(totalFluxes.std())


totalFluxes = numpy.sum(specs[:,2,2,:],axis = -1)
print(totalFluxes.shape)
print(totalFluxes.mean())
print(totalFluxes.std())


totalFluxes = numpy.sum(specs[:,3,2,:],axis = -1)
print(totalFluxes.shape)
print(totalFluxes.mean())
print(totalFluxes.std())



totalFluxes = numpy.sum(specs[:,0,1,:],axis = -1)
print(totalFluxes.shape)
print(totalFluxes.mean())
print(totalFluxes.std())


totalFluxes = numpy.sum(specs[:,1,1,:],axis = -1)
print(totalFluxes.shape)
print(totalFluxes.mean())
print(totalFluxes.std())


totalFluxes = numpy.sum(specs[:,2,1,:],axis = -1)
print(totalFluxes.shape)
print(totalFluxes.mean())
print(totalFluxes.std())


totalFluxes = numpy.sum(specs[:,3,1,:],axis = -1)
print(totalFluxes.shape)
print(totalFluxes.mean())
print(totalFluxes.std())




#oFiName = 'specPosteriors/0608_AOnly_2.pickle'
'''
'''startChains = chains[:,-1,:]

pickle.dump([chains,probs,parnames], open(oFiName,'wb'))

bestFit =  chains[:,-1,:][probs[:,-1] == probs[:,-1].max()][0]
'''


'''mcInfo = [startChains, parnames, nwalkers, niter,priors,dats,sigs]
chains, probs = agnSpectrumClass.runMC(mcInfo)
'''


'''pickle.dump([chains,probs,parnames], open(oFiName,'wb'))
'''

'''bestFit =  chains[:,-1,:][probs[:,-1] == probs[:,-1].max()][0]
specs,sums = agnSpectrumClass.getChainComponents(bestFit,agnSpecs)

agnSpectrumClass.plotVals(bestFit,dats,sigs)
i=0
for name in parnames:
    dat = chains[:, :, i]

    print(name)
    print(dat[:, :].mean())
    print(dat[:, :].std())

#    pylab.plot(dat[:, :].T, '-', alpha=0.2)
#    pylab.ylabel(name)
#    pylab.show()
    i+=1
'''
