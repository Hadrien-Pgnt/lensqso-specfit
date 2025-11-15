import numpy
#import agn_spec_features
import pickle
import initParams_fitSpectra
##Emission Lines--
## Variable widths.
import fitSpectrum

feDictionary = {}

feNames = ['F', 'G', 'IZw1', 'S']
feDir = 'FeII_template_4000_5500'
###0259 doesn't have prominent iron.
feDictionary = {}
#OiiiLoc = 5007
#dxFe = 5
#modVelocity = 1310
for feName in feNames:
    modDict = pickle.load(open('/home/anierenberg/repositories/OSIRIS_QUADS/fitSpectra/FeII_template_4000_5500/%s_modDict_2.pickle' % (feName), 'rb'))
    feDictionary[feName] = modDict


OIIILineInits = {'amp':[0,5000,0.01],'width':[1,30,0.01],'compAmps':[-10,10,0.0001],'dlam':[-40,40,0.01]}
startOIIIAmp = 1.48

OIII_A = {'amp':574, 'width':13.9,    'compAmps':[-0.32,-0.84,0.33],'dlam':-25}

OIII_B = {'amp':463, 'width':'fixed','compAmps':'fixed','dlam':'fixed'}
OIII_C = {'amp':165, 'width':'fixed','compAmps':'fixed','dlam':'fixed'}
OIII_D = {'amp':92, 'width':'fixed','compAmps':'fixed','dlam':'fixed'}

startHBetaAmp = 1.42
HBetaLineInits= {'amp':[0,10000,0.001],'width':[1,600,0.1],'compAmps':[-10,10,0.0001],'dlam':[-50,50,0.01]}

HBeta_A = {'amp':1065, 'width':23.8,'compAmps':[1.05,0.009,0.227,0.02,0.02],'dlam':-15.8}
HBeta_B = {'amp':781, 'width':24,'compAmps':'fixed','dlam':'fixed'}
HBeta_C = {'amp':298, 'width':23,'compAmps':'fixed','dlam':'fixed'}
HBeta_D = {'amp':189, 'width':25,'compAmps':'fixed','dlam':'fixed'}


###width, comp vals and lambda fixed to the [OIII] values, amplitude is a fraction of the OIII amplitude
#startHBetaNarrowAmp = 0.3
#startHBetaNarrowWidth = 9.3
#HBetaNarrowLineInits= {'amp':[0,1000,1e-3]}

#HBetaNarrow_A = {'amp':startHBetaNarrowAmp}
##HBetaNarrow_B = {'amp':0.3}
##HBetaNarrow_C = {'amp':0.3}
#HBetaNarrow_D = {'amp':'fixed'}

####ind=allowed to have a redshift relative to OIII
startHBetaNarrowAmp = 0.1

HBetaNarrowLineInits= {'amp':[0,10000,0.001],'width':[1,600,0.01],'compAmps':[-10,10,0.0001],'dlam':[-60,40,0.01]}


HBetaNarrow_ind_A = {'amp':startHBetaNarrowAmp,'width':4.45,'compAmps':[], 'dlam':-1.83}
HBetaNarrow_ind_B = {'amp':0.1,'width':4.45,'compAmps':'fixed', 'dlam':'fixed'}
HBetaNarrow_ind_C = {'amp':0.1,'width':4.45,'compAmps':'fixed', 'dlam':'fixed'}
HBetaNarrow_ind_D = {'amp':0.1,'width':4.45,'compAmps':'fixed', 'dlam':'fixed'}


startFeAmp = 4.8
startFeVelocity = 3698
FeLineInits = {'amp':[0,100,0.01],'velocity':[1,5990,1],'G':[0,10,0.001],'IZw1':[0,10,0.001],'S':[0,10,0.001],'dlam':[-40,40,0.01]}

##amp is the F component amplitude
FeA = {'amp': 14.3, 'velocity':startFeVelocity,'G':0.26,'IZw1':0.06,'S':9.98,'dlam':11.8}
FeB = {'amp': 8, 'velocity':'fixed','G':'fixed','IZw1':'fixed','S':'fixed','dlam':'fixed'}
FeC = {'amp': 3.4, 'velocity':'fixed','G':'fixed','IZw1':'fixed','S':'fixed','dlam':'fixed'}
FeD = {'amp': 2.4, 'velocity':'fixed','G':'fixed','IZw1':'fixed','S':'fixed','dlam':'fixed'}

startContinuumAmp = 0.04
startContinuumSlope = 4e-5


#zero = 5012
continuumInits = {'amp':[0,100,0.01],'slope':[-0.1,0.1,1e-04]}

continuumA = {'amp':14.1,'slope':0.003}
continuumB = {'amp':9.8,'slope':0.002}
continuumC = {'amp':4.3,'slope':0.001}
continuumD = {'amp':3.31,'slope':0.0004}




'''specA = {'HBeta':HBeta_A,'OIII':OIII_A,'Fe':FeA,'continuum':continuumA,'HBetaNarrow':HBetaNarrow_A}
specB = {'HBeta':HBeta_B,'OIII':OIII_B,'Fe':FeB,'continuum':continuumB, 'HBetaNarrow':HBetaNarrow_B}
specC = {'HBeta':HBeta_C,'OIII':OIII_C,'Fe':FeC,'continuum':continuumC,'HBetaNarrow':HBetaNarrow_C}
specD = {'HBeta':HBeta_D,'OIII':OIII_D,'Fe':FeD,'continuum':continuumD,'HBetaNarrow':HBetaNarrow_D}
agnCompsInitDict = {'HBeta':HBetaLineInits,'OIII':OIIILineInits,'Fe':FeLineInits,'continuum':continuumInits,'HBetaNarrow':HBetaNarrowLineInits}

'''

specA = {'HBeta':HBeta_A,'OIII':OIII_A,'Fe':FeA, 'continuum':continuumA} #'HBetaNarrow_ind':HBetaNarrow_ind_A}
specB = {'HBeta':HBeta_B,'OIII':OIII_B,'Fe':FeB, 'continuum':continuumB} #'HBetaNarrow_ind':HBetaNarrow_ind_B}
specC = {'HBeta':HBeta_C,'OIII':OIII_C,'Fe': FeC, 'continuum':continuumC} #'HBetaNarrow_ind':HBetaNarrow_ind_C}
specD = {'HBeta':HBeta_D,'OIII':OIII_D, 'Fe': FeD,'continuum':continuumD} #'HBetaNarrow_ind':HBetaNarrow_ind_D}


agnCompsInitDict = {'HBeta':HBetaLineInits,'OIII':OIIILineInits,'continuum':continuumInits, 'Fe': FeLineInits}#, 'HBetaNarrow_ind':HBetaNarrowLineInits}




agnSpecs = {'A':specA, 'B':specB, 'C':specC, 'D':specD}



###these are the mcmc parameters
nwalkers = 400
niter =1e4

startChains, priors,parnames = initParams_fitSpectra.returnWalkers(agnSpecs,agnCompsInitDict,nwalkers)



### this is where the spectrum is stored. Change it so that it reads in your spectral file. The original code here assuems there
## will be multiple files, but you will only have one.



specBase = '/home/anierenberg/repositories/OSIRIS_QUADS/mcmcPositions/2145_rotations/specs/' \
           '2145_50_880_myWeight_median_nPix11_dsmallsig_relAmp2_rotation_all_1_21_50_880_mean_'

dats = []
sigs = []

### this opens each spectroscopy file and reads the data. Note the order of the arrays it expects to readout,
# adjust this according to however you have the arrays stored
### also make sure to only pick out part of the spectrum to fit, like 4500-5500 Angstroms rest frame
for specName in agnSpecs.keys():
    lamRest,data,sig = pickle.load(open(specBase+specName+'_noise.pickle','rb'))
    dats.append(data)

    sigs.append(sig)



dats = numpy.atleast_2d(dats)
sigs = numpy.atleast_2d(sigs)
mcInfo = [startChains, parnames, nwalkers, niter,priors,dats,sigs,agnSpecs]

ofiName = 'specPosteriors/ABCD_HB5_DHBW_Fe_OIII3_reinit_1'

print('agncompsinitdict')
print(agnSpecs)
##agnSpecs is the input agn dictionary, which specifies which images/lines are being fit
agnSpectrumClass = fitSpectrum.agnspectrum(lamRest,feDictionary=feDictionary,
                                           ofiName = ofiName,startChains=startChains, parnames=parnames,
                                           nwalkers=nwalkers, priors= priors, dats=dats,sigma = sigs,
                                           agnSpecsDictionary=agnSpecs)



'''specs,specSums = agnSpectrumClass.getChainComponents(useChains = numpy.array([startChains[0]]))
print(specs.shape)

agnSpectrumClass.plotVals(specs[0],useDat = True)

###leave niter for later.
print(agnSpectrumClass.nwalkers)

print(dir(agnSpectrumClass))
'''

print(agnSpectrumClass.__dict__.keys())

pickle.dump(agnSpectrumClass, open('%s.pickle'%ofiName,'wb'))





#oFiName = 'specPosteriors/0608_ABCD_HB7_dHBW_dHBN_OIII5.pickle'
#classPickleName = oFiName.replace('.pickle','_class.pickle')

#pickle.dump(agnSpectrumClass,open(classPickleName,'wb'))

#agnSpectrumClass = pickle.load(open(classPickleName,'rb'))

#chains,probs,parnames = pickle.load(open(oFiName,'rb'))
#startChains = chains[:,-1,:]
#oFiName = 'specPosteriors/0608_ABCD_HB5_dHBW_OIII1.pickle'



#chains, probs = agnSpectrumClass.runMC(mcInfo)


#pickle.dump([chains,probs,parnames], open(oFiName,'wb'))


#oFiName = 'specPosteriors/0608_ABCD_HB5_OIII1.pickle'

#chains,probs,parnames = pickle.load(open(oFiName,'rb'))
#oFiName = 'specPosteriors/0608_AOnly_HB5_OIII3_Hbnarrow_2.pickle'

#chains = chains[::5,-2,:]
#specs = agnSpectrumClass.getChainComponents(chains,agnSpecs)

'''chains,probs,parnames = pickle.load(open(oFiName,'rb'))

print('logp best')

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
