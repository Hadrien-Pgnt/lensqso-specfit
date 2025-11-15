import numpy
import agn_spec_features
import emcee
import copy
#import initParams
##Emission Lines--
## Variable widths.



##agnSpecs is the input agn dictionary, which specifies which images/lines are being fit, the values
##get updated with each mcmc iteration

class agnspectrum:
    def __init__(self, lamRest,fixedImageName = 'A',feDictionary=None):
       #self.agnSpecs = agnSpecs

       ###stores the values for this iteration of the mcmc
       #self.agnSpecsOrig = agnSpecs.copy()

       #self.agnSpecs = agnSpecs


       self.fixedImageName = fixedImageName
       self.lamRest = lamRest
       self.feDictionary = feDictionary
        ###where to load in the FEs?
    def checkPriors(self, pars,priors,parnames, printBad = False):
        logp = 0

        boo = numpy.any([(pars < priors[:, 0]), (pars > priors[:, 1])], axis=0)
        #print('parnames')
        #print(parnames)
        #print('boo')
        #print(boo)
        if numpy.any(boo):

            # print(pars)
            # print(parnames)
            # print(boo)
            # print(parnames)
            if printBad:
                print(parnames[boo])
                print(pars[boo])
                print(priors[boo])

            # print(banana)

            logp = -1e200
        return logp


    ### takes the drawn waker of parameters and separates them into the different spectral features, depending on the input dictionary values
    def divideParameters(self, pars,agnSpecsMC,parnames = False):

        par_counter = 0
        nameOrder = ['A','B','C','D']
        ospecs = copy.deepcopy(agnSpecsMC)

        for specName in nameOrder:




            if specName in agnSpecsMC.keys():

                line_order = ['OIII','HBeta','HBetaNarrow','Fe','continuum']

                for line_name in line_order:
                    if line_name in agnSpecsMC[specName].keys():

                        ## replace the dictionary value with this value. Pass it back, then iterate through
                        ##and call the appropriate agn function for each dictionary key. OR just assume that things will come
                        ##in a specific order, which is ok.


                        for line_prop_name in agnSpecsMC[specName][line_name].keys():

                            start_line_prop_val = agnSpecsMC[specName][line_name][line_prop_name]



                            if start_line_prop_val != 'fixed':

                                if type(start_line_prop_val) is not list:
                                    #line_dict[line_prop_name] = pars[i]
                                    ospecs[specName][line_name][line_prop_name] = pars[par_counter]
                                    par_counter+=1

                                if type(start_line_prop_val) is list:
                                    olist = []
                                    for elem in start_line_prop_val:
                                        olist.append(pars[par_counter])
                                        par_counter+=1


                                    ospecs[specName][line_name][line_prop_name] = olist

                            if start_line_prop_val == 'fixed':
                                ospecs[specName][line_name][line_prop_name] = ospecs[self.fixedImageName][line_name][line_prop_name]


        return ospecs


        ##spec pars dictionary is the dictionary associated with the image, after it's been updated in divide parameters.
    def simulateSpectrum(self, specParsDictionary):
            ospec = numpy.zeros(self.lamRest.size)
            #print('specParsDictionary')
            #print(specParsDictionary)
            for line_name in specParsDictionary:

                if line_name =='HBeta':

                    hbeta = agn_spec_features.makeHermite(self.lamRest,specParsDictionary[line_name],4863)


                    ospec+=hbeta

                if line_name == 'OIII':
                    oIII1 = agn_spec_features.makeHermite(self.lamRest,specParsDictionary[line_name],5008)
                    oIII2 = agn_spec_features.makeHermite(self.lamRest, specParsDictionary[line_name], 4960)

                    oIII = oIII1+oIII2/3.

                    ospec+= oIII

                if line_name == 'HBetaNarrow':
                    ###same as OIII except position and amplitude
                    #print(specParsDictionary['OIII'])
                    thbetanarrow = specParsDictionary['OIII'].copy()
                    #print(thbetanarrow)

                    ###a percentage of the OIII flux
                    thbetanarrow['amp'] = specParsDictionary[line_name]['amp']*thbetanarrow['amp']
                    #print(thbetanarrow)
                    #print(banana)
                    hbetanarrow = agn_spec_features.makeHermite(self.lamRest, thbetanarrow, 4863)

                    #oIII = oIII1 + oIII2 / 3.

                    ospec += hbetanarrow


                if line_name =='continuum':

                    continuum = agn_spec_features.makeContinuumLine(self.lamRest,specParsDictionary[line_name],5012)
                    ospec+=continuum

                if line_name == 'Fe':
                    fe = agn_spec_features.makeIron(self.lamRest,specParsDictionary[line_name],self.feDictionary)
                    ospec+=fe
            #print(banana)
            return ospec





    ##parLists = [AGNPars1, AGNPars2, AGNPars3..]
    def getProbability(self,all_pars,priors,parnames, dats,sigs,agnSpecsMC):
        #print('all_pars')
        #print(all_pars)
        logp = self.checkPriors(all_pars,priors,parnames)

        if logp>-1e200:


            agnSpecs = self.divideParameters(all_pars,agnSpecsMC)
            i= 0
            #import pylab
            #print(dats.shape)
            for agn_pars_keys in agnSpecs:
                spec = self.simulateSpectrum(agnSpecs[agn_pars_keys])
                #import pylab
                #pylab.plot(self.lamRest,spec)
                #pylab.plot(self.lamRest,dats[i])
                #pylab.show()

                #print(banana)
                tlogp = -(spec - dats[i])**2/sigs[i]**2
                logp+= tlogp.sum()
                i+=1
            #print(banana)
            #pylab.show()
        #print(logp)
        return logp



    def runMC(self,mcInfo):
        startVals, parnames, nwalkers, niter,priors,data,sigs,agnSpecsMC = mcInfo
        # self.startVals = startVals

        sampler = emcee.EnsembleSampler(nwalkers, len(parnames), self.getProbability,args=[priors,parnames,data,sigs,agnSpecsMC])
        sampler.run_mcmc(startVals, niter)

        return sampler.chain, sampler.lnprobability


    ##speclist comes from the output of return spec components, could add in component list later.
    def plotVals(self, lams, specList,dat = None,sigs=None, names = None):
        import pylab
        for i in range(specList.shape[0]):
            pylab.plot(lams,specList[i])

        if dat is not None:
            if sigs is not None:
                pylab.errorbar(lams,dat,yerr = sigs,fmt = 'k.')
            if sigs is None:
                pylab.plot(lams,dat, 'k')
        pylab.show()



    '''def plotVals(self,all_pars, dats,sigs,agnSpecsMC):
        #startVals, parnames, nwalkers, niter,priors,data,sigs = mcInfo
        import pylab
        agn_par_lists = self.divideParameters(all_pars,agnSpecsMC)
        i = 0
        # import pylab
        # print(dats.shape)
        for agn_pars_key in agn_par_lists:
            spec = self.simulateSpectrum(agn_par_lists[agn_pars_key])





            pylab.plot(self.lamRest,dats[i])
            pylab.plot(self.lamRest,spec)

            #pylab.show()

            tagn_par_list = agn_par_lists[agn_pars_key].copy()

            for element in tagn_par_list:
                tdict = {element:tagn_par_list[element]}
                if element=='HBetaNarrow':
                    tdict={element:tagn_par_list[element],'OIII':tagn_par_list['OIII']}



                print(tdict)
                spec = self.simulateSpectrum(tdict)
                pylab.plot(self.lamRest,spec)
            pylab.show()


            # print(banana)
            #tlogp = -(spec - dats[i]) ** 2 / sigs[i] ** 2
            #logp += tlogp.sum()
            i += 1
    '''

    ##takes an input MCMC chain, returns a single output spectrum split by component. Note this is the same as the previous
    ## plotting function, but it returns arrays rather than plotting.

    def returnSpecComponents(self,all_pars, agnSpecsMC):
        #startVals, parnames, nwalkers, niter,priors,data,sigs = mcInfo
        #import pylab
        agn_par_lists = self.divideParameters(all_pars,agnSpecsMC)
        #i = 0
        ospecs = []
        ##for each image

        #print(agn_par_lists)

        for agn_pars_key in agn_par_lists:
            spec = self.simulateSpectrum(agn_par_lists[agn_pars_key])
            ###one for each image.
            tospeclist = []
            #ospecs.append(spec)
            tospeclist.append(spec)
            tagn_par_list = agn_par_lists[agn_pars_key].copy()
            ## for each spectral feature for each image
            for element in tagn_par_list:
                tdict = {element:tagn_par_list[element]}
                if element=='HBetaNarrow':
                    tdict={element:tagn_par_list[element],'OIII':tagn_par_list['OIII']}

                #print(tdict)
                spec = self.simulateSpectrum(tdict)
                #ospecs.append(spec)
                tospeclist.append(spec)

            ospecs.append(tospeclist)

        ## for each chain: for each image: a spectrum for each spectral element + the total spectrum.


        return ospecs




    ###this one calls return spec components for many MCMC chains and generates an NxMxlambda array,
    ##where M is the number of spec components, lambda is the wavelength, and N is the number of draws
    # chain number, iteration number, chain element
    # dat = chains[:, :, i]
    ##make the input just a flattened list of chains (chain number, parameters)
    def getChainComponents(self,chains, agnSpecsMC):
        allcomponents = []

        for chain in chains:
            specs = self.returnSpecComponents(chain, agnSpecsMC)
            allcomponents.append(specs)


        return numpy.array(allcomponents)










