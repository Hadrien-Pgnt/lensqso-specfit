import numpy

##lineDictionary = {Fe,OIII,Hbeta,continuum}
def returnWalkers(agnSpecs,agnCompsInitDict,nwalkers):


    pars = []
    #just have a list of priors
    priors = []
    parnames = []
    startSigs =[]
    nameOrder = ['A', 'B', 'C', 'D']

    # print(self.agnSpecs)

    for specName in nameOrder:
            # print('SPEC NAME')
            # print(specName)
            # print(len(pars))
            # print(self.agnSpecs.keys())
            # print('B' in self.agnSpecs.keys())

          if specName in agnSpecs.keys():
        #firstImageBoolean=True
        #for specName in agnSpecs.keys():
            #print('specName')
            #print(specName)
            agnSpec = agnSpecs[specName]
            #imProps = image_properties[exposure]

            ## name of the feature
            line_order = ['OIII','HBeta','HBetaNarrow','Fe','continuum']


            #print('agn spec')
            #print(agnSpec)
            for line_name in line_order:
                if line_name in agnSpec.keys():

                #for line_name in agnSpec.keys():
                    #print('line name')
                    #print(line_name)
                    line_dict= agnSpec[line_name]
                    ##parameters of the feature e.g. amplitude, number of components, ect.

                    #print('line_dict')
                    #print(line_dict.keys())

                    for line_prop_name in line_dict.keys():
                        line_prop = line_dict[line_prop_name]
                        #print('line_prop')
                        #print(line_prop)
                        ##here is where I pull out the values from the first spectrum if the value is fixed, in the inference code
                        if line_prop is not 'fixed':
                            if type(line_prop) is not list:
                                #print(agnCompsInitDict)
                                #print('comps init dict line name')
                                #print(agnCompsInitDict[line_name])

                                #print('line prop')
                                #print(line_prop)
                                #print(agnCompsInitDict[line_name][line_prop_name])

                                min, max, step = agnCompsInitDict[line_name][line_prop_name]

                                # pars.extend(startVal)
                                parnames.append('%s_%s_%s' % (line_name, line_prop_name, specName))
                                priors.append([min, max])
                                startSigs.append(step)
                                pars.append(line_prop)
                                #print('pars')
                                #print(pars)
                            if type(line_prop) is list:
                                i = 0
                                for element in line_prop:
                                    min, max, step = agnCompsInitDict[line_name][line_prop_name]

                                    # pars.extend(startVal)
                                    parnames.append('%s_%s_%s_%s' % (line_name, line_prop_name, i, specName))
                                    priors.append([min, max])
                                    startSigs.append(step)
                                    pars.append(element)
                                    i+=1



        #print('priors')
        #print(priors)

        #print(bananan)

        #print(startSigs)

        #print(parnames)
        #print(priors)
        #print(pars)
    '''print('length of pars')
    print(len(pars))
    print(startSigs)
    print('nwalkers')
    print(nwalkers)
    print('pars')
    print(pars)
    print('parnames')
    print(parnames)
    '''


    #print(len(pars))
    #print(banana)
    startVals = numpy.array([numpy.array(pars) + numpy.array(startSigs) * numpy.random.randn(len(pars)) for i in range(nwalkers)])


    return startVals, numpy.array(priors),numpy.array(parnames)
