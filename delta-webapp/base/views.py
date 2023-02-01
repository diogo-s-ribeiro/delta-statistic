import sys, os
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

from django.shortcuts import render
from django.http import HttpResponse

import numpy as np
import statistics, re, csv

from Bio import Phylo
from io import StringIO
from datetime import date

import delta_functions
import pastml_functions

##### Download-Files #####
def delta_txt(request):
    response = HttpResponse(content_type='text/plain')

    #Get variables
    name   = request.session.get('name')
    result = request.session.get('result')

    #File-Name
    response['Content-Disposition'] = 'attachment; filename={}_metadata.txt'.format(name)

    #File-Content
    lines = []
    for resulting_delta in result:
        lines.append(f'{resulting_delta}\n')

    response.writelines(lines)
    return response

def delta_txt_metadata(request):
    response = HttpResponse(content_type='text/plain')

    #Get variables
    name   = request.session.get('name')
    today      = request.session.get('today')
    l0         = request.session.get('l0')
    se         = request.session.get('se')
    sim        = request.session.get('sim')
    burn       = request.session.get('burn')
    thin       = request.session.get('thin')
    ent_type   = request.session.get('ent_type')

    #File-Name
    response['Content-Disposition'] = 'attachment; filename={}_metadata.txt'.format(name)

    #File-Content
    lines = [
        'Date\tL0\tSe\tSim\tBurn\tThin\tEntropyType\n',
        str(today)+'\t'+str(l0)+'\t'+str(se)+'\t'+str(sim)+'\t'+str(burn)+'\t'+str(thin)+'\t'+str(ent_type)
        ]

    response.writelines(lines)
    return response

def delta_csv(request):
    response = HttpResponse(content_type='text/csv')

    #Get variables
    name   = request.session.get('name')
    result = request.session.get('result')

    #File-Name
    response['Content-Disposition'] = 'attachment; filename={}.csv'.format(name)

    #Create CSV-Writer
    writer = csv.writer(response)
    
    #Write CSV
    #Add Column-Header
    writer.writerow( ['Delta'] )
    #Add Column-Body
    for resulting_delta in result:
        writer.writerow( [resulting_delta] )

    return response

def delta_csv_metadata(request):
    response = HttpResponse(content_type='text/csv')

    #Get variables
    name       = request.session.get('name')
    today      = request.session.get('today')
    l0         = request.session.get('l0')
    se         = request.session.get('se')
    sim        = request.session.get('sim')
    burn       = request.session.get('burn')
    thin       = request.session.get('thin')
    ent_type   = request.session.get('ent_type')

    #File-Name
    response['Content-Disposition'] = 'attachment; filename={}_metadata.csv'.format(name)

    #Create CSV-Writer
    writer = csv.writer(response)
    
    #Write CSV
    #Add Column-Header
    writer.writerow( ['Date', 'L0', 'Se', 'Sim', 'Burn', 'Thin', 'EntropyType'] )
    #Add Column-Body
    writer.writerow( [str(today), str(l0), str(se), str(sim), str(burn), str(thin), str(ent_type)] )

    return response

##### Get Delta #####
def result(request):

    #Optional variables
    ancest    = 'NA'
    tree      = 'NA'
    states    = 'NA'
    delimiter = 'NA'
    method    = 'NA'
    model     = 'NA'

    if request.method == 'POST':
        
        today     = date.today()
        name      = str(request.POST['n0'])

        doyou1    = str(request.POST['doyou1'])
        if doyou1 == 'Yes':
            ancest    = request.FILES['ancest'].read().decode('UTF-8')
        else:
            tree      = request.FILES['tree'].read().decode('UTF-8')
            states    = request.FILES['states'].read().decode('UTF-8')
            delimiter = str(request.POST['delim'])
            method    = str(request.POST['pml1'])
            model     = str(request.POST['pml2'])

        l0        = float(request.POST['d1'])
        se        = float(request.POST['d2'])
        sim       = int(request.POST['d3'])
        thin      = int(request.POST['d4'])
        burn      = int(request.POST['d5'])
        ent_type  = str(request.POST['ent1'])

    #Generic name if blank
    if len(name) == 0:
        name = 'test'

    astates = []
    if doyou1 != 'Yes':
        #Get real delimiter
        if len(delimiter) != 0:
            list_of_trees  = re.split(delimiter, tree)
        #Different "new line" for Unix/Mac OS/Windows
        else:
            list_of_trees = re.split(r'\n|\r', tree)

        list_of_trees = list(filter(None, list_of_trees))
        for the_tree in list_of_trees:
            marg     = pastml_functions.marginal(data=StringIO(states), tree=the_tree, model=model, prediction_method=method)
            astates += [marg]

        #change latter
        tree   = list_of_trees[0]
        tree   = Phylo.read(StringIO(tree), "newick")        

    else:
        # Transform csv into list
        ancest = ancest.replace("\r", "" )
        ancest = ancest.split('\n')
        #Transform into array
        arr_ancest = [arr.split(',') for arr in ancest]
        arr_ancest = arr_ancest[1:]
        arr_ancest = [list(map(float, lst_arr[1:])) for lst_arr in arr_ancest]
        #Append
        marg = np.array(arr_ancest)
        astates += [marg]


    result = []
    for x in range(len(astates)):
        number = delta_functions.delta(astates[x], l0, se, sim, thin, burn, ent_type)
        result.append([tree, round(number, 15)])

    
    number_of_results = len(result)
    average_result    = sum([res[1] for res in result])/(number_of_results)
    median_result     = statistics.median([res[1] for res in result])
    min_result        = min([res[1] for res in result])
    max_result        = max([res[1] for res in result])


    # File Download
    # Pass variables for a later download
    request.session['name']   = name
    request.session['result'] = [res[1] for res in result]
    # Meta
    request.session['today']      = str(today)
    request.session['l0']         = str(l0)
    request.session['se']         = str(se)
    request.session['sim']        = str(sim)
    request.session['burn']       = str(burn)
    request.session['thin']       = str(thin)
    request.session['ent_type']   = str(ent_type)



    return render(request, 'base/result.html', 
    {'Date': today, 'Phylo': name, 'Lambda0': l0, 'Se':se, 'Sim':sim, 'Thin':thin, 'Burn': burn, 
    'N_Result':number_of_results, 'Avg_Result':average_result, 'Med_Result':median_result, 'Min_Result':min_result, 'Max_Result':max_result,
    'Result': result, 'Tree':tree, 'States':states, 'Ancest':ancest})

##### Web-Pages #####
def home(request):
    return render(request, 'base/home.html')

def about(request):
    return render(request, 'base/about.html')

def calculate(request):
    return render(request, 'base/calculate.html')

# Help-Pages
def help(request):
    return render(request, 'base/help.html')

def h_ps(request):
    return render(request, 'base/h_ps.html')

def h_model(request):
    return render(request, 'base/h_model.html')

def h_ace(request):
    return render(request, 'base/h_ace.html')

def h_entropy(request):
    return render(request, 'base/h_entropy.html')

def h_files(request):
    return render(request, 'base/h_files.html')
