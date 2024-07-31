import gurobipy
from gurobipy import quicksum
from gurobipy import GRB
from gurobipy import Model

import pandas as pd
import numpy as np
import random, itertools, os, json, ast, time
from json import loads



def load_json_instance(target_path, target_file):
    with open(os.path.join(target_path, target_file), 'r') as json_file:
        data = json.load(json_file)

    data['disdur_dict'] = loads(data['disdur_dict'])
    data['disdur_dict'] = {ast.literal_eval(k): v for k, v in data['disdur_dict'].items()}

    return data

def load_rl_solution(file_path):
    with open(file_path, 'r') as file:
        dati = json.load(file)
    return dati

def create_params(NF,NC,ND,NV,disdur):

    np.random.seed(0)
    random.seed((0))

    # corresponding sets:
    F = list(range(NF))  # facilities
    C = list(range(NF, NC + NF)) #clients
    D = list(range(NC + NF, ND + NC + NF)) #deposits
    N = F + C + D
    V = list(range(NV)) #vehicles

    generic_params = {'F': F, 'C': C, 'D': D, 'N': N, 'V': V, 'NF': NF, 'NC':NC, 'ND': ND, 'NV': NV}

    # SP specific parameters

    realistic_capacity = pd.read_excel('BOP_realistic_instance_v2.xlsx', sheet_name='facility').astype({"fraction_cap_daily M": float}, errors='raise')['fraction_cap_daily M'].to_list()
    capacity_dict = { i : realistic_capacity[i] for i in range(0, len(realistic_capacity))}
    H = [1, 2, 3]  # 1 = S, 2 = M, 3 = L
    CAPH = {1: 0.7, 2: 1, 3: 1.3}  # 1 reduces cap of 30%, 2 set its default cap, 3 enlarge cap of 30%
    capf = {(j, h): round(capacity_dict[j]*CAPH[h],2) for (j,h) in itertools.product(F, H)}

    SP_params = {'H':H, 'CAPH': CAPH, 'capf': capf}

    #########################
    # OP specific parameters

    realistic_demand = pd.read_excel('BOP_realistic_instance_v2.xlsx', sheet_name='client').astype({"demand_weekly": int}, errors='raise')['demand_weekly'].to_list()
    d = {NF + i: realistic_demand[i] for i in range(0, len(realistic_demand))}
    cv = {l: 40 for l in V}
    T = 5*60 #5 ore, il tempo è in minuti
    t = {(a, b): disdur[(a, b)]['duration'] for (a, b) in itertools.product(N, N)}
    ######################################################
    # distribute truck across depots s.t. a_k_l == 1 if truck l start its tour from depot k
    a_matrix = {(k, l): 1 for (k, l) in itertools.product(D, V) if
                k - D[0] == l - V[0]}  # trucks distribution across the depots

    for (k, l) in itertools.product(D, V):
        if k - D[0] != l - V[0]:
            a_matrix[(k, l)] = 0

    if len(V) > len(D):
        for i in np.arange(1,1+int(len(V)/len(D))):
            for (k, l) in itertools.product(D, V[i*len(D):]):
                if k - D[0] == l - V[i*len(D):][0]:
                    a_matrix[(k, l)] = 1

    ######################################################

    OP_params = {'t': t, 'cv': cv, 'T': T, 'a_matrix': a_matrix}
    #########################
    generic_params['d'] = d

    params = {'SP_params': SP_params, 'OP_params': OP_params, 'generic_params': generic_params}
    return params


def OP_model(params,rl_solution):
    m = Model('OP')
    m.Params.TimeLimit = 3600 #in secondi
    m.setParam('LogFile', 'gurobi_optimization_log.txt') #stampa il log sul file di txt ATTENZIONE se stesso file non sovrascrive, aggiunge

    # PARAMS
    F = params['generic_params']['F']
    C = params['generic_params']['C']
    D = params['generic_params']['D']
    N = params['generic_params']['N']
    V = params['generic_params']['V']
    d = params['generic_params']['d']

    OP_params = params['OP_params']
    t = OP_params['t']
    cv = OP_params['cv']
    #T = OP_params['T']
    T = 600
    a_matrix = OP_params['a_matrix']

    SP_params = params['SP_params']
    H = SP_params['H']
    capf = SP_params['capf']

    #Variables
    h = {(l, a): m.addVar(vtype=GRB.BINARY, name='h({},{})'.format(l, a))
         for l in V for a in N} #variabile pari a 1 se il veicolo visita il nodo

    z = {(l, a, b): m.addVar(vtype=GRB.BINARY, name='z({},{},{})'.format(l, a, b))
         for l in V for a in N for b in N} #variabile pari a 1 se il veicolo passa per l'arco

    p = {(l, a): m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='p({},{})'.format(l, a))
         for l in V for a in N} #variabile di tempo di visita al nodo

    v = {(l, j): m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='v({},{})'.format(l, j))
         for l in V for j in F} #variabile che indica il carico del veicolo portato alla facility j

    # MODIFICA: Conversione delle chiavi in tuple, aggiunto perchè non leggeva il json
    rl_solution['h'] = {eval(k): v for k, v in rl_solution['h'].items()}
    rl_solution['z'] = {eval(k): v for k, v in rl_solution['z'].items()}
    if 'p' in rl_solution:
        rl_solution['p'] = {eval(k): v for k, v in rl_solution['p'].items()}
    if 'v' in rl_solution:
        rl_solution['v'] = {eval(k): v for k, v in rl_solution['v'].items()}

    if rl_solution:
        # preferibile usare GRB.Attr.Start, fa un vero e proprio warm start, mentre LB e UB fissa i valori delle variabili
        # può convenire LB/UB se si vuole fare un veloce controllo sulla feasibility della soluzione da RL
        for (l, a), value in rl_solution['h'].items():
            h[l, a].setAttr(GRB.Attr.Start, value)
            #h[l, a].setAttr(GRB.Attr.LB, value)
            #h[l, a].setAttr(GRB.Attr.UB, value)

        for (l, a, b), value in rl_solution['z'].items():
            z[l, a, b].setAttr(GRB.Attr.Start, value)
            #z[l, a, b].setAttr(GRB.Attr.LB, value)
            #z[l, a, b].setAttr(GRB.Attr.UB, value)
#MODIFICA, if 'p', 'v' in rl_solution se no dava errore non essendo nel file json
        if 'p' in rl_solution:
            for (l, a), value in rl_solution['p'].items():
                p[l, a].setAttr(GRB.Attr.Start, value)
                #p[l, a].setAttr(GRB.Attr.LB, value)
                #p[l, a].setAttr(GRB.Attr.UB, value)

        if 'v' in rl_solution:
            for (l, j), value in rl_solution['v'].items():
                v[l, j].setAttr(GRB.Attr.Start, value)
                #v[l, j].setAttr(GRB.Attr.LB, value)
                #v[l, j].setAttr.GRB.Attr.UB, value)
    m.update()

    of = quicksum(z[l, a, b] * t[a, b] for a in D + C for b in C + F for l in V)
    m.setObjective(of, GRB.MINIMIZE)

    # Constraints
    # (11)
    for l in V:
        m.addConstr(quicksum(h[l, j] for j in F) == quicksum(z[l, k, i] for k in D for i in C), name='C_11_({})'.format(l))
    # (12) #modificato così da consentire di non dover usare tutti i veicoli
    for l in V:
        m.addConstr(quicksum(h[l, k] for k in D) <= 1, name='C_12_onedeposit({})'.format(l))
    # (13) #modificato così da consentire di non dover usare tutti i veicoli
    for l in V:
        m.addConstr(quicksum(h[l, j] for j in F) <= 1, name='C_13_onefacility({})'.format(l))
    # (14) ## modificato per rimozione più veicoli per lo stesso nodo
    for i in C:
        m.addConstr(quicksum(h[l,i] for l in V) == 1, name='C_14_one_vehicle_per_node({})'.format(i))

    #vincolo 15 mancante, si riferiva alla possibilità di più veicoli per lo stesso nodo

    # 16)
    for l in V:
        for i in N:
            m.addConstr(z[l, i, i] == 0, name ='C_16_no_loop({},{})'.format(l, i))
    # (17)
    for k in D:
        for l in V:
            m.addConstr(quicksum(z[l, k, i] for i in C) <= a_matrix[k, l],
                            name='C_17_exit_from_depot({},{})'.format(k, l))
    # (18)
    for k in D:
        for l in V:
            m.addConstr(quicksum(z[l, j, k] for j in F) <= a_matrix[k, l],
                            name='C_18_enter_in_depot({},{})'.format(k, l))
    # (19)
    for j in F:
        for l in V:
            m.addConstr(quicksum(z[l, j, i] for i in C) == 0,
                            name='C_19_noclient_after_facility({},{})'.format(j, l))
    # (20)
    for j in F:
        for l in V:
            m.addConstr(quicksum(z[l, j, k] for k in D) == quicksum(z[l, i, j] for i in C),
                            name='C_20_depot_after_facility({},{})'.format(j, l))
    # (21)
    for l in V:
        for a in C:
            m.addConstr(quicksum(z[l, a, b] for b in C + F) == h[l, a],
                            name='C_21_fromclient_to_clientORfacility_({},{})'.format(l, a))
    # (22)
    for l in V:
        for a in C:
            m.addConstr(quicksum(z[l, b, a] for b in D + C) == h[l, a],
                            name='C_22_toclient_from_clientORdeposit_({},{})'.format(l, a))

    # (23) ## prima era vincoli 23-24-25
    for l in V:
        m.addConstr(quicksum(h[l,i]*d[i] for i in C) <= cv[l], name='C_23_capacity_vehicle({})'.format(l))

    # 26) calcolo del tempo di arrivo al nodo b se e solo se arriviamo a b da a (logical constraint)
    for l in V:
        for a in D + C:
            for b in C + F:
                m.addConstr((z[l, a, b] == 1) >> (p[l, b] == p[l, a] + t[a, b] * z[l, a, b]),
                            name='C_26_tempo({},{},{})'.format(l, a, b))

    # 27) tempo zero in ogni deposito
    m.addConstr(quicksum(p[l, k] for l in V for k in D) == 0, name='C_27_inizio_tempo')
    # 28) tempo di arrivo solo se si passa per il nodo
    for l in V:
        for a in N:
            m.addConstr((h[l, a] == 0) >> (p[l, a] == 0), name='C_28_tempo_solo_se_visita({},{})'.format(l, a))
    # 29)
    for l in V:
        m.addConstr(quicksum(p[l, j] for j in F) <= T, name='C_29_tempo_fine({})'.format(l))


    # (30) ## modificato per rimozione più veicoli per lo stesso nodo
    for l in V:
        for j in F:
            m.addConstr((h[l,j]==1)>>(quicksum(h[l,i]*d[i] for i in C) == v[l, j]),
                            name='C_30_load_of_l_to_j({},{})'.format(l, j))

    # (31)
    # #modificato perché abbiamo tutte le facilities, selezionata solo la capacità per la taglia media
    # for j in F:
    #     m.addConstr(quicksum(v[l, j] for l in V) <= quicksum(capf[j, h] for h in H if h==2),
    #                     name='C_31_load_of_j({})'.format(j))
    #MODIFICA per tener conto delle diverse capacità delle facilities
    facility_sizes = ["S", "S", "L", "M", "M", "L", "L", "S", "L", "M"]
    size_mapping = {"S": 1, "M": 2, "L": 3}

    for j in F:
        facility_size = facility_sizes[j]  # Ottieni la taglia della facility j
        size_code = size_mapping[facility_size]  # Ottieni il codice della taglia
        m.addConstr(quicksum(v[l, j] for l in V) <= capf[j, size_code],
                    name='C_31_load_of_j({})'.format(j))

    # (32)
    for l in V:
        for j in F:
            m.addConstr((h[l, j] == 0) >> (v[l, j] == 0),
                            name='C_32_no_load_unvisited_facility({},{})'.format(l, j))

    m.update()
    m.optimize()

    status = m.status
    if status == GRB.Status.INFEASIBLE:
        print('INFEASIBLE')
        m.computeIIS()
        iis_file = "model_iis.ilp"
        m.write(iis_file)
        solution = {}
    elif status == GRB.Status.OPTIMAL or status == 9:  # 9 is 'Time Limit Reached'
        if status == 2:
            print('Model was solved to optimality')
        if status == 9:
            print('Time Limit Reached')
        solution = {
            "of": round(m.objVal, 2),
            "h": {key: value.X for key, value in h.items() if abs(value.X) > 1e-6},
            "z": {key: value.X for key, value in z.items() if abs(value.X)> 1e-6},
            "p": {key: value.X for key, value in p.items()},
            "v": {key: value.X for key, value in v.items() if abs(value.X)},
            "final_gap": m.MIPGap
        }
    else:
        print (f"Status code is {status}")
        solution = {}

    return solution



if __name__ == '__main__':
    target_path = r'./'
    target_file = r'inst_realistic.json'

    data = load_json_instance(target_path, target_file)
    inst_data = data['inst_data']
    disdur = data['disdur_dict']
    NF = inst_data['NF']
    NC = inst_data['NC']
    ND = inst_data['ND']
    NV = inst_data['NV']

    params = create_params(NF,NC,ND,NV,disdur)
    rl_solution = load_rl_solution('./percorsi_formulazione.json') #placeholder, qua va inserita una funzione che legga la soluzione
                     # da reinforcement learning e assegni i valori a tutte le variabili
                     # potrebbe essere sufficiente bloccare i valori delle h (visita nodo) e z (visita arco)


    begin_optimization = time.time()
    solution = OP_model(params, rl_solution)
    end_optimization = time.time ()
    time_opt = end_optimization-begin_optimization