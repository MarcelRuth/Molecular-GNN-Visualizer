from graph_generator import mol2graph, small_graph

def get_data(model, molecules):
    if model == 'GNN':
        return [small_graph(mol) for mol in molecules]
    else:
        return [mol2graph(mol)[0] for mol in molecules],  [mol2graph(mol)[1] for mol in molecules]
