import pickle
import random

# Helper methods
def flatten(list):
    "Flatten a 2D list into 1D"
    return set([item for sublist in list for item in sublist])

def parent_bfs(cuis, parent_dict):
    "Gets a unique set of parents for the list of CUIs"
    parents = []
    for cui in cuis:
        if cui in parent_dict:
            parents += [c for _,c in parent_dict[cui]]
    return set(parents)

def extend_lineage(l_given, l_compare, parent_dict):
    "Extend a lineage up one level and check if the LCA is found"
    d = None
    reached_root = False

    # Breadth First Search next level of parents
    parents = parent_bfs(l_given[-1], parent_dict)
    parents = parents.difference(flatten(l_given))

    if len(parents) > 0:
        # Check for common ancestors (only need to check parents)
        common_ancestors = flatten(l_compare).intersection(parents)
        if len(common_ancestors) > 0:
            for i, level in enumerate(l_compare):
                if not level.isdisjoint(parents):
                    d = len(l_given) + i
                    break
        l_given.append(parents)
    else:
        reached_root = True

    # Return extended lineage and distance if found
    return l_given, d, reached_root

class Umls():
    def __init__(self, path):
        with open(f'{path}/children.pickle', 'rb') as f:
            self.children = pickle.load(f)

        with open(f'{path}/parents.pickle', 'rb') as f:
            self.parents = pickle.load(f)

    def get_candidates(self, cui:str, k:int):
        "Returns k ontological candidates for the specified CUI"
        candidates = set()

        # Append up to half parent candidates
        parents = []
        if cui in self.parents:
            parents = self.parents[cui]
            if len(parents) > int(k/2):
                parents = random.sample(parents, int(k/2))
            candidates = candidates.union(set(parents))
            k = k-len(candidates)

        # Append children
        kids = []
        if cui in self.children:
            kids = self.children[cui]
            if len(kids) > k:
                kids = random.sample(kids, k)
            kids = set(kids).difference(candidates)
            candidates = candidates.union(set(kids))
            k = k-len(kids)

        # Append siblings (children of parents) until k is reached
        for _, parent_cui in parents:
            if k == 0:
                break
            if parent_cui in self.children:
                siblings = self.children[parent_cui]
                if len(siblings) > k:
                    siblings = random.sample(siblings, k)
                siblings = set(siblings).difference(candidates)
                candidates = candidates.union(set(siblings))
                k = k-len(siblings)

        # Append siblings (parents of children) until k is reached
        for _, child_cui in kids:
            if k == 0:
                break
            if child_cui in self.parents:
                siblings = self.parents[child_cui]
                if len(siblings) > k:
                    siblings = random.sample(siblings, k)
                siblings = set(siblings).difference(candidates)
                candidates = candidates.union(set(siblings))
                k = k-len(siblings)

        # Fill in with empty strings if enough candidates can't be found
        #TODO: Check that all candidates are in the dictionary?
        #TODO: Return non-random options?
        return [[n,c] for n,c in candidates]  + ([["NAME","CUI"]]*k)

    def dist(self, cui1, cui2):
        "Finds the lowest common ancestor between two CUIs in the UMLS and calculates distance between them"

        # If CUIs are identical, distance is zero
        if cui1==cui2:
            return 0

        lineage1 = [{cui1}]
        lineage2 = [{cui2}]
        reached_root1 = reached_root2 = False
        while not reached_root1 or not reached_root2:
            if not reached_root1:
                lineage1, d, reached_root1 = extend_lineage(lineage1, lineage2, self.parents)
                if d is not None: break

            if not reached_root2:
                lineage2, d, reached_root2 = extend_lineage(lineage2, lineage1, self.parents)
                if d is not None: break

        if reached_root1 and reached_root2:
            d = len(lineage1) + len(lineage2)

        if d is None:
            print("THIS SHOULD NOT HAPPEN")
        return d

    def similarity(self, cui1, cui2):
        "Calculates the ontological similarity between two CUIS"
        d = self.dist(cui1, cui2)
        return 0 if d<0 else 1/(1+d)