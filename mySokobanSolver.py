'''
IFN680 Sokoban Assignment

The functions and classes defined in this module will be called by a marker script. 
You should complete the functions and classes according to their specified interfaces.

You are not allowed to change the defined interfaces.
That is, changing the formal parameters of a function will break the 
interface and triggers to a fail for the test of your code.
'''

import time
import search
import sokoban


def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    e.g.  [ (1234567, 'Ada', 'Lovelace'), (1234568, 'Grace', 'Hopper'), (1234569, 'Eva', 'Tardos') ]
    '''

    return [(12178195, 'Martin','Diaz')]

def create_matrix(warehouse):
    # code from sokoban
    x_size = 1+max(x for x,y in warehouse.walls)
    y_size = 1+max(y for x,y in warehouse.walls)
    vis = [[" "] * x_size for y in range(y_size)]
    for (x,y) in warehouse.walls:
        vis[y][x] = "#"
    for (x,y) in warehouse.targets:
        vis[y][x] = "."
    # if worker is on a target display a "!", otherwise a "@"
    # exploit the fact that Targets has been already processed
    if vis[warehouse.worker[1]][warehouse.worker[0]] == ".": # Note y is worker[1], x is worker[0]
        vis[warehouse.worker[1]][warehouse.worker[0]] = "!"
    else:
        vis[warehouse.worker[1]][warehouse.worker[0]] = "@"
    # if a box is on a target display a "*"
    # exploit the fact that Targets has been already processed
    for (x,y) in warehouse.boxes:
        if vis[y][x] == ".": # if on target
            vis[y][x] = "*"
        else:
            vis[y][x] = "$"
    return vis

# checks if a coordinate is in the matrix
def in_matrix(state, mt):
    x_size = len(mt[0])
    y_size = len(mt)
    (x,y)=state
    if(x<0 or x>=x_size or y<0 or y>=y_size): return False
    return True

# checks if a coordinate is a wall
def is_wall(state, mt):
    if(not in_matrix(state,mt)): return False
    (x,y)=state
    if(mt[y][x]!="#"): return False
    return True

# methods for taboo cells--------------------------------------------------------------------------------------
# check if a cell is a taboo corner
def taboo_corner(warehouse, state):
    (x,y)=state
    if((x,y) in warehouse.walls or (x,y) in warehouse.targets): return False
    if((x+1,y) in warehouse.walls and (x,y+1) in warehouse.walls): return True
    if((x+1,y) in warehouse.walls and (x,y-1) in warehouse.walls): return True
    if((x-1,y) in warehouse.walls and (x,y+1) in warehouse.walls): return True
    if((x-1,y) in warehouse.walls and (x,y-1) in warehouse.walls): return True
    return False

# finds a vertical taboo wall starting from the bottom cell
def up_taboo_wall(warehouse, state, mt):
    if(is_wall(state,mt)): return
    (x,y)=state
    # check that bottom cell is wall
    if(not is_wall((x,y+1),mt)): return
    x_size = len(mt[0])
    y_size = len(mt)
        
    # checks if right cell is wall
    if(is_wall((x+1,y),mt)):
        #goes up creating tabooWall to the right
        isTaboo=True
        (cx,cy)=(x,y)
        while(mt[cy][cx]!="#"):
            if(not is_wall((cx+1,cy),mt) or (cx,cy) in warehouse.targets):
                isTaboo=False
                break
            cy-=1
            
        if(isTaboo):
            (cx,cy)=(x,y)
            while(mt[cy][cx]!="#"):
                mt[cy][cx]="X"
                cy-=1

    # checks if left cell is wall
    if(is_wall((x-1,y),mt)):
        #goes up creating tabooWall to the left
        isTaboo=True
        (cx,cy)=(x,y)
        while(mt[cy][cx]!="#"):
            if(not is_wall((cx-1,cy),mt) or (cx,cy) in warehouse.targets):
                isTaboo=False
                break
            cy-=1
            
        if(isTaboo):
            (cx,cy)=(x,y)
            while(mt[cy][cx]!="#"):
                mt[cy][cx]="X"
                cy-=1


# finds a horizontal taboo wall starting from the left cell
def right_taboo_wall(warehouse, state, mt):
    if(is_wall(state,mt)): return
    (x,y)=state
    # check that left cell is wall
    if(not is_wall((x-1,y),mt)): return
    x_size = len(mt[0])
    y_size = len(mt)
        
    # checks if up cell is wall
    if(is_wall((x,y-1),mt)):
        #goes right creating tabooWall above the cells
        isTaboo=True
        (cx,cy)=(x,y)
        while(mt[cy][cx]!="#"):
            if(not is_wall((cx,cy-1),mt) or (cx,cy) in warehouse.targets):
                isTaboo=False
                break
            cx+=1
            
        if(isTaboo):
            (cx,cy)=(x,y)
            while(mt[cy][cx]!="#"):
                mt[cy][cx]="X"
                cx+=1

    # checks if down cell is wall
    if(is_wall((x,y+1),mt)):
        #goes right creating tabooWall below the cells
        isTaboo=True
        (cx,cy)=(x,y)
        while(mt[cy][cx]!="#"):
            if(not is_wall((cx,cy+1),mt) or (cx,cy) in warehouse.targets):
                isTaboo=False
                break
            cx+=1
            
        if(isTaboo):
            (cx,cy)=(x,y)
            while(mt[cy][cx]!="#"):
                mt[cy][cx]="X"
                cx+=1


def print_matrix(mt):
    x_size = len(mt[0])
    y_size = len(mt)
    cells=""
    for y in range(y_size):
        for x in range(x_size):
            cells+=mt[y][x]
        cells+='\n'
    cells=cells[:-1]
    return cells

# methods for checking sequence of oprations--------------------------------------------------------------------------------------------
# process one action. possible actions: ['Left', 'Down','Right', 'Up']
def process_action(action, warehouse):
    wh=warehouse.copy()
    (x,y)=warehouse.worker
    # Left action
    if(action=="Left"):
        #moving to a empty cell or empty target
        if((x-1,y) not in warehouse.walls and (x-1,y) not in warehouse.boxes):
            wh.worker=(x-1,y)
            return (True,wh)
        #trying to move box
        if((x-1,y) in warehouse.boxes):
            #check that box is pushed into empty or target cell
            if((x-2,y) not in warehouse.walls and (x-2,y) not in warehouse.boxes):
                wh.worker=(x-1,y)
                wh.boxes.remove((x-1,y))
                wh.boxes.append((x-2,y))
                return (True,wh)
        # if none of the options above is met, the movement isn't valid
        return (False,wh)
                
    # Right action
    if(action=="Right"):
        #moving to a empty cell or empty target
        if((x+1,y) not in warehouse.walls and (x+1,y) not in warehouse.boxes):
            wh.worker=(x+1,y)
            return (True,wh)
        #trying to move box
        if((x+1,y) in warehouse.boxes):
            #check that box is pushed into empty or target cell
            if((x+2,y) not in warehouse.walls and (x+2,y) not in warehouse.boxes):
                wh.worker=(x+1,y)
                wh.boxes.remove((x+1,y))
                wh.boxes.append((x+2,y))
                return (True,wh)
        # if none of the options above is met, the movement isn't valid
        return (False,wh)
                
    # Up action
    if(action=="Up"):
        #moving to a empty cell or empty target
        if((x,y-1) not in warehouse.walls and (x,y-1) not in warehouse.boxes):
            wh.worker=(x,y-1)
            return (True,wh)
        #trying to move box
        if((x,y-1) in warehouse.boxes):
            #check that box is pushed into empty or target cell
           if((x,y-2) not in warehouse.walls and (x,y-2) not in warehouse.boxes):
                wh.worker=(x,y-1)
                wh.boxes.remove((x,y-1))
                wh.boxes.append((x,y-2))
                return (True,wh)
        # if none of the options above is met, the movement isn't valid
        return (False,wh)
    
    # Down action
    if(action=="Down"):
        #moving to a empty cell or empty target
        if((x,y+1) not in warehouse.walls and (x,y+1) not in warehouse.boxes):
            wh.worker=(x,y+1)
            return (True,wh)
        #trying to move box
        if((x,y+1) in warehouse.boxes):
            #check that box is pushed into empty or target cell
           if((x,y+2) not in warehouse.walls and (x,y+2) not in warehouse.boxes):
                wh.worker=(x,y+1)
                wh.boxes.remove((x,y+1))
                wh.boxes.append((x,y+2))
                return (True,wh)
        # if none of the options above is met, the movement isn't valid
        return (False,wh)
        
    raise Exception("Invalid instruction")
        
    
def taboo_cells(warehouse):
    '''  
    Identify the taboo cells of a warehouse. A cell inside a warehouse is 
    called 'taboo' if whenever a box get pushed on such a cell then the puzzle 
    becomes unsolvable.  
    When determining the taboo cells, you must ignore all the existing boxes, 
    simply consider the walls and the target cells.  
    Use only the following two rules to determine the taboo cells;
     Rule 1: if a cell is a corner inside the warehouse and not a target, 
             then it is a taboo cell.
     Rule 2: all the cells between two corners inside the warehouse along a 
             wall are taboo if none of these cells is a target.
    
    @param warehouse: a Warehouse object

    x_size = 1+max(x for x,y in warehouse.walls)
    y_size = 1+max(y for x,y in warehouse.walls)

    @return
       A string representing the puzzle with only the wall cells marked with 
       an '#' and the taboo cells marked with an 'X'.  
       The returned string should NOT have marks for the worker, the targets,
       and the boxes.  
    '''
    ##         "INSERT YOUR CODE HERE" 
    mt=create_matrix(warehouse)
    x_size = len(mt[0])
    y_size = len(mt)

    # iterate all points inside problem
    for y in range(y_size):
        # find the bounds of the problem given the y coordinate
        startx=-1
        endx=-1;
        for x in range(x_size):
            if(mt[y][x]=='#' and startx==-1): startx=x
            if(mt[y][x]=='#'): endx=x

        #iterate cells inside bounds to find tabooCorners
        for x in range(startx+1, endx):
            if(taboo_corner(warehouse, (x,y))):
                mt[y][x]='X'
            #finds taboo wall going up starting at that corner
            up_taboo_wall(warehouse, (x,y), mt)
            #finds taboo wall going right starting at that corner
            right_taboo_wall(warehouse, (x,y), mt)

    for y in range(y_size):
        for x in range(x_size):
            if(mt[y][x]!="#" and mt[y][x]!="X"):
                mt[y][x]=" ";

    return print_matrix(mt)

class SokobanPuzzle(search.Problem):
    macro=False
    allow_taboo_push=False
    '''
    An instance of the class 'SokobanPuzzle' represents a Sokoban puzzle.
    An instance contains information about the walls, the targets, the boxes
    and the worker.

    Your implementation should be fully compatible with the search functions of 
    the provided module 'search.py'. It uses search.Problem as a sub-class. 
    That means, it should have a:
    - self.actions() function
    - self.result() function
    - self.goal_test() function
    See the Problem class in search.py for more details on these functions.
    
    Each instance should have at least the following attributes:
    - self.allow_taboo_push
    - self.macro
    
    When self.allow_taboo_push is set to True, the 'actions' function should 
    return all possible legal moves including those that move a box on a taboo 
    cell. If self.allow_taboo_push is set to False, those moves should not be
    included in the returned list of actions.
    
    If self.macro is set True, the 'actions' function should return 
    macro actions. If self.macro is set False, the 'actions' function should 
    return elementary actions.
    
    
    '''
    
    def __init__(self, warehouse):
        #initial state is the current warehouse
        self.initial=warehouse
        # # state of the problem
        # self.state=warehouse
        # matrix representation of warehouse
        self.mt=create_matrix(warehouse)
        # find taboo cells
        self.taboo_cells=[]
        # distance to target
        self.distances = {}
        for target in warehouse.targets:
            self.distances[target]=self.get_distances(target)
        #amount of iterations done
        self.iter=0
        # save the visited states to avoid repeting them in future actions (note: this breakes the optimallity that can be achieved in star search, bfs keeps being optimal)
        self.vis={}
        self.vis[warehouse]=1

        
        tabooString=taboo_cells(warehouse)
        lines=tabooString.split('\n')
        while("" in lines): lines.remove("")
        for y in range(len(lines)):
            for x in range(len(lines[0])):
                if(lines[y][x]=='X'): self.taboo_cells.append((x,y))

    def actions(self, state):
        """
        Return the list of actions that can be executed in the given state.
        
        As specified in the header comment of this class, the attributes
        'self.allow_taboo_push' and 'self.macro' should be tested to determine
        what type of list of actions is to be returned.
        """
        # no macro actions are used
        if(not self.macro):
            possible_actions=['Left', 'Down','Right', 'Up']
            real_actions=[]
            for action in possible_actions:
                (ok,wh)=process_action(action, state)
                # check if the action is valid
                if(ok):
                    wh.boxes.sort()
                    if(wh not in self.vis):
                        self.vis[wh]=1
                    else:
                        continue
                    # if boxes can be pushed to taboo cells
                    if(self.allow_taboo_push):
                        real_actions.append(action)
                    #if boxes can not be pushed to taboo cells
                    else:
                        box_in_taboo=False
                        for box in wh.boxes:
                            if(box in self.taboo_cells):
                                box_in_taboo=True
                        if(not box_in_taboo):
                            real_actions.append(action) 
            return real_actions
            
        #using macro actions
        if(self.macro):
            actions=[]
            for box in state.boxes:
                (bx,by)=box
                # push right only if can arrive left and there is no objects on the right
                if(can_go_there(state,(by,bx-1)) and (bx+1,by) not in state.walls  and (bx+1,by) not in state.boxes):
                    # update shortest distance to new states
                    wh=state.copy()
                    wh.boxes.remove((bx,by))
                    wh.boxes.append((bx+1,by))
                    wh.worker=(bx,by)
                    wh.boxes.sort()
                    if(wh not in self.vis):
                        self.vis[wh]=1    
                        if(self.allow_taboo_push):
                            actions.append(((by,bx),"Right"))
                        else:
                            if((bx+1,by) not in self.taboo_cells):
                                actions.append(((by,bx),"Right"))
                    
                # push left only if can arrive right and there is no objects on the left
                if(can_go_there(state,(by,bx+1)) and (bx-1,by) not in state.walls  and (bx-1,by) not in state.boxes):
                    # update shortest distance to new states
                    wh=state.copy()
                    wh.boxes.remove((bx,by))
                    wh.boxes.append((bx-1,by))
                    wh.worker=(bx,by)
                    wh.boxes.sort()
                    if(wh not in self.vis):
                        self.vis[wh]=1
                        if(self.allow_taboo_push):
                            actions.append(((by,bx),"Left"))
                        else:
                            if((bx-1,by) not in self.taboo_cells):
                                actions.append(((by,bx),"Left"))

                # push up only if can arrive down and there is no objects up
                if(can_go_there(state,(by+1,bx)) and (bx,by-1) not in state.walls  and (bx,by-1) not in state.boxes):
                    # update shortest distance to new states
                    wh=state.copy()
                    wh.boxes.remove((bx,by))
                    wh.boxes.append((bx,by-1))
                    wh.worker=(bx,by)
                    wh.boxes.sort()
                    if(wh not in self.vis):
                        self.vis[wh]=1
                        if(self.allow_taboo_push):
                            actions.append(((by,bx),"Up"))
                        else:
                            if((bx,by-1) not in self.taboo_cells):
                                actions.append(((by,bx),"Up"))

                # push down only if can arrive up and there is no objects down
                if(can_go_there(state,(by-1,bx)) and (bx,by+1) not in state.walls  and (bx,by+1) not in state.boxes):
                    # update shortest distance to new states
                    wh=state.copy()
                    wh.boxes.remove((bx,by))
                    wh.boxes.append((bx,by+1))
                    wh.worker=(bx,by)
                    wh.boxes.sort()
                    if(wh not in self.vis):
                        self.vis[wh]=1
                        if(self.allow_taboo_push):
                            actions.append(((by,bx),"Down"))
                        else:
                            if((bx,by+1) not in self.taboo_cells):
                                actions.append(((by,bx),"Down"))
            return actions
                    

    def result(self, state, action):
        # no macro actions are used
        if(not self.macro):
            (ok,wh)=process_action(action, state)
            if(not ok): raise Exception("Invalid instruction")
            wh.boxes.sort()
            # self.state=wh
            return wh
        
            
        # macro actions are used    
        if(self.macro):
            wh=state.copy()
            (box,direction)=action
            (by,bx)=box
            if(direction=="Right"):
                wh.boxes.remove((bx,by))
                wh.boxes.append((bx+1,by))
                wh.worker=(bx,by)
                
            if(direction=="Left"):
                wh.boxes.remove((bx,by))
                wh.boxes.append((bx-1,by))
                wh.worker=(bx,by)
                
            if(direction=="Up"):
                wh.boxes.remove((bx,by))
                wh.boxes.append((bx,by-1))
                wh.worker=(bx,by)

            if(direction=="Down"):
                wh.boxes.remove((bx,by))
                wh.boxes.append((bx,by+1))
                wh.worker=(bx,by)

            wh.boxes.sort()
            
            self.iter+=1
            if(not self.iter%10000):
                print(self.iter)
            return wh

    # auxiliar function to find the shortes path from each box to one of the targets
    def get_distances(self, box):
        inf=1e8
        # initial spot for worker
        (x,y)=box
        mt=create_matrix(self.initial)
        x_size = len(mt[0])
        y_size = len(mt) 
        # array of elements already visited. distance is also stored in this array
        vis = [[-1] * x_size for y in range(y_size)]
        #distance to starting point
        vis[y][x]=0;
        #implement bfs 
        frontier=[(x,y)]
        while(len(frontier)>0):
            #get element and delete it from frontier
            (x,y)=frontier[0]
            frontier.pop(0)
                
            # go right
            if((x+1,y) not in self.initial.walls):
                if(vis[y][x+1]==-1):
                    frontier.append((x+1,y))
                    vis[y][x+1]=vis[y][x]+1

            # go left
            if((x-1,y) not in self.initial.walls):
                if(vis[y][x-1]==-1):
                    frontier.append((x-1,y))
                    vis[y][x-1]=vis[y][x]+1
            # go up
            if((x,y-1) not in self.initial.walls):
                if(vis[y-1][x]==-1):
                    frontier.append((x,y-1))
                    vis[y-1][x]=vis[y][x]+1
            # go down
            if((x,y+1) not in self.initial.walls):
                if(vis[y+1][x]==-1):
                    frontier.append((x,y+1))
                    vis[y+1][x]=vis[y][x]+1
                    
        # return the vis matrix which contains the distances
        return vis

    # function used in star_search that passes a node
    def h(self, node):
        return self.heuristic(node.state)
    # heuristic function for a state, the heuristic returns the amount of boxes not in a target
    def heuristic(self, state):
        inf = 1e8
        total = 0
        
        # Create a set to keep track of used targets
        used_targets = set()
        
        # For each box, find the closest available (unused) target
        for box in state.boxes:
            (x,y)=box
            min_distance = inf
            best_target = None
            for target in state.targets:
                if target not in used_targets:
                    # Calculate distance to the target
                    distance = self.distances[target][y][x]
                    
                    # Check if this is the closest available target
                    if distance < min_distance:
                        min_distance = distance
                        best_target = target
                    

            used_targets.add(best_target)
            total += min_distance

        total*=3
        
        # check if some boxes can no longer be moved due to other boxes
        for box in state.boxes:
            if box in state.targets: continue
            (x,y)= box
            if((x+1,y) in state.boxes and (x+1,y) not in state.targets):
                if(((x,y+1) in state.walls and (x+1,y+1) in state.walls) or((x,y-1) in state.walls and (x+1,y-1) in state.walls)):
                    total+=inf
                    break
            if((x,y+1) in state.boxes and (x,y+1) not in state.targets):
                if(((x+1,y) in state.walls and (x+1,y+1) in state.walls) or((x-1,y) in state.walls and (x-1,y+1) in state.walls)):
                    total+=inf
                    break

        return total
            
    # checks if a state is a goal (not only one goal can be definded since the order of the boxes affects the comparison)
    def goal_test(self, state):
        state.boxes.sort()
        state.targets.sort()
        return (state.boxes==state.targets)

    # prints the solution found
    def get_solution(self, goal_node):
        # taken from week9 practical class
        """
            Shows solution represented by a specific goal node.
            For example, goal node could be obtained by calling 
                goal_node = breadth_first_tree_search(problem)
        """
        # path is list of nodes from initial state (root of the tree)
        # to the goal_node
        path = goal_node.path()
        # print the solution
        moves = []
        for node in path:
            if node.action:
                moves += [node.action]
        return moves
        


def check_action_seq(warehouse, action_seq):
    '''
    
    Determine if the sequence of actions listed in 'action_seq' is legal or not.
    
    Important notes:
      - a legal sequence of actions does not necessarily solve the puzzle.
      - an action is legal even if it pushes a box onto a taboo cell.
        
    @param warehouse: a valid Warehouse object

    @param action_seq: a sequence of legal actions.
           For example, ['Left', 'Down', Down','Right', 'Up', 'Down']
           
    @return
        The string 'Failure', if one of the action was not successul.
           For example, if the agent tries to push two boxes at the same time,
                        or push one box into a wall, or walk into a wall.
        Otherwise, if all actions were successful, return                 
               A string representing the state of the puzzle after applying
               the sequence of actions.  This must be the same string as the
               string returned by the method  Warehouse.__str__()
    '''
    
    ##         "INSERT YOUR CODE HERE"
    mt=create_matrix(warehouse)
    x_size = len(mt[0])
    y_size = len(mt)

    # process all actions one by one, store result in ok
    ok=True
    wh=warehouse.copy()
    for action in action_seq:
        (ok,wh)=process_action(action, wh)
        if(ok==False): return 'Failure'
    # all actions where valid
    return wh.__str__()
        

def solve_sokoban_elem(warehouse):
    '''    
    This function should solve using elementary actions 
    the puzzle defined in a file.
    
    @param warehouse: a valid Warehouse object

    @return
        If puzzle cannot be solved return the string 'Impossible'
        If a solution was found, return a list of elementary actions that solves
            the given puzzle coded with 'Left', 'Right', 'Up', 'Down'
            For example, ['Left', 'Down', Down','Right', 'Up', 'Down']
            If the puzzle is already in a goal state, simply return []
    '''
    solver=SokobanPuzzle(warehouse)
    solver.macro=False
    
    # Solve with star
    sol_ts = search.astar_graph_search(solver)

    # Solve with BFS
    # sol_ts = search.breadth_first_graph_search(solver)

    if(sol_ts is None): return "Impossible"
    return solver.get_solution(sol_ts)


def can_go_there(warehouse, dst):
    '''    
    Determine whether the worker can walk to the cell dst=(row,column) 
    without pushing any box.
    
    @param warehouse: a valid Warehouse object

    @return
      True if the worker can walk to cell dst=(row,column) without pushing any box
      False otherwise
    '''
    # initial spot for worker
    (x,y)=warehouse.worker
    # target
    (ty,tx)=dst
    mt=create_matrix(warehouse)
    x_size = len(mt[0])
    y_size = len(mt) 
    # array of elements already visited
    vis = [[0] * x_size for y in range(y_size)]
    # parent of every cell to reconstruct path
    par = [[0] * x_size for y in range(y_size)]
    # starting point is its own parent
    par[y][x]=(x,y)
    #implement dfs 
    frontier=[(x,y)]
    while(len(frontier)>0):
        #break if target is found:
        if(vis[ty][tx]): break
        #get element and delete it from frontier
        (x,y)=frontier[-1]
        frontier.pop()
        if(vis[y][x]): continue
        vis[y][x]=1
        # if on target
        if((x,y)==(tx,ty)): break
        # go right
        if((x+1,y) not in warehouse.walls and (x+1,y) not in warehouse.boxes):
            frontier.append((x+1,y))
            par[y][x+1]=(x,y)
        # go left
        if((x-1,y) not in warehouse.walls and (x-1,y) not in warehouse.boxes):
            frontier.append((x-1,y))
            par[y][x-1]=(x,y)
        # go up
        if((x,y-1) not in warehouse.walls and (x,y-1) not in warehouse.boxes):
            frontier.append((x,y-1))
            par[y-1][x]=(x,y)
        # go down
        if((x,y+1) not in warehouse.walls and (x,y+1) not in warehouse.boxes):
            frontier.append((x,y+1))
            par[y+1][x]=(x,y)
            
    # if the target node was visited
    if(vis[ty][tx]): return True
    else: return False

def solve_sokoban_macro(warehouse):
    '''    
    Solve using macro actions the puzzle defined in the warehouse passed as
    a parameter. A sequence of macro actions should be 
    represented by a list M of the form
            [ ((r1,c1), a1), ((r2,c2), a2), ..., ((rn,cn), an) ]
    For example M = [ ((3,4),'Left') , ((5,2),'Up'), ((12,4),'Down') ] 
    means that the worker first goes the box at row 3 and column 4 and pushes it left,
    then goes to the box at row 5 and column 2 and pushes it up, and finally
    goes the box at row 12 and column 4 and pushes it down.
    
    @param warehouse: a valid Warehouse object

    @return
        If puzzle cannot be solved return the string 'Impossible'
        Otherwise return M a sequence of macro actions that solves the puzzle.
        If the puzzle is already in a goal state, simply return []
    '''
    
    solver=SokobanPuzzle(warehouse)
    solver.macro=True

    # Solve with star
    sol_ts = search.astar_graph_search(solver)

    # Solve with BFS
    # sol_ts = search.breadth_first_graph_search(solver)

    if(sol_ts is None): return "Impossible"
    print("graph iterations: " + str(solver.iter))
    return solver.get_solution(sol_ts)

