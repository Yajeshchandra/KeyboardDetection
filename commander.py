class Commander:
    def __init__(self):
        self.keypoints = None
        self.solver = None
        
    def load_keypoints(self, keypoints):
        self.keypoints = keypoints
        
    def load_solver(self, solver):
        self.solver = solver
        
    def solve(self,input):
        waypoints = []
        for char in input:
            if char in self.keypoints:
                
                waypoints.append(("hover",self.keypoints[char]+[0,0,0.1])) # need to add perpendicular distance
                waypoints.append(("press",self.keypoints[char]))
                waypoints.append(("hover",self.keypoints[char]+[0,0,0.1])) # need to add perpendicular distance
            
        for i in range(waypoints):
            pos = waypoints[i]
            if pos[0]=="hover":
                self.go_to_position(pos[1])
                self.correct_position()
            elif pos[0]=="press":
                success = self.press_key()
                if not success:
                    waypoints.insert(waypoints.index(i),self.keypoints["backspace"])
                    waypoints.insert(waypoints.index(i) + 1, ("hover", self.keypoints["backspace"] + [0, 0, 0.1]))
                    waypoints.insert(waypoints.index(i) + 2, ("hover", self.keypoints["backspace"] + [0, 0, 0.1]))
                    waypoints.insert(waypoints.index(i) + 3, ("press", self.keypoints["backspace"]))
                    i = i-1 #idk wtf i did here its not cpp
                    
            
    