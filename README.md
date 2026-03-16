# Swarm Patrol 
### This case study is under the Agentic AI (Decentralised Swarm Intelligence) track. It aligns with Sustainable Development Goals 9 (Industry, Innovation, and Infrastructure) and 3 (Good Health and Well-being).

## Preview
![Preview](Snapshots/Preview1.png)

## __Feature/Function__  
### •	**The Map:**   
The search area is represented as a 21 x 21 grid cell.

### •	**The Actors:**  
A "Command Agent" powered by an LLM sits at the origin (10,10). It deploys a fleet of drones (minimum of 3) based on the size of the grid.  

### •	**Time Mechanics:**   
The simulation or real-world logic operates in turns, proposing "1 minutes as one round". In each round, each drone can move one step and perform thermal scan.  

### •	**Battery Drain:**   
Moving one step or 1 minute consumes 1% of a drone's battery. Charging a drone takes 30 minutes.  

### •	**Exploration Logic:**  
As drones move, they mark grid cells as "unvisited," "clear," or "suspect." The notes explicitly mention using a A* algorithm to handle this grid traversal.  

### •	**Swarm Intelligence:**   
Drones utilize "shared memory." When one drone maps an area or finds an obstacle, other drones instantly update their own pathfinding logic based on that shared data.  

### •	**Survivor Verification:**   
To avoid false alarms, the drone uses multimodal inputs (temperature, sound, shape) to autonomously determine if it has actually found a survivor. Might have a chance of being a false alarm  

### •	**Routing for Humans:**  
Once a survivor is confirmed, the AI analyzes the grid to plot a safe, optimal route for human rescue teams to follow with using the A* algorithm.  
 

## __Tech Stack:__    
•	Next.js   
•	Mesa  
•	NumPy & Pandas  
•	NetworkX  
•	FastMCP  
•	FastAPI  
•	LangChain  

![Preview](Snapshots/Tech.png)

## __Future Roadmap:__
•	**Risk Management:** The system accounts for "rescue priority" triage and handles edge cases like drone hardware failures. 

•	**Dynamic Routing:** The system introduces random environmental hazards like strong winds or bird flocks. The onboard LLM allows the drone to autonomously recalculate and change its route to avoid these.  

•	**Signal Attenuation:** Every grid space away from the Command Agent causes a 5% drop in signal strength. Drones must switch modes to act as "signal extensions" (relays) for each other to reach further into the grid
