import heapq
import math

# Ορισμός των τύπων εδάφους και των αντίστοιχων κόστων
terrain_costs = {
    'R': 1,    # Κανονικός δρόμος
    'H': 0.5,  # Αυτοκινητόδρομος
    'P': 2,    # Πάρκο
    'B': math.inf,  # Κτίριο (αδιάβατο εμπόδιο)
    'W': math.inf,  # Νερό (αδιάβατο εμπόδιο)
}

# Το 10x10 πλέγμα της πόλης
city_grid = [
    ['S', 'R', 'R', 'R', 'B', 'W', 'R', 'H', 'H', 'H'],
    ['R', 'B', 'B', 'R', 'H', 'H', 'R', 'R', 'B', 'H'],
    ['R', 'P', 'P', 'R', 'B', 'R', 'R', 'R', 'B', 'R'],
    ['R', 'R', 'R', 'R', 'W', 'R', 'P', 'P', 'R', 'R'],
    ['R', 'R', 'B', 'R', 'R', 'R', 'H', 'H', 'R', 'B'],
    ['B', 'W', 'R', 'P', 'P', 'R', 'B', 'R', 'R', 'R'],
    ['P', 'P', 'R', 'R', 'R', 'R', 'R', 'R', 'B', 'B'],
    ['R', 'B', 'R', 'R', 'R', 'W', 'H', 'H', 'R', 'R'],
    ['R', 'R', 'R', 'R', 'B', 'R', 'R', 'R', 'B', 'R'],
    ['H', 'H', 'H', 'B', 'B', 'R', 'R', 'G', 'R', 'R'],
]

# Συντεταγμένες εκκίνησης και στόχου
start = (0, 0)  # Σημείο 'S'
goal = (9, 7)   # Σημείο 'G'

# Ευρετική συνάρτηση: Manhattan απόσταση διά 2
def heuristic(current, goal):
    return (abs(current[0] - goal[0]) + abs(current[1] - goal[1])) / 2

# Εύρεση του κόστους μετακίνησης στον επόμενο κόμβο
def get_cost(x, y):
    terrain = city_grid[x][y]
    return terrain_costs.get(terrain, math.inf)  # Επιστρέφει κόστος ή απειρο αν είναι μη προσβάσιμο

# Συνάρτηση για εύρεση γειτονικών κελιών (πάνω, κάτω, αριστερά, δεξιά)
def get_neighbors(node):
    x, y = node
    neighbors = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # επάνω, κάτω, αριστερά, δεξιά
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < len(city_grid) and 0 <= ny < len(city_grid[0]):
            if get_cost(nx, ny) < math.inf:  # Αποφεύγουμε τα αδιάβατα εμπόδια
                neighbors.append((nx, ny))
    return neighbors

# Συνάρτηση A*
def a_star(start, goal):
    open_list = []  # Ορίζουμε τη λίστα σύνορο
    heapq.heappush(open_list, (0, start))  # Πρώτη κατάσταση στη λίστα
    came_from = {}  # Για την ανίχνευση διαδρομής
    g_score = {start: 0}  # Κόστος εκκίνησης
    f_score = {start: heuristic(start, goal)}  # f(n) = g(n) + h(n)

    while open_list:
        current_priority, current = heapq.heappop(open_list)
        
        # Αν φτάσαμε στον στόχο, σταματάμε και εξάγουμε τη διαδρομή
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path, g_score[goal]
        
        for neighbor in get_neighbors(current):
            tentative_g_score = g_score[current] + get_cost(*neighbor)

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                
                # Αν δεν είναι ήδη στο open_list, προσθέτουμε
                if neighbor not in [i[1] for i in open_list]:
                    heapq.heappush(open_list, (f_score[neighbor], neighbor))

    return None, math.inf  # Αν δεν βρεθεί διαδρομή

# Εκτέλεση του αλγορίθμου
path, cost = a_star(start, goal)
if path:
    print("Βέλτιστη Διαδρομή:", path)
    print("Συνολικό Κόστος Διαδρομής:", cost)
else:
    print("Δεν βρέθηκε διαδρομή.")
