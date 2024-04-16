import numpy as np
from scipy.optimize import linear_sum_assignment

# make cost matrix from r_t only
def make_cost_matrix(obs, m, n):
    # Create a list of keys for all UAV-target pairs
    keys = [f"uav{uav_id+1}_target{target_id+1}" for uav_id in range(m) for target_id in range(n)]

    # Extract the relevant observations and convert to a NumPy array
    observations = np.array([obs[key][0] for key in keys])

    # Reshape the array into the cost matrix
    cost_matrix = observations.reshape(m, n)
    return cost_matrix

# testing env: heuristic policy
def hungarian_assignment(cost_matrix):
    uav_idx, target_idx = linear_sum_assignment(cost_matrix) # maximize=True does not work. instead use -value
    # print("uav_idx-target_idx: ", uav_idx, target_idx)
    return uav_idx, target_idx

def uav1_target1_heuristic(battery, age, b1, b2, a1):
    if battery > b1:
        action = 1
    elif battery > b2:
        # previous_action = obs['previous_action']
        if age == 0 or age > a1: # uav was surveilling
        # if previous_action:
            action = 1
        else: # uav was charging
            action = 0
    else:
        action = 0
    return action

def get_action_from_pairs(UAV_idx, Target_idx, battery, age, m, n, b1, b2, a1):
    action = np.zeros(m, dtype=int)
    for uav_idx, target_idx in zip(UAV_idx, Target_idx):
        action[uav_idx] = uav1_target1_heuristic(battery[uav_idx], age[target_idx], b1, b2, a1)*(target_idx+1)

    if m > n: # in case of m > n: unselected uav stay charge even full battery
        unselected_uav_idx = np.setdiff1d(np.arange(m), UAV_idx) # returns the unique values in array1 that are not present in array2
        action[unselected_uav_idx] = 0
    return action

def r_t_hungarian(obs, m, n, b1=2000, b2=1000, a1=800):
    bat = obs['battery']
    age = obs['age']
    uav_idx, target_idx = hungarian_assignment(make_cost_matrix(obs, m, n))
    action = get_action_from_pairs(uav_idx, target_idx, bat, age, m, n, b1, b2, a1)
    return action

def high_age_first(obs, m, b3=1000):
    bat = obs['battery']
    age = obs['age']
    uav_list = [uav_idx for uav_idx in range(m)]
    action = np.zeros(m, dtype=int)
    for uav_idx in range(m):
        if bat[uav_idx] < b3:
            uav_list.remove(uav_idx)
    # highest age target gets to choose first the closest uav to it
    # sort age from high to low and return the index
    sorted_age_indices = np.argsort(age)[::-1]
    for target_idx in sorted_age_indices:
        closest_uav_idx = None
        closest_distance = float('inf')
        if uav_list == []:
            pass
        else:
            for uav_idx in uav_list:
                # Find the closest UAV to the target
                distance = obs[f"uav{uav_idx+1}_target{target_idx+1}"][0]
                if distance < closest_distance:
                    closest_distance = distance
                    closest_uav_idx = uav_idx
            action[closest_uav_idx] = target_idx+1
            uav_list.remove(closest_uav_idx)
    return action