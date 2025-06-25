import threading
import random
import time
import math
import numpy as np

class Scenario3:
    def __init__(self):
        """
        This scenario represents a logistic process. The supervisor will ramdonly distribute tasks for the AGVs.
        A task can be the material transportation from warehouse to docks and vice versa.
        """

        self.get_logger().info(f"Loading scenario 3...")

        self.positions = {'hangar1': [(-30.0, -30.0, 0.0), (-37.5, -30.0, 0.0), (-45.0, -30.0, 0.0)],
                                'hangar2': [(-30.0, 0.0, 0.0), (-37.5, 0.0, 0.0), (-45.0, 0.0, 0.0)],
                                'hangar3': [(-30.0, 30.0, 0.0), (-37.5, 30.0, 0.0), (-45.0, 30.0, 0.0)],
                                'dock1': [(-5.0, -90.0, -90.0), (5.0, -90.0, -90.0)],
                                'dock2': [(40.0, -45.0, 0.0), (40.0, -55.0, 0.0)],
                                'dock3': [(40.0, 55.0, 0.0), (40, 45.0, 0.0)],
                                'dock4': [(-5.0, 90.0, 90.0), (5.0, 90.0, 90.0)]}

        self.wait_positions = {'hangar1': [(-15.0, -35.0, 0.0), (-15.0, -25.0, 0.0)], 'hangar2': [(-15.0, -5.0, 0.0), (-15.0, 5.0, 0.0)],
                                'hangar3': [(-15.0, 25.0, 0.0), (-15.0, 35.0, 0.0)], 'dock1': [(-5.0, -80.0, -90.0), (5.0, -80.0, -90.0)],
                                'dock2': [(30.0, -45.0, 0.0), (30.0, -55.0, 0.0)], 'dock3': [(30.0, 55.0, 0.0), (30.0, 45.0, 0.0)],
                                'dock4':[(-5.0, 80.0, 90.0), (5.0, 80.0, 90.0)]}

        self.entrance_positions = {'hangar1': (-25.0, -30.0, 0.0), 'hangar2': (-25.0, 0.0, 0.0), 'hangar3': (-25.0, 30.0, 0.0)}

        self.exit_positions = {'hangar1': (-15.0, -30.0, 0.0), 'hangar2': (-15.0, 0.0, 0.0), 'hangar3': (-15.0, 30.0, 0.0)}

        self.occupancy = {'hangar1': False, 'hangar2': False, 'hangar3': False, 'dock1': [False, False], 
                            'dock2': [False, False], 'dock3': [False, False], 'dock4': [False, False]}
        
        self.wait_occupancy = {'hangar1': [False, False], 'hangar2': [False, False], 
                                'hangar3': [False, False], 'dock1': [False, False], 
                                'dock2': [False, False], 'dock3': [False, False],
                                'dock4': [False, False]}

        self.agvs = ['agv_1', 'agv_2', 'agv_3', 'agv_4', 'agv_5', 'agv_6']

        self.agv_status = {'agv_1': 'idle', 'agv_2': 'idle', 'agv_3': 'idle', 'agv_4': 'idle', 'agv_5': 'idle', 'agv_6': 'idle'}

        self.agv_tasks = {'agv_1': None, 'agv_2': None, 'agv_3': None, 'agv_4': None, 'agv_5': None, 'agv_6': None}

        self.agv_tasks_count = {'agv_1': 0, 'agv_2': 0, 'agv_3': 0, 'agv_4': 0, 'agv_5': 0, 'agv_6': 0}

        self.poses = {'agv_1': [40.0, 40.0, 0.0], 'agv_2': [60.0, 60.0, 0.0], 'agv_3': [-15.0, -12.0, 0.0], 'agv_4': [-20.0, 15.0, 0.0], 
                        'agv_5': [-20.0, -45.0, 0.0], 'agv_6': [-30.0, -40.0, 0.0]}

        self.task_id = 1

        self.tasks = []

        self.started_tasks = []

        self.finished_tasks = []

        self.generated_tasks = []

        self.stop_time = 5

        self.lock = threading.Lock()

    def task_allocation(self, rate):
        position_keys = list(self.positions.keys())
        hangar_list = [key for key in position_keys if 'hangar' in key]
        dock_list = [key for key in position_keys if 'dock' in key]
        # Ramdomly choose a hangar and a dock
        while self.run:
            hangar = random.choice(hangar_list)
            dock = random.choice(dock_list)
            task_to_shuffle = [(hangar, random.randint(0, 2)), (dock, random.randint(0, 1))]
            random.shuffle(task_to_shuffle)
            task_to_shuffle.insert(0, self.task_id)
            self.tasks.append(task_to_shuffle)
            self.generated_tasks.append(task_to_shuffle)
            self.task_id += 1
            time.sleep(rate)

    def check_hangar_occupancy(self, hangar):
        self.lock.acquire()
        occupancy = self.occupancy[hangar]
        self.lock.release()
        if not occupancy:
            return True
        else:
            free_occupancy = []
            i = 0
            self.lock.acquire()
            wait_occupancy = self.wait_occupancy[hangar]
            self.lock.release()
            for occupancy in wait_occupancy:
                if not occupancy:
                    free_occupancy.append(i)
                i += 1
            return free_occupancy

    def get_task(self):
        if len(self.tasks):
            task = self.tasks.pop(0)
            return task
        else:
            return None

    def calc_dist(self, pos1, pos2, deg1=False, deg2=False):
        if type(pos1) == list or type(pos1) == tuple:
            pos1=list(pos1)
            if deg1:
                pos1[2] = math.radians(pos1[2])
        else:
            if deg1:
                pos1 = math.radians(pos1)
        if type(pos2) == list or type(pos2) == tuple:
            pos2=list(pos2)
            if deg2:
                pos2[2] = math.radians(pos2[2])
        else:
            if deg2:
                pos2 = math.radians(pos2)
        return np.linalg.norm(np.array(pos1) - np.array(pos2))

    def estimate_motion_time(self, **kwargs):
        pose = kwargs.get('pose')
        target_1 = kwargs.get('target_1')
        target_2 = kwargs.get('target_2')

        if pose and target_1:
            target_1_position = self.positions[target_1[0]][target_1[1]]
            target_1 = [target_1_position[0], target_1_position[1], math.radians(target_1_position[2])]
            dist = self.calc_dist(pose, target_1, deg2=True)
        elif pose and target_2:
            target_2_position = self.positions[target_2[0]][target_2[1]]
            target_2 = [target_2_position[0], target_2_position[1], math.radians(target_2_position[2])]
            dist = self.calc_dist(pose, target_2, deg2=True)
        elif target_1 and target_2:
            target_1_position = self.positions[target_1[0]][target_1[1]]
            target_2_position = self.positions[target_2[0]][target_2[1]]
            target_1 = [target_1_position[0], target_1_position[1], math.radians(target_1_position[2])]
            target_2 = [target_2_position[0], target_2_position[1], math.radians(target_2_position[2])]
            dist = self.calc_dist(target_1, target_2, deg1=True, deg2=True)
        motion_time = dist/2.0
        return motion_time

    def choose_agv(self, task):
        motion_time = {}
        for agv in self.agvs:
            if self.agv_status[agv] == 'idle':
                agv_instance = self.get_agv_instance(agv)
                motion_time[agv] = self.estimate_motion_time(pose=agv_instance.pose, target_1=task[1])
        if len(motion_time):
            chosen_agv = min(motion_time, key=motion_time.get)
            return chosen_agv
        else:
            return None

    def hangar_task(self, agv, hangar, position):
        agv_instance = self.get_agv_instance(agv)
        occupancy = self.check_hangar_occupancy(hangar)
        if occupancy == True:
            pass
        else:
            dist = []
            for position in occupancy:
                dist.append(self.calc_dist(agv_instance.pose, self.wait_positions[hangar][position], deg2=True))
            position = dist.index(min(dist))
            target = self.wait_positions[hangar][position]
            x, y, theta = target
            self.call_mpc(agv_instance, x, y, theta, tolerance=0.5)
            while self.calc_dist(agv_instance.pose, target, deg2=True) > 1.5:
                time.sleep(0.5)
            while self.run:
                self.lock.acquire()
                occupancy = self.occupancy[hangar]
                self.lock.release()
                if not occupancy:
                    break
                time.sleep(0.5)
        self.lock.acquire()
        self.occupancy[hangar] = True
        self.lock.release()
        target = self.entrance_positions[hangar]
        x, y, theta = target
        self.call_mpc(agv_instance, x, y, theta, 1.0)
        while self.calc_dist(agv_instance.pose, target, deg2=True) > 2.0:
            time.sleep(0.5)
        target = self.positions[hangar][position]
        x, y, theta = target
        self.call_mpc(agv_instance, x, y, theta, 0.5)
        while self.calc_dist(agv_instance.pose, target, deg2=True) > 1.5:
            time.sleep(0.5)
        time.sleep(5)
        
    def hangar_exit(self, agv, hangar):
        agv_instance = self.get_agv_instance(agv)
        target = self.exit_positions[hangar]
        x, y, theta = target
        self.call_mpc(agv_instance, x, y, theta, 1.0)
        while self.calc_dist(agv_instance.pose, self.exit_positions[hangar], deg2=True) > 2.0:
            time.sleep(0.5)
        self.lock.acquire()
        self.occupancy[hangar] = False
        self.lock.release()

    def dock_task(self, agv, dock, position):
        agv_instance = self.get_agv_instance(agv)
        self.lock.acquire()
        occupancy = self.occupancy[dock][position]
        self.lock.release()
        if occupancy:
            self.lock.acquire()
            self.wait_occupancy[dock][position] = True
            self.lock.release()
            target = self.wait_positions[dock][position]
            x, y, theta = target
            self.call_mpc(agv_instance, x, y, theta, 0.5)
            while self.calc_dist(agv_instance.pose, target, deg2=True) > 1.5:
                self.lock.acquire()
                occupancy = self.occupancy[dock][position]
                self.lock.release()
                if not occupancy:
                    break
                time.sleep(0.5)
            while self.run:
                self.lock.acquire()
                occupancy = self.occupancy[dock][position]
                self.lock.release()
                if not occupancy:
                    break
                time.sleep(0.5)
            self.lock.acquire()
            self.occupancy[dock][position] = True
            self.wait_occupancy[dock][position] = False
            self.lock.release()
        self.lock.acquire()
        self.occupancy[dock][position] = True
        self.lock.release()
        target = self.positions[dock][position]
        x, y, theta = target
        self.call_mpc(agv_instance, x, y, theta, 0.5)
        while self.calc_dist(agv_instance.pose, target, deg2=True) > 1.5:
            time.sleep(0.5)
        time.sleep(5)
        
    def dock_exit(self, agv, dock, position):
        time.sleep(5)
        self.lock.acquire()
        self.occupancy[dock][position] = False
        self.lock.release()
        print(f'liberando vaga {dock}, {position}')
        print(self.occupancy[dock][position])
        print('dock exit finished')

    def assign_task(self, agv, task):
        self.lock.acquire()
        self.agv_status[agv] = 'busy'
        self.lock.release()
        t = threading.Thread(target=self.start_task, args=(agv, task))
        t.start()

    def start_task(self, agv, task):

        print(f'{agv.upper()} starting task {task}')
        self.started_tasks.append(task)
        self.agv_tasks[agv] = task
        agv_instance = self.get_agv_instance(agv)

        if 'hangar' in task[1][0]:
            self.hangar_task(agv, task[1][0], task[1][1])
            self.hangar_exit(agv, task[1][0])
        elif 'dock' in task[1][0]:
            self.dock_task(agv, task[1][0], task[1][1])
            threading.Thread(target=self.dock_exit, args=(agv, task[1][0], task[1][1])).start()

        if 'hangar' in task[2][0]:
            self.hangar_task(agv, task[2][0], task[2][1])
            self.hangar_exit(agv, task[2][0])
        elif 'dock' in task[2][0]:
            self.dock_task(agv, task[2][0], task[2][1])
            threading.Thread(target=self.dock_exit, args=(agv, task[2][0], task[2][1])).start()
            self.call_mpc(agv_instance, 0.0, 0.0, math.degrees(agv_instance.pose[2]), 1.0)
        self.lock.acquire()
        self.agv_status[agv] = 'idle'
        self.lock.release()


        print(f'{agv.upper()} finished task {task}')
        self.started_tasks.remove(task)
        self.finished_tasks.append(task)
        self.call_mpc(agv_instance, 0.0, 0.0, 0.0, 1.0)
        self.agv_tasks[agv] = None
        self.agv_status[agv] = 'idle'

    def get_agv_instance(self, agv_id):
        for agv_instance in self.agv_instances:
            if agv_instance.agv_id == agv_id:
                return agv_instance

    def call_mpc(self, agv_instance, x, y, theta, tolerance=None):
        agv_instance.call_mpc('move', float(x), float(y), float(theta), tolerance)

    def show_tasks(self):
        while self.run:
            print(f'Task queue: {self.tasks}')
            print(f'Started tasks: {self.started_tasks}')
            print(f'Finished tasks: {self.finished_tasks}')
            time.sleep(30)

    def task_loop(self):
        while self.run:
            task = self.get_task()
            if task:
                agv = self.choose_agv(task)
                if agv == None:
                    self.tasks.insert(0, task)
                else:
                    self.assign_task(agv, task)
            time.sleep(1)

    def start_scenario(self):
        t = []
        self.run = True
        t.append(threading.Thread(target=self.task_allocation, args=(30,)))
        t.append(threading.Thread(target=self.task_loop))

        for thread in t:
            thread.start()

    def stop_scenario(self):
        self.run = False
        for agv_instance in self.agv_instances:
            agv_instance.call_mpc('stop', 0.0, 0.0, 0.0)
