import csv
import sys

import numpy as np
import time, random
import matplotlib.pyplot as plt


class StopNode(object):
    def __init__(self, trip_id, stop_id, time, index, time_string):
        self.stop_id = stop_id
        self.trip_id = trip_id
        self.time = time
        self.visited = False
        self.neighbors = set()
        self.shared_trips = []
        self.weight = float("inf")
        self.index = index
        self.time_string = time_string


class Approxer(object):
    def elapsed_time(self,time_string):
        time_array = time_string.split(':')
        seconds = int(time_array[0])*3600 + int(time_array[1])*60 + int(time_array[2])
        return seconds

    def learn(self,count):

        self.firstline = True
        c = 1
        with open('dublin_dataset/stop_times.txt', newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for row in spamreader:
                if c % 100000 == 0:
                    print('line: ',c)
                if self.firstline is True:
                    self.firstline = False
                    continue
                elif c == count:
                    break
                else:
                    if row[3] in self.stop_dictionary:
                        self.stop_dictionary[row[3]][row[0]] = self.index
                    else:
                        self.stop_dictionary[row[3]] = {}
                        self.stop_dictionary[row[3]][row[0]] = self.index

                    if row[0] in self.trip_dictionary:
                        self.trip_dictionary[row[0]][row[3]] = self.index
                    else:
                        self.trip_dictionary[row[0]] = {}
                        self.trip_dictionary[row[0]][row[3]] = self.index

                    self.nodes[self.index] = StopNode(row[0], row[3], self.elapsed_time(time_string=row[1]), self.index, row[1])
                    c = c+1
                self.index = self.index + 1

        for trip_id, stop_dict in self.trip_dictionary.items():
            self.trips_key[trip_id] = list(stop_dict)[0] + list(stop_dict)[-1]

        # print(self.trips_key)

        for key, value in self.trips_key.items():
            if value in self.equivalent_trips:
                self.equivalent_trips[value].append(key)
            else:
                self.equivalent_trips[value] = []
                self.equivalent_trips[value].append(key)
        # print(self.trip_dictionary)
        count = 1
        start = time.time()
        first_iter = True
        last_index = 0
        for trip_id, stop_dict in self.trip_dictionary.items():
            for stop_id1, index1 in stop_dict.items():
                if first_iter is True:
                    last_index = index1
                    first_iter = False
                else:
                    if last_index != index1 and self.nodes[last_index].time <= self.nodes[index1].time:
                        self.nodes[last_index].neighbors.add(index1)
                    last_index = index1


        for stop_id, trips_dict in self.stop_dictionary.items():
            for trip_id1, index1 in trips_dict.items():
                for trip_id2, index2 in trips_dict.items():
                    if index1 != index2:
                        string_key = self.trips_key[trip_id2]
                        eq_trips = self.equivalent_trips[string_key]
                        min_time = float('inf')
                        min_index = None
                        for eq_trip in eq_trips:
                            if eq_trip in self.stop_dictionary[stop_id]:
                                eq_trip_index = self.stop_dictionary[stop_id][eq_trip]
                                if self.nodes[eq_trip_index].time >= self.nodes[index1].time:
                                    if self.nodes[eq_trip_index].time < min_time:
                                        min_time = self.nodes[eq_trip_index].time
                                        min_index = eq_trip_index
                        if min_index is not None:
                            if min_index != index1:
                                self.nodes[index1].neighbors.add(min_index)



        np.save('out',self.nodes)
        print('time: ', time.time() - start)
        # print(self.nodes)


    def load(self):
        # print('loading...')
        start = time.time()
        self.nodes = np.load('out.npy').item()
        # print('loaded! in: ',time.time() - start,' seconds')

    def __init__(self):
        self.stop_dictionary = {}
        self.trip_dictionary = {}
        self.trips_key = {}
        self.equivalent_trips = {}
        self.firstline = True
        self.stops = {}
        self.useless_stops = []
        self.nodes = {}
        self.first_iter = True
        self.min_weight = float('inf')
        self.min_index = None
        self.last_node = None
        self.index = 1

    def approx_time(self,stop1, stop2, relaxed):
        self.nodes[stop1].visited = True
        self.min_weight = float('inf')
        # print('index: ',self.nodes[stop1].index,'\ttime: ',self.nodes[stop1].time_string)
        # print('trip_id: ',self.nodes[stop1].trip_id,'\t stop_id: ',self.nodes[stop1].stop_id)
        # print('neighbors: ',self.nodes[stop1].neighbors)
        # print('weight: ',self.nodes[stop1].weight)
        # print('< ========== >')
        # print(self.nodes[stop1].weight + self.nodes[stop1].time)
        # print(self.nodes[stop2].time)
        if self.nodes[stop1].stop_id == self.nodes[stop2].stop_id:
            return self.nodes[stop1].weight
        else:
            neighbors = self.nodes[stop1].neighbors
            for node_x in neighbors:
                if self.first_iter is True:
                    temp = self.nodes[node_x].time - self.nodes[stop1].time
                    if temp < self.nodes[node_x].weight:
                        self.nodes[node_x].weight = temp
                else:
                    temp = self.nodes[node_x].time - self.nodes[stop1].time\
                                                + self.nodes[stop1].weight
                    if temp < self.nodes[node_x].weight:
                        self.nodes[node_x].weight = temp
            self.first_iter = False
            for i in neighbors:
                relaxed.add(i)
            for relax in relaxed:
                if self.nodes[relax].weight < self.min_weight and self.nodes[relax].visited is False:
                    # print(value.visited)
                    # print('min index is: ',value.index)
                    self.min_weight = self.nodes[relax].weight
                    self.min_index = self.nodes[relax].index
        if self.min_index is None or self.nodes[self.min_index].time > self.nodes[stop2].time or len(neighbors) == 0 :
            return False
        return self.approx_time(self.min_index,stop2,relaxed)


if __name__ == '__main__':
    sys.setrecursionlimit(8000)
    # approxer = Approxer()
    # approxer.learn(10000)
    # approxer = Approxer()
    # approxer.load()
    # print(approxer.approx_time(1, 30,set([])))

    counter = 1
    learned = []
    tested = []
    while True:
        # print(counter)
        approxer = Approxer()
        approxer.load()
        x1 = random.randint(1, 9000)
        x2 = random.randint(1, 9000)
        # if approxer.nodes[x1].time < approxer.nodes[x2].time and approxer.nodes[x2].time - approxer.nodes[x1].time < 30000:
        # print('testing for nodes: ',x1,' ',x2)
        t = approxer.approx_time(x1, x2, set([]))
        if not t:
            # print('baited')
            continue
        l = approxer.nodes[x2].time - approxer.nodes[x1].time
        tested.append(t)
        learned.append(l)
        counter = counter + 1
        print(counter)
        if counter == 50:
            break
        # time.sleep(0.2)

    # t = np.arange(0., 51., 1)
    plt.plot(learned,color='darkblue',marker='^',linestyle=' ')
    plt.plot(learned, color='slateblue',linestyle='dotted')
    plt.plot(tested,color='red',linestyle=' ',marker='v')
    plt.plot(tested,color='indianred',linestyle='dashed')
    plt.savefig('results.pdf')
    # plt.show()


