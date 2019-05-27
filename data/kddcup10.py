import numpy as np
import random
import gzip as gz


class Kddcup:
    def __init__(self, filename):
        self.num_dim = 120
        self.num_points = 494021
        self.xy = []
        self.num_assign_points = 0
        self.num_train_points = 0
        self.num_test_points = 0
        self.train_data = 0
        self.test_data = 0
        self.test_label = 0

        outlier = 0
        with gz.open(filename, 'rt') as fin:
            for line in fin:
                line = line.strip()
                field = line.split(',')
                p = []
                for i in range(0, len(field)):
                    p.append(float(field[i]))
                outlier = outlier + p[len(p) - 1]
                self.xy.append(p)
        ratio = outlier / len(self.xy)
        print("all: %d; outlier: %d; ratio: %g" % (len(self.xy), outlier, ratio))

    def get_clean_training_testing_data(self, ratio):
        self.num_assign_points = int(self.num_points * ratio)
        self.num_test_points = self.num_points - self.num_assign_points
        random.shuffle(self.xy)
        self.num_train_points = 0
        for i in range(self.num_assign_points):
            pl = float(self.xy[i][120])
            if pl < 1:
                self.num_train_points = self.num_train_points + 1
        self.train_data = np.zeros([self.num_train_points, 120], dtype=np.float64)
        self.test_data = np.zeros([self.num_test_points, 120], dtype=np.float64)
        self.test_label = np.zeros([self.num_test_points, 1], dtype=np.float64)
        head = 0
        for i in range(self.num_assign_points):
            pl = float(self.xy[i][120])
            if pl < 1:
                for j in range(120):
                    self.train_data[head, j] = float(self.xy[i][j])
                head = head + 1
        for i in range(self.num_test_points):
            ni = self.num_assign_points + i
            for j in range(120):
                self.test_data[i, j] = float(self.xy[ni][j])
            self.test_label[i, 0] = float(self.xy[ni][120])
        return self.train_data, self.test_data, self.test_label, self.num_test_points

    def get_contaminated_training_testing_data(self, ratio, contamination):
        adjusted_ratio = contamination / (1.0 - contamination)
        self.num_assign_points = int(self.num_points * ratio)
        self.num_test_points = self.num_points - self.num_assign_points
        random.shuffle(self.xy)
        self.num_train_points = 0
        for i in range(self.num_assign_points):
            pl = float(self.xy[i][120])
            if pl < 1:
                self.num_train_points = self.num_train_points + 1
        num_contaminated_points = int(self.num_train_points * adjusted_ratio)
        self.num_train_points = num_contaminated_points + self.num_train_points
        self.train_data = np.zeros([self.num_train_points, 120], dtype=np.float64)
        self.test_data = np.zeros([self.num_test_points, 120], dtype=np.float64)
        self.test_label = np.zeros([self.num_test_points, 1], dtype=np.float64)
        head = 0
        contaminated_cnt = 0
        for i in range(self.num_assign_points):
            pl = float(self.xy[i][120])
            if pl < 1:
                for j in range(120):
                    self.train_data[head, j] = float(self.xy[i][j])
                head = head + 1
            elif contaminated_cnt < num_contaminated_points:
                for j in range(120):
                    self.train_data[head, j] = float(self.xy[i][j])
                head = head + 1
                contaminated_cnt = contaminated_cnt + 1
        #print contaminated_cnt, num_contaminated_points, head, self.num_train_points
        for i in range(self.num_test_points):
            ni = self.num_assign_points + i
            for j in range(120):
                self.test_data[i, j] = float(self.xy[ni][j])
            self.test_label[i, 0] = float(self.xy[ni][120])
        return self.train_data, self.test_data, self.test_label, self.num_test_points

if __name__ == '__main__':
    kddcup10 = Kddcup('../demo/kddcup99-10.data.pp.csv.gz')

